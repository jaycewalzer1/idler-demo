"""Small response compatibility shim for the official ``mlx_lm.server`` CLI.

MLX-LM 0.31.3 returns sampled token IDs and log probabilities, but omits the
``token`` string required by OpenAI-compatible clients when top_logprobs is off.
This module only annotates those responses; model loading, tokenization,
generation, sampling, caching, HTTP handling, and CLI parsing remain upstream.

For policy training, exact prompt IDs come from upstream GenerationContext,
never from re-tokenizing the text. Set DEALROOM_POLICY_FINGERPRINT to the
verified base/adapter identity and request non-streaming logprobs with the
unfiltered temperature-1 sampler. No policy record is emitted for other
sampling settings, speculative decoding, or top_logprobs > 0.
"""

from __future__ import annotations

import copy
import functools
import inspect
import math
import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

SHIM_VERSION = 1


def _token_ids(values: Any, label: str) -> list[int]:
    if not isinstance(values, (list, tuple)) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in values
    ):
        raise ValueError(f"{label} must contain exact non-negative integer token IDs")
    return list(values)


def _logprobs(values: Any) -> list[float]:
    if not isinstance(values, (list, tuple)) or not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value <= 0
        for value in values
    ):
        raise ValueError("Generated log probabilities must be finite and non-positive")
    return list(values)


def complete_logprob_tokens(
    response: dict[str, Any],
    tokenizer: Any,
    *,
    generated_token_ids: list[int] | None = None,
    generated_token_logprobs: list[float] | None = None,
) -> dict[str, Any]:
    """Return an annotated copy without changing sampled IDs or probabilities.

    Explicit sampled arrays take precedence over upstream's top-token branch,
    which otherwise places the top-ranked token in the sampled-token position.
    RL rejects that top-token mode independently in ``_policy_sampling``.
    """
    result = copy.deepcopy(response)
    choices = result.get("choices", [])
    if not choices:
        return result
    content = choices[0].get("logprobs", {}).get("content", [])
    if generated_token_logprobs:
        ids = _token_ids(generated_token_ids, "Generated tokens")
        probabilities = _logprobs(generated_token_logprobs)
        if len(ids) != len(probabilities):
            raise ValueError("Sampled token IDs and log probabilities have different lengths")
        if content and len(content) != len(ids):
            raise ValueError("Response log probabilities do not match the sampled token count")
        content = [
            dict(content[index] if content else {}, id=token, logprob=probability)
            for index, (token, probability) in enumerate(zip(ids, probabilities, strict=True))
        ]
        choices[0]["logprobs"] = {"content": content}
    for entry in content:
        token_id = _token_ids([entry.get("id")], "Response tokens")[0]
        # Decode the sampled ID itself, including stop/control tokens. Text can
        # be empty for a special token; its identity still remains in `id`.
        entry["token"] = tokenizer.decode([token_id])
        entry.setdefault("top_logprobs", [])
    return result


def policy_record(
    response: dict[str, Any],
    *,
    prompt_token_ids: list[int] | tuple[int, ...],
    generated_token_ids: list[int],
    generated_token_logprobs: list[float],
    policy_fingerprint: str,
) -> dict[str, Any]:
    """Validate exact upstream token arrays and build the rollout record.

    The caller must first establish the identity temperature-1 sampler. MLX
    reports log-softmax probabilities before sampling; these equal sampling
    probabilities only with all sampling filters and logits modifiers off.
    """
    prompt = _token_ids(prompt_token_ids, "Prompt tokens")
    generated = _token_ids(generated_token_ids, "Generated tokens")
    probabilities = _logprobs(generated_token_logprobs)
    if not isinstance(policy_fingerprint, str) or not policy_fingerprint.strip():
        raise ValueError("A verified policy fingerprint is required")
    if not prompt or not generated or len(generated) != len(probabilities):
        raise ValueError("Policy records require aligned nonempty prompt and generated tokens")
    usage = response.get("usage", {})
    expected = {
        "prompt_tokens": len(prompt),
        "completion_tokens": len(generated),
        "total_tokens": len(prompt) + len(generated),
    }
    if any(type(usage.get(key)) is not int or usage[key] != value for key, value in expected.items()):
        raise ValueError("Exact policy token IDs do not agree with upstream response usage")
    content = response.get("choices", [{}])[0].get("logprobs", {}).get("content", [])
    if len(content) != len(generated) or any(
        item.get("id") != token or item.get("logprob") != probability
        for item, token, probability in zip(content, generated, probabilities, strict=True)
    ):
        raise ValueError("Exact policy tokens do not agree with sampled response log probabilities")
    return {
        "schema_version": SHIM_VERSION,
        "prompt_token_ids": prompt,
        "generated_token_ids": generated,
        "generated_token_logprobs": probabilities,
        "policy_fingerprint": policy_fingerprint,
        "logprob_basis": "sampling_distribution",
        "temperature": 1.0,
    }


def _policy_sampling(args: Any) -> bool:
    sampling = args.sampling
    logits = args.logits
    return (
        args.logprobs is True
        and args.top_logprobs in (-1, 0)
        and sampling.temperature == 1.0
        and sampling.top_p == 1.0
        and sampling.top_k == 0
        and sampling.min_p == 0.0
        and sampling.xtc_probability == 0.0
        and not logits.logit_bias
        and logits.repetition_penalty == 0.0
        and logits.presence_penalty == 0.0
        and logits.frequency_penalty == 0.0
    )


@dataclass(frozen=True)
class _GenerationRecord:
    prompt: tuple[int, ...]
    tokenizer: Any
    eligible_sampling: bool
    policy_fingerprint: str | None
    model_key: tuple[str | None, str | None, str | None]


def _expected_model_key(provider: Any, model: Any) -> tuple[str | None, str | None, str | None]:
    # Match installed ModelProvider.load's resolution order exactly. In 0.31.3
    # the CLI adapter alias is not applied after default_model resolves to a
    # path. Adapted requests must explicitly pass their `adapters` body field.
    path = provider._model_map.get(model.model, model.model)
    adapter = provider._adapter_map.get(path, model.adapter)
    draft = provider._draft_model_map.get(model.draft, model.draft)
    return path, adapter, draft


# generate() waits for the upstream worker, then returns on the HTTP request
# thread. A ContextVar isolates simultaneous requests; no global last-prompt
# cache can accidentally mix one request's tokens into another response.
_generation: ContextVar[_GenerationRecord | None] = ContextVar("dealroom_generation", default=None)


def install_compatibility(server: Any = None) -> None:
    """Annotate the installed upstream server in place; safe to call twice."""
    if server is None:
        from mlx_lm import server

    if getattr(server, "_dealroom_response_shim", False):
        return
    original_generate = server.ResponseGenerator.generate
    original_response = server.APIHandler.generate_response
    signature = inspect.signature(original_response)
    if not {"tokens", "token_logprobs", "prompt_token_count", "completion_token_count"}.issubset(
        signature.parameters
    ):
        raise RuntimeError("Installed MLX server response API is incompatible with this shim")

    @functools.wraps(original_generate)
    def generate(self: Any, request: Any, generation_args: Any, *args: Any, **kwargs: Any) -> Any:
        _generation.set(None)
        context, responses = original_generate(self, request, generation_args, *args, **kwargs)
        provider = self.model_provider
        if provider.model_key != _expected_model_key(provider, generation_args.model):
            context.stop()
            raise RuntimeError("Loaded model changed before exact response provenance was captured")
        _generation.set(_GenerationRecord(
            prompt=tuple(_token_ids(context.prompt, "Upstream prompt")),
            tokenizer=provider.tokenizer,
            eligible_sampling=(
                _policy_sampling(generation_args)
                and provider.draft_model is None
            ),
            policy_fingerprint=os.environ.get("DEALROOM_POLICY_FINGERPRINT"),
            model_key=tuple(provider.model_key),
        ))
        return context, responses

    @functools.wraps(original_response)
    def generate_response(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        response = original_response(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        values = bound.arguments
        record = _generation.get()
        tokenizer = record.tokenizer if record else self.response_generator.model_provider.tokenizer
        response = complete_logprob_tokens(
            response,
            tokenizer,
            generated_token_ids=values.get("tokens"),
            generated_token_logprobs=values.get("token_logprobs"),
        )
        if record:
            response["dealroom_server"] = {
                "schema_version": SHIM_VERSION,
                "policy_fingerprint": record.policy_fingerprint,
                "model_path": record.model_key[0],
                "adapter_path": record.model_key[1],
                "draft_model_path": record.model_key[2],
            }
        if not self.stream and record and record.eligible_sampling and record.policy_fingerprint:
            try:
                response["dealroom_policy"] = policy_record(
                    response,
                    prompt_token_ids=record.prompt,
                    generated_token_ids=values.get("tokens"),
                    generated_token_logprobs=values.get("token_logprobs"),
                    policy_fingerprint=record.policy_fingerprint,
                )
            except ValueError as error:
                # Generic evaluation remains readable. The policy extractor
                # must reject a response without a valid dealroom_policy.
                response["dealroom_policy_error"] = str(error)
        if not self.stream or values.get("finish_reason") is not None:
            _generation.set(None)
        return response

    server.ResponseGenerator.generate = generate
    server.APIHandler.generate_response = generate_response
    server._dealroom_response_shim = True


def main() -> None:
    """Pass all CLI arguments to the official MLX-LM server."""
    from mlx_lm import server

    install_compatibility(server)
    server.main()


if __name__ == "__main__":
    main()
