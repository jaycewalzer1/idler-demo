"""No model or GPU is needed to check exact-token response compatibility."""

import copy
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from dealroom.learning_serve import (
    _expected_model_key,
    complete_logprob_tokens,
    install_compatibility,
    policy_record,
)


class Tokenizer:
    def decode(self, ids):
        assert len(ids) == 1
        return {10: " hello", 11: "!", 12: "<|eot_id|>"}.get(ids[0], f"token-{ids[0]}")


def packet():
    return {
        "id": "official-response",
        "choices": [{
            "message": {"content": " hello!"},
            "finish_reason": "stop",
            "logprobs": {"content": [{"id": 10, "logprob": -0.5}, {"id": 11, "logprob": -1.5}]},
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def arguments(**changes):
    args = SimpleNamespace(
        model=SimpleNamespace(model="default_model", adapter=None, draft="default_model"),
        sampling=SimpleNamespace(
            temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, xtc_probability=0.0,
        ),
        logits=SimpleNamespace(
            logit_bias=None, repetition_penalty=0.0, presence_penalty=0.0, frequency_penalty=0.0,
        ),
        logprobs=True,
        top_logprobs=-1,
    )
    for key, value in changes.items():
        owner, field = key.split(".") if "." in key else (None, key)
        setattr(getattr(args, owner) if owner else args, field, value)
    return args


def fake_upstream(barrier=None):
    class Generator:
        def __init__(self):
            self.model_provider = SimpleNamespace(
                tokenizer=Tokenizer(), draft_model=None,
                model_key=("/weights/base", None, None),
                _model_map={"default_model": "/weights/base"},
                _adapter_map={"default_model": "/cli/adapter"},
                _draft_model_map={"default_model": None},
            )

        def generate(self, request, generation_args, progress_callback=None):
            context = SimpleNamespace(prompt=request, stop=lambda: None)
            if barrier:
                barrier.wait()
            return context, iter(["official generation iterator"])

    class Handler:
        def __init__(self, generator):
            self.response_generator = generator
            self.stream = False

        def generate_response(
            self, text, finish_reason, prompt_token_count=None,
            completion_token_count=None, prompt_cache_count=None,
            token_logprobs=None, top_tokens=None, tokens=None,
            tool_calls=None, reasoning_text=None,
        ):
            result = packet()
            result["choices"][0]["message"]["content"] = text
            result["choices"][0]["finish_reason"] = finish_reason
            result["usage"] = {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            }
            if token_logprobs:
                result["choices"][0]["logprobs"]["content"] = [
                    {"id": token, "logprob": probability}
                    for token, probability in zip(tokens, token_logprobs, strict=True)
                ]
            else:
                result["choices"][0].pop("logprobs")
            return result

    return SimpleNamespace(ResponseGenerator=Generator, APIHandler=Handler)


def test_missing_token_strings_are_completed_without_mutating_source():
    original = packet()
    result = complete_logprob_tokens(original, Tokenizer())
    assert result["choices"][0]["logprobs"]["content"] == [
        {"id": 10, "logprob": -0.5, "token": " hello", "top_logprobs": []},
        {"id": 11, "logprob": -1.5, "token": "!", "top_logprobs": []},
    ]
    assert "token" not in original["choices"][0]["logprobs"]["content"][0]
    assert result["usage"] == original["usage"]
    assert result["choices"][0]["message"] == original["choices"][0]["message"]


def test_actual_sampled_ids_take_precedence_over_top_ranked_ids():
    original = packet()
    top = [{"id": 99, "token": "wrong sample", "logprob": -0.01}]
    original["choices"][0]["logprobs"]["content"][0] = dict(top[0], top_logprobs=top)
    result = complete_logprob_tokens(
        original, Tokenizer(), generated_token_ids=[10, 12], generated_token_logprobs=[-0.5, -1.5],
    )
    first, last = result["choices"][0]["logprobs"]["content"]
    assert first == {"id": 10, "token": " hello", "logprob": -0.5, "top_logprobs": top}
    assert last["id"] == 12 and last["token"] == "<|eot_id|>"


def test_policy_preserves_exact_ids_including_noncanonical_tokenization():
    response = complete_logprob_tokens(packet(), Tokenizer())
    record = policy_record(
        response, prompt_token_ids=[0, 7, 7], generated_token_ids=[10, 11],
        generated_token_logprobs=[-0.5, -1.5], policy_fingerprint="sha256:checkpoint",
    )
    assert record == {
        "schema_version": 1, "prompt_token_ids": [0, 7, 7], "generated_token_ids": [10, 11],
        "generated_token_logprobs": [-0.5, -1.5], "policy_fingerprint": "sha256:checkpoint",
        "logprob_basis": "sampling_distribution", "temperature": 1.0,
    }


@pytest.mark.parametrize("change", [
    {"prompt_token_ids": [0, 7]},
    {"generated_token_ids": [10]},
    {"generated_token_ids": [10, 12]},
    {"generated_token_ids": [True, 11]},
    {"generated_token_logprobs": [-0.5, float("nan")]},
    {"generated_token_logprobs": [-0.5, float("-inf")]},
    {"generated_token_logprobs": [-0.5, 0.1]},
    {"policy_fingerprint": ""},
])
def test_policy_rejects_unverifiable_arrays(change):
    kwargs = dict(
        prompt_token_ids=[0, 7, 7], generated_token_ids=[10, 11],
        generated_token_logprobs=[-0.5, -1.5], policy_fingerprint="sha256:checkpoint",
    )
    kwargs.update(change)
    with pytest.raises(ValueError):
        policy_record(complete_logprob_tokens(packet(), Tokenizer()), **kwargs)


def respond(server, prompt, args=None, *, provider_changes=None, token_count=2):
    generator = server.ResponseGenerator()
    if provider_changes:
        for key, value in provider_changes.items():
            setattr(generator.model_provider, key, value)
    context, responses = generator.generate(prompt, args or arguments())
    assert context.prompt == prompt
    assert list(responses) == ["official generation iterator"]
    handler = server.APIHandler(generator)
    return handler.generate_response(
        " hello!", "stop", len(prompt), token_count,
        token_logprobs=[-0.5, -1.5], tokens=[10, 11],
    )


def test_wrapper_keeps_upstream_generation_and_publishes_exact_provenance(monkeypatch):
    monkeypatch.setenv("DEALROOM_POLICY_FINGERPRINT", "policy-123")
    server = fake_upstream()
    install_compatibility(server)
    wrapped = server.APIHandler.generate_response
    install_compatibility(server)
    assert server.APIHandler.generate_response is wrapped
    response = respond(server, [100, 1, 100])
    assert response["dealroom_policy"]["prompt_token_ids"] == [100, 1, 100]
    assert response["dealroom_server"] == {
        "schema_version": 1, "policy_fingerprint": "policy-123",
        "model_path": "/weights/base", "adapter_path": None, "draft_model_path": None,
    }
    assert response["choices"][0]["logprobs"]["content"][0]["token"] == " hello"


@pytest.mark.parametrize("changes", [
    {"sampling.temperature": 0.0}, {"sampling.temperature": 0.7},
    {"sampling.top_p": 0.9}, {"sampling.top_k": 20}, {"sampling.min_p": 0.1},
    {"sampling.xtc_probability": 0.1}, {"logits.logit_bias": {5: 2.0}},
    {"logits.repetition_penalty": 1.1}, {"logits.presence_penalty": 0.1},
    {"logits.frequency_penalty": 0.1}, {"top_logprobs": 1}, {"logprobs": False},
])
def test_nonidentity_sampler_never_claims_sampling_distribution(monkeypatch, changes):
    monkeypatch.setenv("DEALROOM_POLICY_FINGERPRINT", "policy-123")
    server = fake_upstream()
    install_compatibility(server)
    response = respond(server, [1, 2, 3], arguments(**changes))
    assert "dealroom_policy" not in response
    assert response["choices"][0]["message"]["content"] == " hello!"


def test_missing_fingerprint_and_speculation_have_no_policy_record(monkeypatch):
    monkeypatch.delenv("DEALROOM_POLICY_FINGERPRINT", raising=False)
    server = fake_upstream()
    install_compatibility(server)
    assert "dealroom_policy" not in respond(server, [1, 2, 3])
    monkeypatch.setenv("DEALROOM_POLICY_FINGERPRINT", "policy-123")
    response = respond(server, [1, 2, 3], provider_changes={"draft_model": object()})
    assert "dealroom_policy" not in response


def test_usage_mismatch_keeps_evaluation_readable_but_rejects_policy(monkeypatch):
    monkeypatch.setenv("DEALROOM_POLICY_FINGERPRINT", "policy-123")
    server = fake_upstream()
    install_compatibility(server)
    response = respond(server, [1, 2, 3], token_count=3)
    assert "dealroom_policy" not in response
    assert "usage" in response["dealroom_policy_error"]
    assert response["choices"][0]["logprobs"]["content"][0]["id"] == 10


def test_exact_prompts_are_isolated_across_simultaneous_http_threads(monkeypatch):
    monkeypatch.setenv("DEALROOM_POLICY_FINGERPRINT", "policy-123")
    server = fake_upstream(Barrier(2))
    install_compatibility(server)
    prompts = [[111, 1], [222, 2, 2, 2]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda prompt: respond(server, prompt), prompts))
    assert [response["dealroom_policy"]["prompt_token_ids"] for response in responses] == prompts


def test_adapter_provenance_requires_explicit_body_adapter(monkeypatch):
    monkeypatch.setenv("DEALROOM_POLICY_FINGERPRINT", "policy-adapter")
    server = fake_upstream()
    provider = server.ResponseGenerator().model_provider
    assert _expected_model_key(provider, arguments().model) == ("/weights/base", None, None)
    args = arguments(**{"model.adapter": "/explicit/adapter"})
    assert _expected_model_key(provider, args.model) == ("/weights/base", "/explicit/adapter", None)
    install_compatibility(server)
    response = respond(
        server, [1, 2, 3], args,
        provider_changes={"model_key": ("/weights/base", "/explicit/adapter", None)},
    )
    assert response["dealroom_server"]["adapter_path"] == "/explicit/adapter"
    with pytest.raises(RuntimeError, match="Loaded model changed"):
        respond(server, [1, 2, 3], args)


def test_policy_rejects_modified_usage_or_sampled_probabilities():
    response = complete_logprob_tokens(packet(), Tokenizer())
    for change in ("usage", "sample"):
        bad = copy.deepcopy(response)
        if change == "usage":
            bad["usage"]["total_tokens"] += 1
        else:
            bad["choices"][0]["logprobs"]["content"][0]["logprob"] = -0.25
        with pytest.raises(ValueError):
            policy_record(
                bad, prompt_token_ids=[1, 2, 3], generated_token_ids=[10, 11],
                generated_token_logprobs=[-0.5, -1.5], policy_fingerprint="sha256:checkpoint",
            )


def test_tiny_cpu_official_server_inspect_sdk_and_policy_extractor(tmp_path):
    """Random tiny fixture: protocol validation, never model capability evidence.

    Run in training/.venv to exercise MLX; the normal environment skips it.
    Only tokenizer files from the local download are reused. The 8B weights
    are never loaded: a one-layer, eight-dimensional random model runs on CPU.
    """
    import dataclasses
    import json
    import os
    import shutil
    import socket
    import subprocess
    import sys
    import time
    import urllib.request
    from pathlib import Path

    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("openai")
    mx.set_default_device(mx.cpu)
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.log import read_eval_log
    from inspect_ai.model import GenerateConfig, get_model
    from inspect_ai.solver import generate
    from mlx.utils import tree_flatten
    from mlx_lm.models.llama import Model, ModelArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import load_tokenizer

    from dealroom.learning_policy import extract_policy_segments

    root = Path(__file__).resolve().parents[1]
    source = root / "training/models/llama3.1-8b-4bit"
    if not (source / "tokenizer.json").exists():
        pytest.skip("Local downloaded tokenizer is required for the tiny CPU protocol fixture")
    model_path, adapter_path = tmp_path / "tiny-random-fixture", tmp_path / "tiny-adapter"
    model_path.mkdir()
    adapter_path.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copyfile(source / name, model_path / name)
    config = ModelArgs(
        model_type="llama", hidden_size=8, num_hidden_layers=1, intermediate_size=16,
        num_attention_heads=2, num_key_value_heads=2, rms_norm_eps=1e-5,
        vocab_size=128256, max_position_embeddings=1024, tie_word_embeddings=True,
    )
    mx.random.seed(13)
    model = Model(config)
    mx.save_safetensors(str(model_path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (model_path / "config.json").write_text(json.dumps(dataclasses.asdict(config)))
    model.freeze()
    lora = {"rank": 2, "scale": 1.0, "dropout": 0.0, "keys": ["self_attn.q_proj"]}
    linear_to_lora_layers(model, 1, lora)
    mx.save_safetensors(
        str(adapter_path / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())),
    )
    (adapter_path / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": 1, "lora_parameters": lora,
    }))
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = dict(os.environ, DEALROOM_POLICY_FINGERPRINT="tiny-fixture-adapter-fingerprint")
    command = [
        sys.executable, "-c",
        "import mlx.core as mx; mx.set_default_device(mx.cpu); "
        # The upstream server assumes device_info() describes Metal whenever
        # Metal exists, even when CPU was explicitly selected. Disable only
        # that device-availability probe in this CPU-only fixture process.
        "mx.metal.is_available = lambda: False; "
        "from dealroom.learning_serve import main; main()",
        "--model", str(model_path), "--adapter-path", str(adapter_path),
        "--host", "127.0.0.1", "--port", str(port),
        "--decode-concurrency", "1", "--prompt-concurrency", "1",
    ]
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 25
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=.5) as health:
                    assert json.load(health) == {"status": "ok"}
                break
            except OSError:
                if process.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail(f"Tiny CPU server did not start: {log_path.read_text()}")
                time.sleep(.05)
        client = get_model(
            "openai-api/mlx/default_model", base_url=f"http://127.0.0.1:{port}/v1",
            api_key="local-test-fixture", memoize=False,
            config=GenerateConfig(
                max_tokens=2, temperature=1.0, top_p=1.0, logprobs=True, max_retries=0,
                extra_body={"top_k": 0, "min_p": 0.0, "adapters": str(adapter_path)},
            ),
        )
        logs = inspect_eval(
            Task(dataset=[Sample(input="This is a protocol test fixture. Say hello.")],
                 solver=generate(), name="tiny_cpu_protocol_fixture"),
            model=client, log_dir=str(tmp_path / "inspect"), display="none", log_model_api=True,
        )
        sample = read_eval_log(logs[0].location, resolve_attachments="full").samples[0]
        assert sample.error is None, (sample.error, log_path.read_text())
        event = next(item for item in sample.events if item.event == "model")
        response = event.call.response
        assert response["dealroom_server"]["model_path"] == str(model_path)
        assert response["dealroom_server"]["adapter_path"] == str(adapter_path)
        assert response["dealroom_server"]["policy_fingerprint"] == env["DEALROOM_POLICY_FINGERPRINT"]
        policy = response["dealroom_policy"]
        assert len(policy["prompt_token_ids"]) == response["usage"]["prompt_tokens"]
        assert 1 <= len(policy["generated_token_ids"]) <= 2
        tokenizer = load_tokenizer(model_path)
        content = response["choices"][0]["logprobs"]["content"]
        assert [item["id"] for item in content] == policy["generated_token_ids"]
        assert [item["logprob"] for item in content] == policy["generated_token_logprobs"]
        assert [item["token"] for item in content] == [
            tokenizer.decode([token]) for token in policy["generated_token_ids"]
        ]
        assert all(item["top_logprobs"] == [] for item in content)
        segments = extract_policy_segments(
            sample,
            lambda messages: tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, tools=None,
            ),
            expected_policy_fingerprint=env["DEALROOM_POLICY_FINGERPRINT"],
        )
        assert len(segments) == 1
        assert segments[0]["prompt_ids"] == policy["prompt_token_ids"]
        assert segments[0]["completion_ids"] == policy["generated_token_ids"]
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
