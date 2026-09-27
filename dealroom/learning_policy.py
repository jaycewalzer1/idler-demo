"""Auditable group-relative REINFORCE data preparation, without model training.

Population-standardized group rewards provide an outcome baseline. This module
does not implement PPO or GRPO: there is no clipped importance ratio, critic, KL
penalty, or multi-epoch rollout reuse. One freshly sampled group supports exactly
one gradient update under its recorded policy. Callers must enforce that policy
fingerprint when loading the training checkpoint.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from dealroom.domain import (
    CreditTerms,
    State,
    authority_for,
    current_authority,
    eligible_cost,
    revision_signers,
)

SCORED_STATUSES = frozenset({"success", "failure"})
UNSCORED_STATUSES = frozenset({"limit", "error", "incomplete", "pending", "running"})
MAX_SHAPING = 0.2


def semantic_potential(state: State) -> float:
    """A [0,.2] training potential computed from actual transaction records.

    No read, request, tool-call, prose, or invalid-action count earns a bonus. Equal
    records yield equal potential even after arbitrary repetitions. This optional
    potential is not the binary evaluator and makes no policy-invariance claim:
    terminal potentials are not forced to zero, so partial progress can be learned.
    """
    cost = eligible_cost(state)
    authority = current_authority(state)
    proceed = "proceed" in state.case.acceptable_dispositions
    cancel = "cancel" in state.case.acceptable_dispositions
    evidence = cost is not None
    authority_ready = bool(
        authority
        and (
            (cancel and "cancel" in authority.permissions)
            or (
                proceed
                and "proceed" in authority.permissions
                and (
                    cost is not None
                    and (
                        cost <= authority.maximum_residual_cents
                        or (
                            "credit" in authority.permissions
                            and authority.maximum_credit_cents
                            >= cost - authority.maximum_residual_cents
                        )
                    )
                )
            )
        )
    )
    viable_revisions = []
    for revision in state.revisions.values():
        if not isinstance(revision.terms, CreditTerms) or cost is None:
            continue
        historical_authority = current_authority(state, revision.created_at)
        if (
            historical_authority is not None
            and historical_authority.id == revision.authority_id
            and "credit" in historical_authority.permissions
            and historical_authority.minimum_credit_cents
            <= revision.terms.amount_cents
            <= historical_authority.maximum_credit_cents
            and max(0, cost - revision.terms.amount_cents)
            <= historical_authority.maximum_residual_cents
            and revision.terms.amount_cents <= state.case.counterpart.seller_max_credit_cents
            and set(state.case.counterpart.required_evidence_ids).issubset(revision.evidence_ids)
            and set(revision.evidence_ids).issubset(state.documents)
        ):
            viable_revisions.append(revision)
    drafted = proceed and bool(viable_revisions)
    executed = proceed and any(
        revision.id in state.executed_at
        and set(revision.required_signers).issubset(revision_signers(state, revision.id))
        for revision in viable_revisions
    )
    extension = state.deadline > state.case.deadline
    disposition = state.disposition
    committed = bool(
        disposition
        and disposition.kind in state.case.acceptable_dispositions
        and disposition.at <= disposition.deadline
        and (
            (disposition.kind == "cancel" and authority_for(state, "cancel"))
            or (
                disposition.kind == "proceed"
                and disposition.residual_cents is not None
                and (at_authority := current_authority(state, disposition.at))
                and disposition.residual_cents <= at_authority.maximum_residual_cents
            )
        )
    )
    value = (
        0.025 * evidence
        + 0.025 * authority_ready
        + 0.05 * drafted
        + 0.05 * executed
        + 0.025 * extension
        + 0.025 * committed
    )
    return min(MAX_SHAPING, max(0.0, float(value)))


def training_reward(
    status: str,
    terminal_reward: int | float | None,
    initial_state: State | None = None,
    final_state: State | None = None,
    *,
    shaping: bool = False,
    split: str = "train",
) -> float | None:
    """Return binary terminal reward plus optional bounded training-only shaping.

    Cutoffs, errors, and incomplete samples are excluded, never relabeled failures
    or granted progress reward. Evaluation metrics must use terminal_reward itself.
    For a scored trajectory, the shaping difference lies in [-.2,.2], preserving
    strict ordering between terminal successes and terminal failures.
    """
    if status in UNSCORED_STATUSES:
        return None
    if status not in SCORED_STATUSES:
        raise ValueError(f"Unknown trajectory status: {status}")
    expected = 1 if status == "success" else 0
    if isinstance(terminal_reward, bool) or terminal_reward not in (0, 1):
        raise ValueError("A scored trajectory requires a canonical numeric binary reward")
    if terminal_reward != expected:
        raise ValueError("Canonical terminal reward disagrees with trajectory status")
    if not shaping:
        return float(terminal_reward)
    if split != "train":
        raise ValueError("Progress shaping is restricted to the training split")
    if initial_state is None or final_state is None:
        raise ValueError("Shaping requires both initial and final engine states")
    if initial_state.case != final_state.case:
        raise ValueError("Shaping states belong to different fixtures")
    return (
        float(terminal_reward) + semantic_potential(final_state) - semantic_potential(initial_state)
    )


def _plain(value: Any) -> dict:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        raise ValueError("Expected a native Inspect event or JSON object")
    return value


def _has_attachment(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("attachment://")
    if isinstance(value, Mapping):
        return any(_has_attachment(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_attachment(item) for item in value)
    return False


def _token_ids(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in value
        )
    ):
        raise ValueError(f"{label} must contain actual nonnegative integer token IDs")
    return list(value)


def extract_policy_segments(
    sample: Any,
    tokenize_prompt: Callable[[list[dict]], Sequence[int]],
    *,
    expected_policy_fingerprint: str | None = None,
) -> list[dict]:
    """Extract exact actor tokens from fully resolved native Inspect model calls.

    Read disk logs with read_eval_log(resolve_attachments="full") first. The MLX
    serving shim records `call.response.dealroom_policy` using sampled IDs, not
    retokenized output strings. Existing Ollama logs do not have these traces and
    are intentionally refused. `tokenize_prompt(messages)` must apply the exact
    serving chat template including its generation prefix.

    Only an unmodified temperature-one categorical policy is supported for training;
    the trainer's log-softmax must describe this same policy. No prompt or tool token
    is an actor target, including when text tool responses are represented as users.
    """
    sample_dict = _plain(sample)
    segments = []
    for position, item in enumerate(sample_dict.get("events", [])):
        event = _plain(item)
        if event.get("event") != "model":
            continue
        if event.get("error") or event.get("pending"):
            raise ValueError("Cannot train from an errored or pending model response")
        call = event.get("call")
        if not isinstance(call, dict) or call.get("error"):
            raise ValueError("Native model API request and response are required")
        request, response = call.get("request"), call.get("response")
        if not isinstance(request, dict) or not isinstance(response, dict):
            raise ValueError("Native model API request and response are required")
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages or _has_attachment(messages):
            raise ValueError("Exact resolved wire prompt messages are required")
        trace = response.get("dealroom_policy")
        if not isinstance(trace, dict) or trace.get("schema_version") != 1:
            raise ValueError("Actual sampled token IDs and logprobs are missing")
        if trace.get("logprob_basis") != "sampling_distribution" or trace.get("temperature") != 1:
            raise ValueError(
                "Policy traces require the unmodified temperature-one sampling distribution"
            )
        policy_fingerprint = trace.get("policy_fingerprint")
        if not isinstance(policy_fingerprint, str) or not policy_fingerprint:
            raise ValueError("A sampled policy fingerprint is required")
        if (
            expected_policy_fingerprint is not None
            and policy_fingerprint != expected_policy_fingerprint
        ):
            raise ValueError("Rollout policy does not match the expected training checkpoint")
        prompt_ids = _token_ids(trace.get("prompt_token_ids"), "prompt_token_ids")
        generated_ids = _token_ids(trace.get("generated_token_ids"), "generated_token_ids")
        recreated_prompt_ids = _token_ids(
            list(tokenize_prompt(copy.deepcopy(messages))), "tokenized prompt"
        )
        if recreated_prompt_ids != prompt_ids:
            raise ValueError("Recorded prompt token IDs differ from the exact tokenizer template")
        logprobs = trace.get("generated_token_logprobs")
        if (
            not isinstance(logprobs, list)
            or len(logprobs) != len(generated_ids)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value > 1e-6
                for value in logprobs
            )
        ):
            raise ValueError("Each actual sampled token requires one finite nonpositive logprob")
        usage = response.get("usage", {})
        if usage.get("prompt_tokens") != len(prompt_ids) or usage.get("completion_tokens") != len(
            generated_ids
        ):
            raise ValueError("Native API token usage disagrees with the sampled policy trace")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Exactly one sampled completion per call is required")
        if choices[0].get("finish_reason") not in {"stop", "tool_calls", "length"}:
            raise ValueError("An unfinished or errored model completion is not a policy segment")
        segments.append(
            {
                "prompt_ids": prompt_ids,
                "completion_ids": generated_ids,
                "behavior_logprobs": [float(value) for value in logprobs],
                "prompt_messages": copy.deepcopy(messages),
                "policy_fingerprint": policy_fingerprint,
                "model_event_index": position,
                "model_event_uuid": event.get("uuid"),
            }
        )
    if not segments:
        raise ValueError("No sampled model responses were recorded")
    if len({segment["policy_fingerprint"] for segment in segments}) != 1:
        raise ValueError("A trajectory cannot mix sampled policies")
    return segments


def group_relative_advantages(trajectories: Sequence[Mapping[str, Any]]) -> dict:
    """Population-standardized advantages among scored rollouts of one case.

    Missing/limited/error rollouts never contribute a zero reward or a denominator.
    At least two scored outcomes with differing rewards are necessary. Every
    trajectory (including exclusions) must have a unique trajectory_id and case_id.
    """
    ids = [trajectory.get("trajectory_id") for trajectory in trajectories]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Trajectory IDs must be unique nonempty strings")
    case_ids = {trajectory.get("case_id") for trajectory in trajectories}
    if len(case_ids) != 1 or not isinstance(next(iter(case_ids)), str):
        raise ValueError("A policy group must contain rollouts of exactly one case")
    scored = []
    excluded: Counter[str] = Counter()
    for trajectory in trajectories:
        status = trajectory.get("status")
        if status in UNSCORED_STATUSES:
            excluded[status] += 1
            continue
        if status not in SCORED_STATUSES:
            raise ValueError(f"Unknown trajectory status: {status}")
        reward = trajectory.get("reward")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
        ):
            raise ValueError("Scored trajectories require finite numeric training rewards")
        if not -MAX_SHAPING - 1e-9 <= reward <= 1 + MAX_SHAPING + 1e-9:
            raise ValueError("Training reward exceeds binary reward plus bounded shaping")
        terminal_reward = 1 if status == "success" else 0
        if abs(reward - terminal_reward) > MAX_SHAPING + 1e-9:
            raise ValueError("Training reward disagrees with status and bounded shaping")
        scored.append(trajectory)
    mean = (
        math.fsum(trajectory["reward"] for trajectory in scored) / len(scored) if scored else None
    )
    variance = (
        math.fsum((trajectory["reward"] - mean) ** 2 for trajectory in scored) / len(scored)
        if scored
        else 0.0
    )
    std = math.sqrt(variance)
    skip_reason = (
        "insufficient_scored_trajectories"
        if len(scored) < 2
        else ("equal_rewards" if std <= 1e-12 else None)
    )
    return {
        "case_id": next(iter(case_ids)),
        "scored_count": len(scored),
        "excluded": dict(excluded),
        "mean_reward": mean,
        "reward_population_std": std,
        "skip_reason": skip_reason,
        "advantages": {}
        if skip_reason
        else {
            trajectory["trajectory_id"]: (trajectory["reward"] - mean) / std
            for trajectory in scored
        },
    }


def make_policy_segments(trajectories: Sequence[Mapping[str, Any]]) -> dict:
    """Weight batch-one segments for exactly one mean-accumulated group update.

    The trainer computes `weight * sum(actor_token_cross_entropy)` for each
    segment, accumulates all S gradients, divides by S, then updates once. Therefore
    weight = advantage * S / (N * trajectory_actor_tokens), giving the objective
    `mean_trajectories[-advantage * mean_actor_tokens(log_probability)]`.

    Only completion IDs are targets. Repeated prompt/tool context is context only;
    segment lengths and unequal trajectory lengths cannot change their relative
    contribution. Zero-advantage segments remain in S with zero loss.
    """
    statistics = group_relative_advantages(trajectories)
    if statistics["skip_reason"]:
        return {"segments": [], "statistics": statistics}
    selected = [
        trajectory for trajectory in trajectories if trajectory["status"] in SCORED_STATUSES
    ]
    segment_count = sum(len(trajectory.get("segments", [])) for trajectory in selected)
    output = []
    fingerprints = set()
    for trajectory in selected:
        segments = trajectory.get("segments")
        if not isinstance(segments, (list, tuple)) or not segments:
            raise ValueError("Every scored trajectory requires complete sampled actor segments")
        actor_counts = [
            len(_token_ids(segment.get("completion_ids"), "completion_ids")) for segment in segments
        ]
        token_count = sum(actor_counts)
        advantage = statistics["advantages"][trajectory["trajectory_id"]]
        for segment in segments:
            _token_ids(segment.get("prompt_ids"), "prompt_ids")
            fingerprint = segment.get("policy_fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError("Policy segments require their sampled checkpoint fingerprint")
            fingerprints.add(fingerprint)
            output.append(
                {
                    **copy.deepcopy(segment),
                    "weight": advantage
                    * segment_count
                    / (statistics["scored_count"] * token_count),
                    "advantage": advantage,
                    "trajectory_id": trajectory["trajectory_id"],
                    "case_id": trajectory["case_id"],
                    "trajectory_actor_tokens": token_count,
                }
            )
    if len(fingerprints) != 1:
        raise ValueError("One group update cannot mix sampled checkpoints")
    statistics = {
        **statistics,
        "segment_count": segment_count,
        "actor_tokens": sum(len(segment["completion_ids"]) for segment in output),
        "policy_fingerprint": next(iter(fingerprints)),
        "gradient_accumulation_steps": segment_count,
        "optimizer_updates": 1,
        "algorithm": "population-standardized group-relative REINFORCE",
    }
    return {"segments": output, "statistics": statistics}
