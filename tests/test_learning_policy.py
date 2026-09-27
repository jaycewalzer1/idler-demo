"""Policy masks, trace integrity, and per-trajectory gradient weighting."""

import copy
import math

import pytest

from dealroom.curriculum import _case_and_witness
from dealroom.domain import reset, step
from dealroom.learning_policy import (
    MAX_SHAPING,
    extract_policy_segments,
    group_relative_advantages,
    make_policy_segments,
    semantic_potential,
    training_reward,
)


def _native_sample():
    # Native Inspect ModelEvent -> ModelCall request/response schema, including
    # the raw OpenAI extra preserved by the instrumented MLX serving adapter.
    return {
        "id": "sample-a",
        "events": [
            {"event": "tool", "function": "read", "result": "Tool text is context only"},
            {
                "event": "model",
                "uuid": "event-a",
                "pending": None,
                "error": None,
                "input": [{"role": "user", "content": "Logical input before tool emulation"}],
                "call": {
                    "request": {
                        "messages": [
                            {"role": "system", "content": "Exact wire tool schema"},
                            {"role": "user", "content": "Exact input"},
                        ]
                    },
                    "response": {
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": "Text is not retokenized"},
                            }
                        ],
                        "dealroom_policy": {
                            "schema_version": 1,
                            "prompt_token_ids": [11, 12, 13],
                            "generated_token_ids": [201, 202],
                            "generated_token_logprobs": [-0.4, -1.2],
                            "policy_fingerprint": "checkpoint-a",
                            "logprob_basis": "sampling_distribution",
                            "temperature": 1.0,
                        },
                    },
                },
            },
        ],
    }


def _segment(ids, prompt=(11, 12)):
    return {
        "prompt_ids": list(prompt),
        "completion_ids": list(ids),
        "policy_fingerprint": "checkpoint-a",
        "behavior_logprobs": [-0.5] * len(ids),
    }


def _trajectories():
    return [
        {
            "trajectory_id": "success",
            "case_id": "same-case",
            "status": "success",
            "reward": 1.0,
            "segments": [_segment([2, 3]), _segment([4])],
        },
        {
            "trajectory_id": "failure",
            "case_id": "same-case",
            "status": "failure",
            "reward": 0.0,
            "segments": [_segment([5, 6, 7, 8])],
        },
        {"trajectory_id": "cutoff", "case_id": "same-case", "status": "limit", "reward": None},
    ]


def test_extract_uses_wire_prompt_and_actual_sampled_ids_not_retokens():
    sample = _native_sample()
    seen = []

    def tokenizer(messages):
        seen.append(messages)
        return [11, 12, 13]

    segments = extract_policy_segments(
        sample, tokenizer, expected_policy_fingerprint="checkpoint-a"
    )
    assert seen == [sample["events"][1]["call"]["request"]["messages"]]
    assert segments[0]["completion_ids"] == [201, 202]
    assert segments[0]["behavior_logprobs"] == [-0.4, -1.2]
    assert segments[0]["model_event_index"] == 1
    assert segments[0]["prompt_messages"] != sample["events"][1]["input"]
    segments[0]["prompt_messages"][0]["content"] = "changed copy"
    assert (
        sample["events"][1]["call"]["request"]["messages"][0]["content"] == "Exact wire tool schema"
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda e: e["call"]["response"].pop("dealroom_policy"), "sampled token IDs"),
        (
            lambda e: e["call"]["request"]["messages"][0].update(content="attachment://unresolved"),
            "resolved wire prompt",
        ),
        (lambda e: e["call"]["response"]["usage"].update(prompt_tokens=4), "token usage"),
        (lambda e: e["call"]["response"]["usage"].update(completion_tokens=1), "token usage"),
        (
            lambda e: e["call"]["response"]["dealroom_policy"].update(
                generated_token_ids=[201, True]
            ),
            "integer token IDs",
        ),
        (
            lambda e: e["call"]["response"]["dealroom_policy"].update(
                generated_token_logprobs=[-0.4]
            ),
            "one finite",
        ),
        (
            lambda e: e["call"]["response"]["dealroom_policy"].update(
                generated_token_logprobs=[float("nan"), -1]
            ),
            "one finite",
        ),
        (
            lambda e: e["call"]["response"]["dealroom_policy"].update(
                generated_token_logprobs=[0.2, -1]
            ),
            "one finite",
        ),
        (
            lambda e: e["call"]["response"]["dealroom_policy"].update(temperature=0.7),
            "temperature-one",
        ),
        (
            lambda e: e["call"]["response"]["dealroom_policy"].update(logprob_basis="model_logits"),
            "sampling distribution",
        ),
        (lambda e: e["call"]["response"]["choices"][0].update(finish_reason=None), "unfinished"),
        (lambda e: e.update(error="request failed"), "errored"),
    ],
)
def test_invalid_or_unverifiable_traces_are_refused(mutation, match):
    sample = _native_sample()
    mutation(sample["events"][1])
    with pytest.raises(ValueError, match=match):
        extract_policy_segments(sample, lambda messages: [11, 12, 13])


def test_prompt_identity_checks_ids_not_only_count_and_policy_match():
    sample = _native_sample()
    with pytest.raises(ValueError, match="differ from the exact tokenizer"):
        extract_policy_segments(sample, lambda messages: [11, 12, 99])
    with pytest.raises(ValueError, match="checkpoint"):
        extract_policy_segments(
            sample, lambda messages: [11, 12, 13], expected_policy_fingerprint="other"
        )


def test_length_stopped_generation_can_belong_to_a_completed_scored_trajectory():
    sample = _native_sample()
    sample["events"][1]["call"]["response"]["choices"][0]["finish_reason"] = "length"
    assert len(extract_policy_segments(sample, lambda messages: [11, 12, 13])) == 1


def test_population_standardized_advantages_exclude_cutoffs():
    result = group_relative_advantages(_trajectories())
    assert result["scored_count"] == 2
    assert result["excluded"] == {"limit": 1}
    assert result["mean_reward"] == 0.5
    assert result["reward_population_std"] == 0.5
    assert result["advantages"] == {"success": 1.0, "failure": -1.0}


@pytest.mark.parametrize("status,reward", [("failure", 0.0), ("success", 1.0)])
def test_equal_reward_groups_do_not_create_an_update(status, reward):
    trajectories = _trajectories()[:2]
    for trajectory in trajectories:
        trajectory.update(status=status, reward=reward)
    result = make_policy_segments(trajectories)
    assert result["segments"] == []
    assert result["statistics"]["skip_reason"] == "equal_rewards"
    assert result["statistics"]["advantages"] == {}


def test_group_with_one_scored_episode_is_excluded_without_fabricated_reward():
    trajectories = _trajectories()
    trajectories[1].update(status="error", reward=None)
    result = make_policy_segments(trajectories)
    assert result["segments"] == []
    assert result["statistics"]["skip_reason"] == "insufficient_scored_trajectories"
    assert result["statistics"]["scored_count"] == 1


def test_accumulated_segment_gradient_matches_mean_trajectory_actor_objective():
    result = make_policy_segments(_trajectories())
    segments = result["segments"]
    assert result["statistics"]["gradient_accumulation_steps"] == 3
    assert result["statistics"]["optimizer_updates"] == 1
    assert [segment["weight"] for segment in segments] == [0.5, 0.5, -0.375]
    # Derivatives of arbitrary actor-token log probabilities with respect to a
    # hypothetical model parameter; prompt/tool values deliberately huge.
    actor_derivatives = {2: 2.0, 3: -1.0, 4: 3.0, 5: -2.0, 6: 4.0, 7: 1.0, 8: 5.0, 11: 1e9, 12: 1e9}
    accumulated_gradient = sum(
        -segment["weight"] * sum(actor_derivatives[token] for token in segment["completion_ids"])
        for segment in segments
    ) / len(segments)
    expected_gradient = (-1.0 * (2 - 1 + 3) / 3 - (-1.0) * (-2 + 4 + 1 + 5) / 4) / 2
    assert accumulated_gradient == pytest.approx(expected_gradient)
    assert sum(segment["weight"] * len(segment["completion_ids"]) for segment in segments) == 0


def test_mixed_policies_or_cases_and_missing_actor_segments_are_refused():
    trajectories = _trajectories()
    trajectories[1]["segments"][0]["policy_fingerprint"] = "other"
    with pytest.raises(ValueError, match="mix sampled checkpoints"):
        make_policy_segments(trajectories)
    trajectories = _trajectories()
    trajectories[1]["case_id"] = "other"
    with pytest.raises(ValueError, match="exactly one case"):
        make_policy_segments(trajectories)
    trajectories = _trajectories()
    trajectories[1]["segments"] = []
    with pytest.raises(ValueError, match="complete sampled actor segments"):
        make_policy_segments(trajectories)
    trajectories = _trajectories()
    trajectories[1]["reward"] = 1
    with pytest.raises(ValueError, match="disagrees with status"):
        make_policy_segments(trajectories)


def test_binary_rewards_and_unscored_exclusion_apply_before_shaping():
    for status in ("limit", "error", "incomplete", "pending", "running"):
        assert training_reward(status, None, shaping=True) is None
    assert training_reward("success", 1) == 1
    assert training_reward("failure", 0) == 0
    with pytest.raises(ValueError, match="disagrees"):
        training_reward("success", 0)
    with pytest.raises(ValueError, match="binary"):
        training_reward("success", float("nan"))


def test_shaping_only_rewards_semantic_progress_and_remains_bounded():
    case, witness, _ = _case_and_witness("train", 2, "credit_execution", 2)
    initial = reset(case)
    state = copy.deepcopy(initial)
    initial_potential = semantic_potential(initial)
    for _ in range(20):
        step(state, {"type": "read", "resource": "summary"})
        step(state, {"type": "bogus", "prose": "I succeeded perfectly"})
    assert semantic_potential(state) == initial_potential
    potentials = []
    for action in witness:
        step(state, action)
        potentials.append(semantic_potential(state))
    assert max(potentials) > initial_potential
    assert all(0 <= value <= MAX_SHAPING for value in potentials)
    reward = training_reward("success", 1, initial, state, shaping=True)
    assert reward == pytest.approx(1 + semantic_potential(state) - initial_potential)
    assert 0.8 <= reward <= 1.2
    with pytest.raises(ValueError, match="training split"):
        training_reward("success", 1, initial, state, shaping=True, split="validation")
    with pytest.raises(ValueError, match="training split"):
        training_reward("success", 1, initial, state, shaping=True, split="sealed_test")


def test_partial_progress_can_distinguish_failed_training_rollouts_without_changing_terminal_score():
    case, witness, _ = _case_and_witness("train", 2, "credit_execution", 2)
    initial = reset(case)
    partial = reset(case)
    for action in witness:
        if action["type"] == "set_disposition":
            break
        step(partial, action)
    reward = training_reward("failure", 0, initial, partial, shaping=True)
    assert 0 < reward <= MAX_SHAPING
    assert training_reward("failure", 0, initial, partial, shaping=False) == 0
    group = _trajectories()[:2]
    group[0].update(status="failure", reward=reward)
    group[1].update(status="failure", reward=0.0)
    assert group_relative_advantages(group)["advantages"] == {"success": 1.0, "failure": -1.0}
    assert math.isfinite(reward)
