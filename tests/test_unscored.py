"""Canonical Inspect grades distinguish incomplete computation from task failure."""

import asyncio
import json
import math

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import read_eval_log
from inspect_ai.model import get_model
from inspect_ai.solver import solver
from inspect_ai.util import time_limit

from dealroom.demo import tool_output
from dealroom.domain import load_case
from dealroom.task import dealroom, sample_for
from dealroom.witnesses import WITNESSES, wait


def mock_model(actions):
    return get_model(
        "mockllm/model",
        custom_outputs=[tool_output(action, index) for index, action in enumerate(actions)],
        memoize=False,
    )


def assert_unscored_log(log, reason):
    sample = log.samples[0]
    score = sample.scores["transaction_success"]
    assert math.isnan(score.value)
    assert score.reason == reason and score.answer == "unscored"
    assert "not a terminal reward" in score.explanation
    assert score.metadata["diagnostics"]
    restored = read_eval_log(log.location)
    assert math.isnan(restored.samples[0].scores["transaction_success"].value)
    assert restored.samples[0].scores["transaction_success"].reason == reason
    aggregate = restored.results.scores[0]
    assert aggregate.scored_samples == 0 and aggregate.unscored_samples == 1
    assert math.isnan(aggregate.metrics["mean"].value)


@pytest.mark.parametrize("kind", ["token", "turn", "message", "time"])
def test_framework_limits_remain_unscored_in_canonical_inspect_logs(tmp_path, kind):
    task = dealroom(case_id="case-02")
    options = {f"{kind}_limit": 1 if kind != "message" else 2}
    model = mock_model([{"type": "read"}] * 3)
    if kind == "time":

        async def delayed_output(messages, tools, tool_choice, config):
            await asyncio.sleep(2)
            return tool_output({"type": "read"}, 0)

        model = get_model("mockllm/model", custom_outputs=delayed_output, memoize=False)
    log = inspect_eval(task, model=model, log_dir=str(tmp_path), display="none", **options)[0]
    assert log.samples[0].error is None
    assert log.samples[0].limit.type == kind
    assert_unscored_log(log, "sample_limit")


def test_incomplete_solver_is_unscored_without_a_framework_limit(tmp_path):
    @solver
    def no_actions():
        async def solve(state, generate):
            return state

        return solve

    task = dealroom(case_id="case-02")
    task.solver = no_actions()
    log = inspect_eval(task, model=mock_model([]), log_dir=str(tmp_path), display="none")[0]
    assert log.samples[0].error is None and log.samples[0].limit is None
    assert_unscored_log(log, "incomplete")


def test_domain_budget_is_excluded_from_mean_and_final_allowed_finish_is_scored(tmp_path):
    action_limit = len(WITNESSES["case-02"])
    task = dealroom(case_id="case-02", action_limit=action_limit)
    task.dataset = MemoryDataset(
        [sample_for(case_id, action_limit=action_limit) for case_id in ("case-02", "case-06")]
    )

    def outputs(messages, tools, tool_choice, config):
        case_id = json.loads(next(message.text for message in messages if message.role == "user"))[
            "case_id"
        ]
        index = sum(message.role == "assistant" for message in messages)
        actions = WITNESSES[case_id] if case_id == "case-02" else [{"type": "read"}] * action_limit
        return tool_output(actions[index], index)

    model = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
    log = inspect_eval(task, model=model, log_dir=str(tmp_path), display="none", max_samples=2)[0]
    samples = {sample.id: sample for sample in log.samples}
    success = samples["case-02"].scores["transaction_success"]
    unfinished = samples["case-06"].scores["transaction_success"]
    assert all(sample.error is None and sample.limit is None for sample in samples.values())
    assert success.value == 1 and success.metadata["finished"]
    assert success.metadata["budget_exhausted"]
    assert len(samples["case-02"].metadata["dealroom"]["actions"]) == action_limit
    assert math.isnan(unfinished.value) and unfinished.reason == "budget_exhausted"
    assert not unfinished.metadata["finished"] and unfinished.metadata["budget_exhausted"]
    aggregate = log.results.scores[0]
    assert aggregate.scored_samples == 1 and aggregate.unscored_samples == 1
    assert aggregate.metrics["mean"].value == 1
    restored = read_eval_log(log.location)
    assert restored.results.scores[0].metrics["mean"].value == 1
    assert math.isnan(
        next(sample for sample in restored.samples if sample.id == "case-06")
        .scores["transaction_success"]
        .value
    )


@pytest.mark.parametrize("framework_limit", [False, True])
def test_exact_final_action_expiry_is_scored_unless_processing_hits_a_framework_limit(
    tmp_path, framework_limit
):
    task = dealroom(case_id="case-02", action_limit=1, end_on_expiry=True)
    if framework_limit:
        original_solver = task.solver

        @solver
        def expire_then_exceed_time():
            async def solve(state, generate):
                state = await original_solver(state, generate)
                # An actual public limit records a SampleLimitEvent and propagates
                # to Inspect, after the genuine expired snapshot already exists.
                with time_limit(1):
                    await asyncio.sleep(2)
                return state

            return solve

        task.solver = expire_then_exceed_time()
    log = inspect_eval(
        task,
        model=mock_model([wait(load_case("case-02").deadline.isoformat())]),
        log_dir=str(tmp_path),
        display="none",
    )[0]
    sample = log.samples[0]
    score = sample.scores["transaction_success"]
    assert sample.error is None
    assert score.metadata["outcome"] == "failure"
    assert score.metadata["budget_exhausted"] and not score.metadata["finished"]
    assert sample.metadata["dealroom"]["termination_reason"] == "contract_deadline_expired"
    if framework_limit:
        assert sample.limit.type == "time"
        assert_unscored_log(log, "sample_limit")
    else:
        assert sample.limit is None
        assert score.value == 0 and score.answer == "failure"
        assert log.results.scores[0].scored_samples == 1
        assert log.results.scores[0].unscored_samples == 0
