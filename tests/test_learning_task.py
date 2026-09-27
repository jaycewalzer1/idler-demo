"""Exercise the versioned learning adapter through the actual Inspect harness."""

import asyncio
import json
import math

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import read_eval_log
from inspect_ai.model import get_model
from inspect_ai.tool import ToolDef

from dealroom import learning_task
from dealroom.curriculum import load_manifest, training_witness
from dealroom.demo import tool_output
from dealroom.domain import load_case, observe, parse_action, reset
from dealroom.learning_task import (
    LEARNING_WORKFLOW_VERSION,
    effective_learning_case,
    learning_case_for_record,
    learning_dealroom,
    learning_prompt,
    learning_sample_for,
    learning_tools_for,
    public_learning_input,
    replay_learning_record,
)
from dealroom.score import evaluate
from dealroom.task import PROMPT, public_index, tools_for
from dealroom.witnesses import WITNESSES


@pytest.fixture
def learning_cases(monkeypatch):
    cases = {
        "opaque-a": load_case("case-02").model_copy(update={"id": "opaque-a"}),
        "opaque-b": load_case("case-06").model_copy(update={"id": "opaque-b", "split": "evaluation"}),
    }
    monkeypatch.setattr(learning_task, "load_learning_case", lambda case_id, **kwargs: cases[case_id])
    return cases


def run_actions(tmp_path, actions, **kwargs):
    model = get_model(
        "mockllm/model",
        custom_outputs=[tool_output(action, i) for i, action in enumerate(actions)],
        memoize=False,
    )
    return inspect_eval(
        learning_dealroom("opaque-a", **kwargs),
        model=model,
        log_dir=str(tmp_path),
        display="none",
    )[0]


def test_initial_input_is_public_and_identical_between_workflows(learning_cases):
    case = effective_learning_case(learning_cases["opaque-b"])
    original = public_learning_input(case, "original")
    clarified = public_learning_input(case, "clarified")
    assert original == clarified
    assert json.loads(clarified) == public_index(observe(reset(case)), include_policy=True)
    assert json.loads(clarified)["action_limit"] == 256
    for private_key in (
        "counterpart", "acceptable_dispositions", "prelude", "buyer_grants",
        "seller_max_credit_cents", "signature_minutes", "split", "pending_events",
    ):
        assert f'"{private_key}":' not in clarified
    assert learning_prompt("original", end_on_expiry=False) == PROMPT
    assert "finish only submits" in learning_prompt("clarified")
    assert "set_disposition is the tool that commits" in learning_prompt("clarified")


@pytest.mark.parametrize("workflow", ["original", "clarified"])
def test_real_inspect_witness_success_and_exact_replay(learning_cases, tmp_path, workflow):
    log = run_actions(tmp_path, WITNESSES["case-02"], workflow=workflow)
    sample = read_eval_log(log.location).samples[0]
    assert sample.error is None and sample.limit is None
    assert sample.scores["learning_success"].value == 1
    record = sample.metadata["dealroom"]
    assert record["workflow"] == workflow
    assert record["workflow_version"] == LEARNING_WORKFLOW_VERSION
    assert record["source_fingerprints"]["domain.py"]
    assert [parse_action(a) for a in record["actions"]] == [parse_action(a) for a in WITNESSES["case-02"]]
    replay = replay_learning_record(record)
    assert evaluate(replay).model_dump(mode="json") == sample.scores["learning_success"].metadata
    assert replay.events == record["events"]
    assert learning_case_for_record(record).action_limit == 256
    record["case_snapshot"]["title"] = "tampered after logging"
    with pytest.raises(ValueError, match="fingerprint"):
        replay_learning_record(record)


def test_concurrent_fresh_states_and_reviewer_metadata_not_in_model_messages(learning_cases, tmp_path):
    task = learning_dealroom("opaque-a")
    task.dataset = MemoryDataset([learning_sample_for(case) for case in learning_cases.values()])
    captured = []

    def outputs(messages, tools, tool_choice, config):
        initial = next(message.text for message in messages if message.role == "user")
        case_id = json.loads(initial)["case_id"]
        index = sum(message.role == "assistant" for message in messages)
        captured.append(json.dumps({
            "messages": [message.model_dump(mode="json") for message in messages],
            "tools": [tool.model_dump(mode="json") for tool in tools],
        }))
        assert len(tools) == 7
        assert config.parallel_tool_calls is False
        source = "case-02" if case_id == "opaque-a" else "case-06"
        return tool_output(WITNESSES[source][index], index)

    model = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
    log = inspect_eval(task, model=model, log_dir=str(tmp_path), display="none", max_samples=2)[0]
    samples = {sample.id: sample for sample in log.samples}
    assert len(samples) == 2
    assert all(sample.error is None and sample.scores["learning_success"].value == 1 for sample in samples.values())
    for case_id, source in (("opaque-a", "case-02"), ("opaque-b", "case-06")):
        assert [parse_action(a) for a in samples[case_id].metadata["dealroom"]["actions"]] == [
            parse_action(a) for a in WITNESSES[source]
        ]
    assert replay_learning_record(samples["opaque-b"].metadata["dealroom"]).disposition.kind == "cancel"
    text = "\n".join(captured)
    for forbidden in (
        "case_snapshot", "source_fingerprints", "fixture_sha256", "acceptable_dispositions",
        "buyer_grants", "signature_minutes", "seller_max_credit_cents", "pending_events",
    ):
        assert forbidden not in text


async def test_clarified_read_failure_returns_index_without_changing_transition(learning_cases):
    observations = {}
    records = {}
    for workflow in ("original", "clarified"):
        engine = reset(learning_cases["opaque-a"])
        record = {"actions": [], "events": []}
        tools, _ = learning_tools_for(engine, record, workflow)
        receipt = json.loads(await tools[0](resource="document", resource_id="missing"))
        assert receipt["accepted"] is False
        assert engine.attempts == 1 and engine.now == engine.case.start_at
        observations[workflow] = receipt.pop("observation")
        records[workflow] = (record, receipt)
    assert records["original"] == records["clarified"]
    assert "documents" in observations["original"]
    assert "documents" not in observations["clarified"]
    assert observations["clarified"] == public_index(observations["original"], include_policy=True)


async def test_valid_read_and_all_typed_parameters_are_unchanged(learning_cases):
    case = learning_cases["opaque-a"]
    original, original_finish = tools_for(reset(case), {"actions": [], "events": []})
    clarified, clarified_finish = learning_tools_for(reset(case), {"actions": [], "events": []})
    for before, after in zip(original + [original_finish], clarified + [clarified_finish], strict=True):
        assert ToolDef(before).parameters == ToolDef(after).parameters
        assert ToolDef(before).name == ToolDef(after).name
        assert ToolDef(after).parallel is False
        assert ToolDef(after).max_output == 0
    resource_id = case.documents[0].id
    assert json.loads(await original[0](resource="document", resource_id=resource_id)) == json.loads(
        await clarified[0](resource="document", resource_id=resource_id)
    )


def test_finish_does_not_commit_business_and_expiry_is_a_real_failure(learning_cases, tmp_path):
    claims_only = run_actions(tmp_path / "claims", [{
        "type": "finish", "claims": {"disposition": "proceed"}, "report": "Claimed completion.",
    }])
    sample = claims_only.samples[0]
    assert sample.error is None and sample.scores["learning_success"].value == 0
    assert replay_learning_record(sample.metadata["dealroom"]).disposition is None
    expired = run_actions(tmp_path / "expiry", [{
        "type": "wait_until", "at": learning_cases["opaque-a"].deadline.isoformat(),
    }], action_limit=1)
    sample = expired.samples[0]
    assert sample.error is None and sample.limit is None
    assert sample.scores["learning_success"].value == 0
    assert sample.metadata["dealroom"]["termination_reason"] == "contract_deadline_expired"


@pytest.mark.parametrize("kind", ["token", "turn", "message", "time"])
def test_framework_limits_unscored_in_native_logs(learning_cases, tmp_path, kind):
    task = learning_dealroom("opaque-a")
    setattr(task, f"{kind}_limit", 2 if kind == "message" else 1)
    model = get_model(
        "mockllm/model", custom_outputs=[tool_output({"type": "read"}, i) for i in range(3)],
        memoize=False,
    )
    if kind == "time":
        async def delayed_output(messages, tools, tool_choice, config):
            await asyncio.sleep(2)
            return tool_output({"type": "read"}, 0)

        model = get_model("mockllm/model", custom_outputs=delayed_output, memoize=False)
    log = inspect_eval(task, model=model, log_dir=str(tmp_path), display="none")[0]
    restored = read_eval_log(log.location)
    sample = restored.samples[0]
    assert sample.error is None and sample.limit.type == kind
    score = sample.scores["learning_success"]
    assert math.isnan(score.value) and score.reason == "sample_limit"
    assert restored.results.scores[0].scored_samples == 0


def test_generated_training_curriculum_runs_in_native_inspect(tmp_path):
    """Use train fixtures only; sealed test cases are not consumed by this test."""
    selected = {}
    for entry in load_manifest()["tasks"]:
        if entry["split"] == "train" and entry["stage"] == "coordination":
            selected.setdefault(entry["family"], entry["id"])
    witnesses = {case_id: training_witness(case_id) for case_id in selected.values()}
    tasks = [learning_dealroom(case_id) for case_id in selected.values()]
    task = tasks[0]
    task.dataset = MemoryDataset([item.dataset[0] for item in tasks])

    def outputs(messages, tools, tool_choice, config):
        case_id = json.loads(next(m.text for m in messages if m.role == "user"))["case_id"]
        index = sum(m.role == "assistant" for m in messages)
        return tool_output(witnesses[case_id][index], index)

    model = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
    log = inspect_eval(task, model=model, max_samples=2, log_dir=str(tmp_path), display="none")[0]
    assert len(log.samples) == len(selected) == 8
    assert all(sample.error is None and sample.scores["learning_success"].value == 1 for sample in log.samples)


def test_domain_budget_unscored_but_successful_last_action_scores(learning_cases, tmp_path):
    limited = run_actions(tmp_path / "limited", [{"type": "read"}], action_limit=1)
    sample = read_eval_log(limited.location).samples[0]
    assert sample.error is None and sample.limit is None
    assert math.isnan(sample.scores["learning_success"].value)
    assert sample.scores["learning_success"].reason == "budget_exhausted"
    succeeded = run_actions(
        tmp_path / "succeeded", WITNESSES["case-02"], action_limit=len(WITNESSES["case-02"])
    )
    sample = succeeded.samples[0]
    assert sample.error is None
    assert sample.scores["learning_success"].value == 1
    assert sample.scores["learning_success"].metadata["budget_exhausted"]


@pytest.mark.parametrize("kwargs", [
    {"workflow": "unknown"}, {"action_limit": 0}, {"action_limit": True},
    {"compaction_threshold": -1},
])
def test_reject_invalid_configuration(learning_cases, kwargs):
    with pytest.raises(ValueError):
        learning_dealroom("opaque-a", **kwargs)
