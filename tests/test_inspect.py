"""Exercise Inspect's actual agent, registered tools, scorer and public log API."""

import json
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import get_model
from inspect_ai.tool import ToolDef
from jsonschema import ValidationError as SchemaValidationError
from jsonschema import validate

from dealroom.demo import export_logs, run_script, tool_output
from dealroom.domain import parse_action, reset
from dealroom.task import dealroom, sample_for, tools_for
from dealroom.witnesses import (
    FLAGSHIP_FLAWED_SUFFIX,
    FLAGSHIP_PREFIX,
    FLAGSHIP_REPAIRED_SUFFIX,
    WITNESSES,
)


@pytest.mark.parametrize("case_id", ["case-01", "case-03", "case-04", "case-05", "case-06"])
def test_actual_inspect_tool_loop_and_export(case_id, tmp_path):
    log = run_script(
        case_id,
        WITNESSES[case_id],
        {
            "run_kind": "fixture_witness",
            "run_id": case_id,
        },
        tmp_path / "logs",
    )
    sample = log.samples[0]
    assert sample.scores["transaction_success"].value == 1
    actions = sample.metadata["dealroom"]["actions"]
    assert [parse_action(a) for a in actions] == [parse_action(a) for a in WITNESSES[case_id]]
    bundle = export_logs([Path(log.location)], tmp_path / "runs.json")
    assert bundle["summary"]["successes"] == 1
    assert bundle["runs"][0]["score"]["success"] is True
    assert bundle["runs"][0]["steps"][-1]["state"]["disposition"] in {"proceed", "cancel"}
    # No second copy of the model transcript is shipped to the frontend.
    assert "messages" not in bundle["runs"][0]
    assert read_eval_log(log.location).samples[0].metadata["dealroom"]["events"]
    details = bundle["runs"][0]["task"]
    assert details["initial_observation"] == json.loads(sample.input)
    assert details["system_prompt"] == sample.messages[0].text
    assert len(details["tools"]) == 7
    assert "counterpart" not in details["initial_observation"]
    assert "acceptable_dispositions" not in details["initial_observation"]


def test_failed_repaired_prefix_and_truthful_labels(tmp_path):
    paths = []
    for suffix, kind, success in [
        (FLAGSHIP_FLAWED_SUFFIX, "scripted_failure", False),
        (FLAGSHIP_REPAIRED_SUFFIX, "repaired_script", True),
    ]:
        log = run_script(
            "case-01",
            FLAGSHIP_PREFIX + suffix,
            {"run_id": kind, "run_kind": kind, "branch_step": len(FLAGSHIP_PREFIX)},
            tmp_path / "logs",
        )
        assert bool(log.samples[0].scores["transaction_success"].value) is success
        paths.append(Path(log.location))
    bundle = export_logs(paths, tmp_path / "runs.json")
    failure, repaired = bundle["runs"]
    branch = len(FLAGSHIP_PREFIX)
    assert failure["steps"][: branch + 1] == repaired["steps"][: branch + 1]
    assert failure["steps"][-1]["state"]["effective_credit_cents"] == 0
    assert repaired["steps"][-1]["state"]["effective_credit_cents"] == 800_000
    assert repaired["steps"][-1]["state"]["residual_cents"] == 400_000
    assert failure["steps"][0]["state"]["residual_cents"] is None
    assert bundle["summary"]["successes"] == bundle["summary"]["failures"] == 1


def test_concurrent_samples_are_isolated_and_requests_public_only(tmp_path):
    task = dealroom()
    task.dataset = MemoryDataset([sample_for("case-02"), sample_for("case-06")])
    captured = []

    def output(messages, tools, tool_choice, config):
        initial = next(m.text for m in messages if m.role == "user")
        case_id = json.loads(initial)["case_id"]
        index = sum(m.role == "assistant" for m in messages)
        captured.append(
            {
                "case_id": case_id,
                "messages": [m.model_dump(mode="json") for m in messages],
                "tools": [t.model_dump(mode="json") for t in tools],
            }
        )
        assert config.parallel_tool_calls is False
        assert {t.name for t in tools} == {
            "read",
            "request",
            "draft_amendment",
            "send_for_signature",
            "wait_until",
            "set_disposition",
            "finish",
        }
        return tool_output(WITNESSES[case_id][index], index)

    model = get_model("mockllm/model", custom_outputs=output, memoize=False)
    log = inspect_eval(task, model=model, max_samples=2, log_dir=str(tmp_path), display="none")[0]
    assert len(log.samples) == 2
    assert all(s.error is None and s.scores["transaction_success"].value == 1 for s in log.samples)
    histories = {s.id: s.metadata["dealroom"]["actions"] for s in log.samples}
    assert histories["case-02"][-1]["claims"]["effective_credit_cents"] == 350_000
    assert histories["case-06"][-1]["claims"]["effective_credit_cents"] == 0
    raw = json.dumps(captured)
    for forbidden in [
        "acceptable_dispositions",
        "counterpart",
        "seller_max_credit_cents",
        "buyer_grants",
        "fixture_witness",
        "WITNESSES",
        "diagnostics",
        "pending_events",
    ]:
        # 'counterpart' is ordinary public prose, but never an internal key.
        if forbidden == "counterpart":
            assert '\\"counterpart\\":' not in raw
        else:
            assert forbidden not in raw


def test_nested_typed_schema_preserves_discriminators():
    tools, _ = tools_for(reset("case-01"), {"actions": [], "events": []})
    request_schema = ToolDef(tools[1]).parameters.model_dump(exclude_none=True)
    good = {
        "request": {
            "intent": "seller_negotiation",
            "terms": {"type": "credit", "amount_cents": 800_000},
            "evidence_ids": [],
        }
    }
    validate(good, request_schema)
    assert parse_action({"type": "request", **good})
    with pytest.raises(SchemaValidationError):
        validate(
            {"request": {"intent": "seller_negotiation", "terms": {"amount_cents": 800_000}}},
            request_schema,
        )


@pytest.mark.parametrize("mode", ["error", "limit"])
def test_execution_error_and_budget_counts_preserve_denominator(mode, tmp_path):
    task = dealroom(case_id="case-02")
    if mode == "limit":
        task.token_limit = 1
    model = get_model(
        "mockllm/model", custom_outputs=[tool_output({"type": "read"}, 0)], memoize=False
    )
    log = inspect_eval(task, model=model, log_dir=str(tmp_path / "logs"), display="none")[0]
    sample = log.samples[0]
    assert sample.limit if mode == "limit" else sample.error
    bundle = export_logs([Path(log.location)], tmp_path / "runs.json")
    assert bundle["summary"]["total_samples"] == 1
    assert bundle["summary"]["budget_exhaustions" if mode == "limit" else "execution_errors"] == 1
    assert bundle["summary"]["completed_domain_outcomes"] == 0
    assert bundle["runs"] == [] and bundle["issues"]


def test_job_cancel_before_samples_is_not_silently_dropped(tmp_path):
    log = run_script("case-02", WITNESSES["case-02"], {}, tmp_path / "logs")
    log.status = "cancelled"
    log.samples = []
    log.results = None
    path = tmp_path / "cancelled.json"
    write_eval_log(log, path, format="json")
    bundle = export_logs([path], tmp_path / "runs.json")
    assert bundle["summary"]["cancelled_runs"] == 1
    assert bundle["summary"]["total_samples"] == bundle["summary"]["incomplete"] == 1
    assert bundle["issues"]


def test_invalid_finish_can_be_corrected_in_builtin_loop(tmp_path):
    invalid = {
        "type": "finish",
        "claims": {"disposition": "unresolved", "deadline": "2026-10-06T17:00:00"},
        "report": "Invalid timezone in structured claim.",
    }
    log = run_script("case-02", [invalid] + WITNESSES["case-02"], {}, tmp_path)
    assert log.samples[0].scores["transaction_success"].value == 1
    assert len(log.samples[0].metadata["dealroom"]["actions"]) == len(WITNESSES["case-02"]) + 1


def test_export_rejects_changed_fixture(tmp_path):
    log = run_script("case-02", WITNESSES["case-02"], {}, tmp_path / "logs")
    log.samples[0].metadata["dealroom"]["fixture_sha256"] = "changed-fixture"
    path = tmp_path / "changed.json"
    write_eval_log(log, path, format="json")
    with pytest.raises(ValueError, match="Fixture has changed"):
        export_logs([path], tmp_path / "runs.json")
