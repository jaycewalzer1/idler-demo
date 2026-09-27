"""Opt-in episode policy tests; mock tool outputs never measure model capability."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalError, EvalSampleLimit, write_eval_log
from inspect_ai.model import get_model

from dealroom.benchmark import summarize_log
from dealroom.demo import export_logs, tool_output
from dealroom.domain import load_case, reset, step
from dealroom.score import evaluate
from dealroom.task import case_for_record, dealroom, fixture_fingerprint, replay
from dealroom.witnesses import WITNESSES, draft, finish, proceed, route, wait


def run_episode(actions, directory, **options):
    task = dealroom(case_id="case-02", **options)
    task.metadata.update(run_id="test-episode", run_kind="fixture_witness")
    model = get_model(
        "mockllm/model",
        custom_outputs=[tool_output(action, index) for index, action in enumerate(actions)],
        memoize=False,
    )
    log = inspect_eval(task, model=model, log_dir=str(directory), display="none")[0]
    assert log.samples[0].error is None and log.samples[0].limit is None
    return log


@pytest.fixture
def expired_episode(tmp_path):
    deadline = load_case("case-02").deadline.isoformat()
    return run_episode(
        [wait(deadline)],
        tmp_path / "logs",
        action_limit=256,
        end_on_expiry=True,
        compaction_threshold=28672,
    )


def test_expiry_is_an_objective_episode_failure_without_fabricated_finish(expired_episode):
    sample = expired_episode.samples[0]
    record = sample.metadata["dealroom"]
    assert record["end_on_expiry"] is True
    assert record["termination_reason"] == "contract_deadline_expired"
    assert record["action_limit"] == 256 and record["compaction_threshold"] == 28672
    assert len(record["actions"]) == 1 and record["actions"][0]["type"] == "wait_until"
    assert sample.turn_count == 1
    score = sample.scores["transaction_success"]
    assert score.value == 0 and score.metadata["outcome"] == "failure"
    assert score.metadata["finished"] is False
    check = next(d for d in score.metadata["diagnostics"] if d["name"] == "completed_episode")
    assert check["passed"] and "no final report or claims were fabricated" in check["reason"]
    state = replay("case-02", record["actions"], action_limit=record["action_limit"])
    assert not state.finished and state.final_claims is None and state.final_report == ""
    assert state.business_status == "expired" and state.disposition is None
    assert len(next(e for e in sample.events if e.event == "model").tools) == 7


def test_default_policy_still_allows_read_and_finish_after_expiry(tmp_path):
    log = run_episode(
        [
            wait(load_case("case-02").deadline.isoformat()),
            {"type": "read"},
            finish(0, [], "unresolved"),
        ],
        tmp_path,
    )
    sample = log.samples[0]
    record = sample.metadata["dealroom"]
    assert record["end_on_expiry"] is False
    assert "termination_reason" not in record
    assert len(record["actions"]) == 3
    assert sample.scores["transaction_success"].metadata["finished"] is True
    assert any(
        d["name"] == "completed_attempt"
        for d in sample.scores["transaction_success"].metadata["diagnostics"]
    )


def test_action_limit_override_is_public_recorded_and_replayable(expired_episode):
    sample = expired_episode.samples[0]
    record = sample.metadata["dealroom"]
    base = load_case("case-02")
    effective = case_for_record(record)
    assert base.action_limit == 80 and effective.action_limit == 256
    assert json.loads(sample.input)["action_limit"] == 256
    assert record["base_fixture_sha256"] == fixture_fingerprint(base)
    assert record["fixture_sha256"] == fixture_fingerprint(effective)
    assert record["fixture_sha256"] != record["base_fixture_sha256"]
    actions = [{"type": "read"}] * 81
    assert replay("case-02", actions).budget_exhausted
    assert not replay("case-02", actions, action_limit=256).budget_exhausted
    assert load_case("case-02").action_limit == 80


def test_expiry_scoring_is_opt_in_and_requires_real_expiration():
    state = reset("case-02")
    assert evaluate(state, end_on_expiry=True).outcome == "incomplete"
    assert step(state, wait(state.deadline.isoformat())).accepted
    assert evaluate(state).outcome == "incomplete"
    opted = evaluate(state, end_on_expiry=True)
    assert opted.outcome == "failure" and opted.reward == 0 and not opted.finished


def test_disposition_at_exact_deadline_can_still_finish_successfully(tmp_path):
    deadline = load_case("case-02").deadline
    log = run_episode(
        [
            draft(350_000),
            route(),
            wait((deadline - timedelta(minutes=5)).isoformat()),
            proceed(),
            finish(350_000, [1], deadline=deadline.isoformat()),
        ],
        tmp_path,
        end_on_expiry=True,
        action_limit=256,
    )
    sample = log.samples[0]
    assert sample.scores["transaction_success"].value == 1
    assert "termination_reason" not in sample.metadata["dealroom"]
    assert sample.metadata["dealroom"]["actions"][-1]["type"] == "finish"


def test_export_includes_genuine_expiry_failure_with_no_finish(expired_episode, tmp_path):
    bundle = export_logs([Path(expired_episode.location)], tmp_path / "runs.json")
    assert bundle["summary"]["failures"] == 1
    assert bundle["summary"]["completed_domain_outcomes"] == 1
    assert bundle["summary"]["incomplete"] == 0
    run = bundle["runs"][0]
    assert run["score"]["success"] is False
    assert run["episode"] == {
        "termination_reason": "contract_deadline_expired",
        "finished": False,
        "action_limit": 256,
    }
    assert run["steps"][-1]["state"]["disposition"] == "expired"
    assert "no finish call or final claims were fabricated" in run["steps"][-1]["reviewer"]["note"]
    assert all(
        step["action"] is None or step["action"]["type"] != "finish" for step in run["steps"]
    )


@pytest.mark.parametrize("missing", ["opt_in", "callback_reason", "real_expiration"])
def test_export_cannot_promote_unfinished_attempts_without_verified_expiry_policy(
    expired_episode, tmp_path, missing
):
    log = expired_episode.model_copy(deep=True)
    record = log.samples[0].metadata["dealroom"]
    if missing == "opt_in":
        record["end_on_expiry"] = False
    elif missing == "callback_reason":
        del record["termination_reason"]
    else:
        record["actions"] = [{"type": "read"}]
    path = tmp_path / "changed.eval"
    write_eval_log(log, path)
    bundle = export_logs([path], tmp_path / "runs.json")
    assert bundle["summary"]["incomplete"] == 1
    assert bundle["summary"]["completed_domain_outcomes"] == 0
    assert bundle["runs"] == []


@pytest.mark.parametrize("kind", ["limit", "error"])
def test_operational_limit_or_error_takes_precedence_over_expiry(expired_episode, tmp_path, kind):
    log = expired_episode.model_copy(deep=True)
    if kind == "limit":
        log.samples[0].limit = EvalSampleLimit(type="turn", limit=1, reason="Test budget")
    else:
        log.samples[0].error = EvalError(message="Test error", traceback="", traceback_ansi="")
    path = tmp_path / "operational.eval"
    write_eval_log(log, path)
    bundle = export_logs([path], tmp_path / "runs.json")
    assert bundle["summary"]["failures"] == 0 and bundle["runs"] == []
    assert bundle["summary"]["budget_exhaustions" if kind == "limit" else "execution_errors"] == 1


def test_old_records_reconstruct_without_new_episode_metadata(tmp_path):
    log = run_episode(WITNESSES["case-02"], tmp_path / "logs")
    record = log.samples[0].metadata["dealroom"]
    for key in ("action_limit", "end_on_expiry", "compaction_threshold", "base_fixture_sha256"):
        del record[key]
    assert case_for_record(record) == load_case("case-02")
    path = tmp_path / "legacy.eval"
    write_eval_log(log, path)
    bundle = export_logs([path], tmp_path / "runs.json")
    assert bundle["summary"]["successes"] == 1


@pytest.mark.parametrize("argument", ["action_limit", "compaction_threshold"])
def test_episode_limits_require_positive_integer_values(argument):
    for value in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            dealroom(case_id="case-02", **{argument: value})


@pytest.mark.parametrize("operational_status", [None, "limit", "error"])
def test_benchmark_summary_distinguishes_expired_episode_from_operational_limits(
    expired_episode, operational_status
):
    log = expired_episode.model_copy(deep=True)
    log.eval.metadata["benchmark_config"] = {
        "domain_action_limit": 256,
        "end_on_expiry": True,
        "compaction_threshold": 28672,
    }
    if operational_status == "limit":
        log.samples[0].limit = EvalSampleLimit(type="turn", limit=1, reason="Test budget")
    elif operational_status == "error":
        log.samples[0].error = EvalError(message="Test error", traceback="", traceback_ansi="")
    attempt = {
        "id": "mockllm/model::case-02",
        "model_id": log.eval.model,
        "task_id": "case-02",
    }
    result = summarize_log(log, attempt)
    if operational_status:
        assert result["status"] == operational_status
        assert result["reward"] is None and "run_id" not in result
    else:
        assert result["status"] == "failure" and result["reward"] == 0
        assert result["termination_reason"] == "contract_deadline_expired"
        assert result["actions"] == 1
        assert "deadline" in result["reason"]
    assert log.samples[0].scores["transaction_success"].metadata["finished"] is False
    assert all(a["type"] != "finish" for a in log.samples[0].metadata["dealroom"]["actions"])
