"""Exercise reporting against actual Inspect mock logs; never call a model runtime."""

import json
import sys
from argparse import Namespace
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ModelUsage, get_model

from dealroom import benchmark
from dealroom.benchmark import DEFAULT_CONFIG, FINAL_STATUSES, create_report, summarize_log
from dealroom.demo import run_script, tool_output
from dealroom.synthesis import DEFAULT_CASE_DIR, validation_witness
from dealroom.task import dealroom
from dealroom.witnesses import FLAGSHIP_FLAWED_SUFFIX, FLAGSHIP_PREFIX, WITNESSES


@pytest.fixture(scope="module")
def canonical_logs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("benchmark-logs")
    success = run_script(
        "case-02",
        WITNESSES["case-02"],
        {"run_id": "test-success", "run_kind": "fixture_witness"},
        directory / "success",
    )
    failure = run_script(
        "case-01",
        FLAGSHIP_PREFIX + FLAGSHIP_FLAWED_SUFFIX,
        {"run_id": "test-failure", "run_kind": "scripted_failure"},
        directory / "failure",
    )
    logs = {"success": success, "failure": failure}
    synthetic_task = dealroom(
        suite="synthetic",
        case_id="synth-007",
        action_limit=DEFAULT_CONFIG["domain_action_limit"],
        end_on_expiry=DEFAULT_CONFIG["end_on_expiry"],
        compaction_threshold=DEFAULT_CONFIG["compaction_threshold"],
    )
    synthetic_task.metadata.update(
        run_id="test-synthetic",
        run_kind="fixture_witness",
        attempt_id="mockllm/model::synth-007",
        benchmark_id="resume-test",
    )
    model = get_model(
        "mockllm/model",
        custom_outputs=[tool_output(a, i) for i, a in enumerate(validation_witness("synth-007"))],
        memoize=False,
    )
    logs["synthetic"] = inspect_eval(
        synthetic_task, model=model, log_dir=str(directory / "synthetic"), display="none"
    )[0]
    assert logs["synthetic"].samples[0].scores["transaction_success"].value == 1
    for mode in ("error", "limit"):
        task = dealroom(case_id="case-02")
        if mode == "limit":
            task.token_limit = 1
        model = get_model(
            "mockllm/model", custom_outputs=[tool_output({"type": "read"}, 0)], memoize=False
        )
        logs[mode] = inspect_eval(task, model=model, log_dir=str(directory / mode), display="none")[
            0
        ]
    return logs


def planned(log):
    return {
        "id": f"{log.eval.model}::{log.samples[0].id}",
        "task_id": str(log.samples[0].id),
        "model_id": log.eval.model,
        "status": "running",
    }


@pytest.mark.parametrize("status", ["success", "failure", "error", "limit"])
def test_canonical_classifications_keep_operational_failures_separate(canonical_logs, status):
    log = canonical_logs[status]
    attempt = summarize_log(log, planned(log))
    assert attempt["status"] == status
    assert attempt["inspect_log"] == Path(log.location).name
    assert attempt["reason"]
    if status in {"success", "failure"}:
        assert attempt["reward"] == int(status == "success")
        assert attempt["run_id"] == f"test-{status}"
        assert attempt["actions"] == len(log.samples[0].metadata["dealroom"]["actions"])
    else:
        assert attempt["reward"] is None
        assert "run_id" not in attempt
    assert attempt["input_tokens"] >= 0
    assert attempt["output_tokens"] >= 0
    assert attempt["total_tokens"] >= 0


def test_recovery_preserves_canonical_completion_time(canonical_logs, monkeypatch):
    log = canonical_logs["success"]
    monkeypatch.setattr(benchmark, "now", lambda: "2099-01-01T00:00:00+00:00")
    recovered = summarize_log(log, planned(log))
    assert recovered["started_at"] == log.samples[0].started_at
    assert recovered["completed_at"] == log.samples[0].completed_at
    no_sample = log.model_copy(deep=True)
    no_sample.samples = []
    assert summarize_log(no_sample, planned(log))["completed_at"] == log.stats.completed_at


@pytest.mark.parametrize(
    "field,value", [("action_limit", 64), ("end_on_expiry", False), ("compaction_threshold", None)]
)
def test_recorded_episode_policy_must_match_declared_configuration(canonical_logs, field, value):
    log = canonical_logs["synthetic"].model_copy(deep=True)
    log.eval.metadata["benchmark_config"] = deepcopy(DEFAULT_CONFIG)
    log.samples[0].metadata["dealroom"][field] = value
    with pytest.raises(ValueError, match="episode policy"):
        summarize_log(log, planned(log))


def test_incomplete_attempts_never_become_reward_zero(canonical_logs):
    original = canonical_logs["success"]
    attempt = planned(original)
    no_sample = original.model_copy(deep=True)
    no_sample.samples = []
    no_sample.results = None
    no_sample.status = "cancelled"
    result = summarize_log(no_sample, attempt)
    assert result["status"] == "incomplete" and result["reward"] is None

    no_record = original.model_copy(deep=True)
    no_record.samples[0].metadata = {}
    result = summarize_log(no_record, attempt)
    assert result["status"] == "incomplete" and result["reward"] is None

    no_finish = original.model_copy(deep=True)
    no_finish.samples[0].metadata["dealroom"]["actions"].pop()
    result = summarize_log(no_finish, attempt)
    assert result["status"] == "incomplete" and result["reward"] is None
    assert "finish" in result["reason"]


def test_empty_foreign_model_log_is_not_assigned_to_a_planned_attempt(canonical_logs):
    log = canonical_logs["success"].model_copy(deep=True)
    attempt = planned(log)
    log.eval.model = "mockllm/foreign"
    log.samples = []
    with pytest.raises(ValueError, match="model"):
        summarize_log(log, attempt)


def test_error_recovery_discards_stale_success_fields(canonical_logs):
    log = canonical_logs["error"]
    attempt = planned(log)
    attempt.update(run_id="stale-success", actions=500, reward=1, arbitrary_derived="stale")
    recovered = summarize_log(log, attempt)
    assert recovered["status"] == "error" and recovered["reward"] is None
    assert {"run_id", "arbitrary_derived"}.isdisjoint(recovered)
    assert recovered["actions"] == len(log.samples[0].metadata["dealroom"]["actions"])


@pytest.mark.parametrize("status", ["success", "failure", "error", "limit"])
def test_canonical_activity_counts_are_reported_for_scored_and_unscored_logs(
    canonical_logs, status
):
    log = canonical_logs[status]
    sample = log.samples[0]
    result = summarize_log(log, planned(log))
    tool_events = [event for event in sample.events if event.event == "tool"]
    assert result["actions"] == len(sample.metadata["dealroom"]["actions"])
    assert result["tool_calls"] == len(tool_events)
    assert result["tool_errors"] == sum(event.error is not None for event in tool_events)
    assert result["generations"] == sum(event.event == "model" for event in sample.events)


def test_activity_counts_include_a_real_inspect_tool_error(tmp_path):
    invalid_finish = {
        "type": "finish",
        "claims": {"disposition": "unresolved", "deadline": "2026-10-06T17:00:00"},
        "report": "Invalid timezone in structured claim.",
    }
    actions = [invalid_finish] + WITNESSES["case-02"]
    log = run_script("case-02", actions, {"run_id": "count-tool-errors"}, tmp_path / "logs")
    result = summarize_log(log, planned(log))
    assert result["status"] == "success"
    assert result["actions"] == result["tool_calls"] == result["generations"] == len(actions)
    assert result["tool_errors"] == 1


def test_missing_activity_records_remain_unknown(canonical_logs):
    log = canonical_logs["success"].model_copy(deep=True)
    log.samples[0].events = None
    log.samples[0].metadata = {}
    result = summarize_log(log, planned(log))
    assert result["status"] == "incomplete"
    assert {"actions", "tool_calls", "tool_errors", "generations"}.isdisjoint(result)


def test_domain_action_exhaustion_is_a_limit_without_a_terminal_reward(canonical_logs):
    log = canonical_logs["success"].model_copy(deep=True)
    # Invalid calls count against the real domain budget even if they use no time.
    log.samples[0].metadata["dealroom"]["actions"] = [
        {"type": "send_for_signature", "revision_id": "not-a-revision"}
    ] * 80
    result = summarize_log(log, planned(log))
    assert result["status"] == "limit" and result["reward"] is None
    assert result["actions"] == 80


def test_usage_is_summed_as_scalars_with_unknown_usage_preserved(canonical_logs):
    log = canonical_logs["success"].model_copy(deep=True)
    log.samples[0].model_usage = {
        "first": ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        "second": ModelUsage(input_tokens=20, output_tokens=8, total_tokens=28),
    }
    result = summarize_log(log, planned(log))
    assert [result[k] for k in ("input_tokens", "output_tokens", "total_tokens")] == [30, 13, 43]
    log.samples[0].model_usage = {}
    result = summarize_log(log, planned(log))
    assert all(result[k] is None for k in ("input_tokens", "output_tokens", "total_tokens"))


def test_planned_matrix_preserves_96_denominator_and_matched_task_order():
    manifest = json.loads((DEFAULT_CASE_DIR / "manifest.json").read_text())
    models = [
        {"id": "ollama/model-one", "label": "One"},
        {"id": "ollama/model-two", "label": "Two"},
    ]
    report = create_report(manifest, models, "test-matched", {"temperature": 0})
    assert len(report["attempts"]) == len({a["id"] for a in report["attempts"]}) == 96
    assert report["task_count"] == 48
    assert Counter(a["model_id"] for a in report["attempts"]) == {
        "ollama/model-one": 48,
        "ollama/model-two": 48,
    }
    orders = [
        [a["task_id"] for a in report["attempts"] if a["model_id"] == model["id"]]
        for model in models
    ]
    assert orders[0] == orders[1]
    assert len({task["family"] for task in report["tasks"][:8]}) == 8
    assert FINAL_STATUSES == {"success", "failure", "error", "limit", "incomplete"}
    assert Counter(a["status"] for a in report["attempts"]) == {"pending": 96}


@pytest.mark.parametrize(
    "mismatch", ["sample", "model", "fixture", "score", "fractional_score", "record_case"]
)
def test_canonical_integrity_rejects_swapped_or_changed_records(canonical_logs, mismatch):
    log = canonical_logs["success"].model_copy(deep=True)
    attempt = planned(log)
    if mismatch == "sample":
        log.samples[0].id = "case-01"
    elif mismatch == "model":
        log.eval.model = "mockllm/different"
    elif mismatch == "fixture":
        log.samples[0].metadata["dealroom"]["fixture_sha256"] = "altered"
    elif mismatch == "score":
        log.samples[0].scores["transaction_success"].value = 0
    elif mismatch == "fractional_score":
        log.samples[0].scores["transaction_success"].value = 1.5
    else:
        # A different task's internally consistent record cannot stand in for this task.
        log.samples[0].metadata["dealroom"] = deepcopy(
            canonical_logs["failure"].samples[0].metadata["dealroom"]
        )
        log.samples[0].scores["transaction_success"].value = 0
    with pytest.raises(ValueError):
        summarize_log(log, attempt)


def test_resume_rejects_duplicate_model_matrix():
    manifest = json.loads((DEFAULT_CASE_DIR / "manifest.json").read_text())
    model = {"id": "ollama/same-model", "label": "Same"}
    with pytest.raises(ValueError):
        create_report(manifest, [model, model], "test-duplicate", {})


@pytest.fixture
def resume_environment(monkeypatch, tmp_path, canonical_logs):
    models = [
        {"id": "mockllm/model", "label": "Mock fixture validator", "digest": "test-digest-one"},
        {"id": "mockllm/second", "label": "Other mock validator", "digest": "test-digest-two"},
    ]
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "model_metadata", lambda *_: deepcopy(models))
    monkeypatch.setattr(benchmark, "local_api", lambda *_: {"version": "test-only"})
    monkeypatch.setattr(
        benchmark,
        "inspect_log_path",
        lambda path: path.resolve().relative_to((tmp_path / "logs").resolve()).as_posix(),
    )

    def no_inference(*args, **kwargs):
        raise AssertionError("Initialization/resume review must not run inference")

    monkeypatch.setattr(benchmark, "inspect_eval", no_inference)
    args = Namespace(
        id="resume-test",
        models=[m["id"] for m in models],
        base_url="http://127.0.0.1:11434/v1",
        manifest=DEFAULT_CASE_DIR / "manifest.json",
        output=tmp_path / "new-output-directory" / "benchmark.json",
        runs_output=tmp_path / "new-replay-directory" / "runs.json",
        time_limit=DEFAULT_CONFIG["time_limit_seconds"],
        initialize_only=True,
        max_attempts=0,
    )
    directory = tmp_path / "logs" / "benchmark" / args.id
    directory.mkdir(parents=True)
    log = canonical_logs["synthetic"].model_copy(deep=True)
    log.eval.metadata.update(
        environment=benchmark.environment_metadata(),
        suite_fingerprint=json.loads(args.manifest.read_text())["fingerprint"],
        benchmark_config={**DEFAULT_CONFIG, "time_limit_seconds": args.time_limit},
        model_digest=models[0]["digest"],
    )
    write_eval_log(log, directory / "canonical.eval")
    return args, directory


def test_resume_recovers_canonical_log_and_corrects_report_tampering(resume_environment):
    args, _ = resume_environment
    report = benchmark.run_benchmark(args)
    assert len(report["attempts"]) == 96
    assert Counter(a["status"] for a in report["attempts"]) == {"pending": 95, "success": 1}
    assert args.output.is_file() and args.runs_output.is_file()
    assert len(json.loads(args.runs_output.read_text())["runs"]) == 1
    attempt = next(a for a in report["attempts"] if a["task_id"] == "synth-007")
    attempt.update(status="failure", reward=0, reason="Tampered summary", run_id="tampered")
    args.output.write_text(json.dumps(report))
    recovered = benchmark.run_benchmark(args)
    attempt = next(a for a in recovered["attempts"] if a["task_id"] == "synth-007")
    assert attempt["status"] == "success" and attempt["reward"] == 1
    assert attempt["run_id"] == "test-synthetic"
    assert attempt["reason"] != "Tampered summary"


@pytest.mark.parametrize("problem", ["missing_log", "duplicate_log", "matrix"])
def test_resume_rejects_missing_duplicate_or_mixed_attempt_records(resume_environment, problem):
    args, directory = resume_environment
    report = benchmark.run_benchmark(args)
    if problem == "missing_log":
        (directory / "canonical.eval").unlink()
    elif problem == "duplicate_log":
        (directory / "duplicate.eval").write_bytes((directory / "canonical.eval").read_bytes())
    else:
        report["attempts"][0]["task_id"] = "case-01"
        args.output.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        benchmark.run_benchmark(args)


@pytest.mark.parametrize(
    "field",
    [
        "benchmark_id",
        "attempt_id",
        "environment",
        "suite_fingerprint",
        "benchmark_config",
        "model_digest",
    ],
)
def test_resume_rejects_foreign_log_metadata(resume_environment, field):
    args, directory = resume_environment
    path = directory / "canonical.eval"
    log = read_eval_log(path)
    log.eval.metadata[field] = "foreign-run"
    write_eval_log(log, path)
    with pytest.raises(ValueError):
        benchmark.run_benchmark(args)


def test_resume_rejects_changed_task_adapter_or_verifier(resume_environment, monkeypatch):
    args, _ = resume_environment
    benchmark.run_benchmark(args)
    changed = {**benchmark.environment_metadata(), "test_changed_source": True}
    monkeypatch.setattr(benchmark, "environment_metadata", lambda: changed)
    with pytest.raises(ValueError):
        benchmark.run_benchmark(args)


@pytest.mark.parametrize("emulate_tools", [False, True])
def test_runner_executes_one_planned_attempt_and_preserves_unattempted_denominator(
    resume_environment, monkeypatch, emulate_tools
):
    args, directory = resume_environment
    args.initialize_only = False
    args.max_attempts = 1
    args.emulate_tools = emulate_tools
    if emulate_tools:
        # The native fixture log belongs to a different tool-calling experiment.
        (directory / "canonical.eval").unlink()
    submitted = []

    def run_mock_inspect(task, **kwargs):
        submitted.append(kwargs)
        case_id = task.dataset[0].id
        model = get_model(
            "mockllm/model",
            custom_outputs=[
                tool_output(action, index)
                for index, action in enumerate(validation_witness(case_id))
            ],
            memoize=False,
            **kwargs["model_args"],
        )
        metadata = {
            **kwargs["metadata"],
            "run_kind": "fixture_witness",
            "run_description": "Test-only mock fixture validation, not model performance.",
        }
        return inspect_eval(
            task, model=model, log_dir=kwargs["log_dir"], metadata=metadata, display="none"
        )

    monkeypatch.setattr(benchmark, "inspect_eval", run_mock_inspect)
    report = benchmark.run_benchmark(args)
    assert len(submitted) == 1
    requested = submitted[0]
    assert requested["metadata"]["attempt_id"] == "mockllm/model::synth-001"
    assert requested["cache"] is False and requested["max_retries"] == 0
    assert requested["parallel_tool_calls"] is False
    assert requested["temperature"] == 0
    assert requested["time_limit"] == args.time_limit
    assert requested["max_tokens"] == DEFAULT_CONFIG["output_tokens_per_turn"]
    assert requested["model_args"] == ({"emulate_tools": True} if emulate_tools else {})
    assert requested["metadata"]["benchmark_config"] == report["config"]
    if emulate_tools:
        assert report["config"]["tool_calling"] == "inspect_text_emulation"
        assert "built-in text-tool" in report["title"]
        assert "built-in text-tool" in requested["metadata"]["run_description"]
    else:
        assert report["config"] == DEFAULT_CONFIG
        assert "tool_calling" not in DEFAULT_CONFIG
        assert report["title"] == "DealRoom · expanded-budget model evaluation"
    assert len(report["attempts"]) == 96
    complete_count = 1 if emulate_tools else 2
    assert Counter(a["status"] for a in report["attempts"]) == {
        "pending": 96 - complete_count,
        "success": complete_count,
    }
    assert report["status"] == "interrupted"
    bundle = json.loads(args.runs_output.read_text())
    assert len(bundle["runs"]) == complete_count
    assert all(run["kind"] == "fixture_witness" for run in bundle["runs"])
    logged_attempt = next(
        item for item in report["attempts"] if item["id"] == "mockllm/model::synth-001"
    )
    log = read_eval_log(benchmark.ROOT / "logs" / logged_attempt["inspect_log"])
    assert log.eval.metadata["benchmark_config"] == report["config"]
    # MockLLM accepts these model args for provenance only. This test verifies
    # orchestration and logging; it does not claim to exercise text parsing.
    assert log.eval.model_args.get("emulate_tools", False) is emulate_tools
    args.initialize_only = True
    resumed = benchmark.run_benchmark(args)
    assert resumed["attempts"] == report["attempts"]
    assert len(submitted) == 1


@pytest.mark.parametrize("initial_emulation", [False, True])
def test_resume_rejects_changing_tool_calling_mode(resume_environment, initial_emulation):
    args, directory = resume_environment
    args.emulate_tools = initial_emulation
    if initial_emulation:
        (directory / "canonical.eval").unlink()
    benchmark.run_benchmark(args)
    before = args.output.read_bytes()
    args.emulate_tools = not initial_emulation
    with pytest.raises(ValueError, match="evaluation settings changed"):
        benchmark.run_benchmark(args)
    assert args.output.read_bytes() == before


def test_resume_rejects_canonical_tool_mode_mislabeled_as_native(resume_environment):
    args, directory = resume_environment
    path = directory / "canonical.eval"
    log = read_eval_log(path)
    log.eval.model_args["emulate_tools"] = True
    write_eval_log(log, path)
    with pytest.raises(ValueError, match="different tool-calling mode"):
        benchmark.run_benchmark(args)
    assert not args.output.exists()


@pytest.mark.parametrize("emulate_tools", [False, True])
def test_cli_propagates_opt_in_tool_emulation(monkeypatch, emulate_tools):
    arguments = ["dealroom-benchmark", "--id", "test-tool-format", "--initialize-only"]
    if emulate_tools:
        arguments.append("--emulate-tools")
    monkeypatch.setattr(sys, "argv", arguments)
    captured = []

    def capture(args):
        captured.append(args)
        return {"attempts": []}

    monkeypatch.setattr(benchmark, "run_benchmark", capture)
    benchmark.main()
    assert len(captured) == 1
    assert captured[0].emulate_tools is emulate_tools


def test_cli_requires_distinct_id_for_emulated_experiment(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dealroom-benchmark", "--emulate-tools"])

    def no_run(args):
        raise AssertionError("A default-ID emulated experiment must not be initialized")

    monkeypatch.setattr(benchmark, "run_benchmark", no_run)
    with pytest.raises(SystemExit) as caught:
        benchmark.main()
    assert caught.value.code == 2


@pytest.fixture
def ollama_report_environment(resume_environment, monkeypatch):
    """Only model identifiers mimic Ollama; all logs and runtime calls are test doubles."""
    args, directory = resume_environment
    models = [
        {"id": "ollama/test-one", "label": "Test one", "digest": "test-digest-one"},
        {"id": "ollama/test-two", "label": "Test two", "digest": "test-digest-two"},
    ]
    args.models = [model["id"] for model in models]
    monkeypatch.setattr(benchmark, "model_metadata", lambda *_: deepcopy(models))
    log = read_eval_log(directory / "canonical.eval")
    log.eval.model = models[0]["id"]
    log.eval.metadata["attempt_id"] = "ollama/test-one::synth-007"
    write_eval_log(log, directory / "canonical.eval")
    manifest = json.loads(args.manifest.read_text())
    config = {**DEFAULT_CONFIG, "time_limit_seconds": args.time_limit}
    report = create_report(manifest, models, args.id, config)
    attempt = next(a for a in report["attempts"] if a["id"] == "ollama/test-one::synth-007")
    attempt.update(status="success", reward=1, served_context_window=config["context_window"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report))
    return args, directory


def test_resume_preserves_measured_context_alongside_canonical_outcome(ollama_report_environment):
    args, _ = ollama_report_environment
    report = benchmark.run_benchmark(args)
    attempt = next(a for a in report["attempts"] if a["id"] == "ollama/test-one::synth-007")
    assert attempt["status"] == "success" and attempt["reward"] == 1
    assert attempt["served_context_window"] == DEFAULT_CONFIG["context_window"]
    persisted = json.loads(args.output.read_text())
    assert persisted["attempts"] == report["attempts"]


@pytest.mark.parametrize("measurement", [None, 8192, "32768", True])
def test_resume_refuses_unknown_or_changed_serving_observation(
    ollama_report_environment, measurement
):
    args, _ = ollama_report_environment
    report = json.loads(args.output.read_text())
    attempt = next(a for a in report["attempts"] if a["id"] == "ollama/test-one::synth-007")
    if measurement is None:
        del attempt["served_context_window"]
    else:
        attempt["served_context_window"] = measurement
    args.output.write_text(json.dumps(report))
    before = args.output.read_bytes()
    with pytest.raises(ValueError, match="(served-context observation|Persisted served context)"):
        benchmark.run_benchmark(args)
    assert args.output.read_bytes() == before


@pytest.mark.parametrize("measurement", [None, 8192, DEFAULT_CONFIG["context_window"]])
def test_runtime_context_is_validated_before_terminal_result_is_published(
    ollama_report_environment, monkeypatch, measurement
):
    args, directory = ollama_report_environment
    args.initialize_only = False
    args.max_attempts = 1
    calls = []

    def mocked_api(base_url, route, payload=None):
        if route == "/api/version":
            return {"version": "test-only"}
        assert route == "/api/ps"
        return {
            "models": []
            if measurement is None
            else [{"name": "test-one", "context_length": measurement}]
        }

    def mocked_inference(task, **kwargs):
        # Drive only mock outputs through Inspect; never connect to Ollama.
        calls.append(kwargs["metadata"]["attempt_id"])
        model = get_model(
            "mockllm/model",
            custom_outputs=[
                tool_output(action, index)
                for index, action in enumerate(validation_witness(task.dataset[0].id))
            ],
            memoize=False,
        )
        metadata = {**kwargs["metadata"], "run_kind": "fixture_witness"}
        logs = inspect_eval(
            task, model=model, log_dir=kwargs["log_dir"], metadata=metadata, display="none"
        )
        # This isolated test double activates the serving-observation branch.
        logs[0].eval.model = kwargs["model"]
        write_eval_log(logs[0], logs[0].location)
        return logs

    monkeypatch.setattr(benchmark, "local_api", mocked_api)
    monkeypatch.setattr(benchmark, "inspect_eval", mocked_inference)
    if measurement == DEFAULT_CONFIG["context_window"]:
        benchmark.run_benchmark(args)
    else:
        with pytest.raises(ValueError, match="attempt remains unvalidated"):
            benchmark.run_benchmark(args)
    assert calls == ["ollama/test-one::synth-001"]
    assert len(list(directory.glob("*.eval"))) == 2
    report = json.loads(args.output.read_text())
    attempt = next(a for a in report["attempts"] if a["id"] == "ollama/test-one::synth-001")
    if measurement == DEFAULT_CONFIG["context_window"]:
        assert attempt["status"] == "success" and attempt["reward"] == 1
        assert attempt["served_context_window"] == measurement
        args.initialize_only = True
        resumed = benchmark.run_benchmark(args)
        assert (
            next(a for a in resumed["attempts"] if a["id"] == attempt["id"])[
                "served_context_window"
            ]
            == measurement
        )
    else:
        assert report["status"] == "interrupted" and attempt["status"] == "running"
        assert {"reward", "run_id", "served_context_window"}.isdisjoint(attempt)
        # A canonical log without its historical serving observation must not
        # be retried or inferred from whichever runtime is currently available.
        args.initialize_only = True
        with pytest.raises(ValueError, match="No persisted served-context observation"):
            benchmark.run_benchmark(args)
        assert calls == ["ollama/test-one::synth-001"]
