"""Protocol, canonical scoring, checkpoint integrity, and resume boundaries."""

import copy
import json
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ModelCall, get_model

from dealroom import learning_experiment as experiment_module
from dealroom.curriculum import load_manifest, training_witness
from dealroom.learning_data import demonstration_output
from dealroom.learning_experiment import (
    BASELINE_ID,
    MODEL_ID,
    STAGES,
    Experiment,
    aggregate_attempts,
    atomic_json,
    baseline_is_complete,
    build_protocol,
    file_digest,
    fingerprint,
    initial_state,
    planned_rl_groups,
    report_from_state,
    stop_owned_process,
    summarize_learning_log,
    validate_server_receipt,
)
from dealroom.learning_task import learning_dealroom


@pytest.fixture
def protocol():
    curriculum = load_manifest()
    return build_protocol(curriculum, {
        "curriculum_fingerprint": curriculum["fingerprint"],
        "splits": {split: {"sha256": f"{split}-source"} for split in ("train", "validation")},
    }, {"repo_id": "fixture/base", "revision": "frozen-base-revision", "path": "/fixture/base"})


@pytest.fixture
def experiment(tmp_path, monkeypatch, protocol):
    directory = tmp_path / "experiment"
    atomic_json(directory / "protocol.json", protocol)
    atomic_json(directory / "state.json", initial_state(protocol))
    monkeypatch.setattr(experiment_module, "publish_report", lambda directory, report: atomic_json(directory / "learning.json", report))
    return Experiment(directory)


def base_checkpoint():
    return {"id": "mlx-base", "digest": "base-fingerprint", "step": 0,
            "policy_fingerprint": "base-fingerprint"}


def server_receipt(checkpoint=None):
    checkpoint = checkpoint or base_checkpoint()
    return {"dealroom_server": {"schema_version": 1, "policy_fingerprint": checkpoint["policy_fingerprint"],
                                 "model_path": "/fixture/base", "adapter_path": checkpoint.get("path"), "draft_model_path": None}}


def test_curriculum_schedule_is_frozen_training_only():
    manifest = load_manifest()
    groups = planned_rl_groups(manifest)
    entries = {entry["id"]: entry for entry in manifest["tasks"]}
    assert len(groups) == len({group["case_id"] for group in groups}) == 16
    assert [group["stage"] for group in groups[:4]] == ["commit_ready", "commit_ready", "route_ready", "route_ready"]
    assert all(entries[group["case_id"]]["split"] == "train" for group in groups)
    assert all(not group["conditional"] for group in groups[:8])
    assert all(group["conditional"] for group in groups[8:])


def test_protocol_and_empty_report_do_not_claim_training_or_scores(protocol):
    state = initial_state(protocol)
    report = report_from_state(protocol, state)
    assert report["status"] == "preparing" and report["dataset"]["test_status"] == "sealed"
    assert all(stage["status"] == "pending" and stage["evaluations"] == [] for stage in report["stages"])
    assert report["attempts"] == report["curves"] == []
    assert protocol["rl"]["shaping"] and protocol["rl"]["max_potential"] == 0.2
    assert protocol["sft"]["selection"] == "fixed final checkpoint"
    assert protocol["preflight"]["steps"] == 2
    assert protocol["chat_template_args"] == {"date_string": "27 Sep 2026"}


def test_new_experiment_ids_cannot_escape_log_directory(protocol):
    manifest = load_manifest()
    data = {"curriculum_fingerprint": manifest["fingerprint"], "splits": {}}
    with pytest.raises(ValueError, match="Experiment ID"):
        build_protocol(manifest, data, protocol["model_source"], "../outside")


def test_implementation_freeze_waits_for_baseline_gpu_handoff(experiment, monkeypatch):
    calls = []
    monkeypatch.setattr(experiment, "verify_inputs", lambda **kw: calls.append(kw))
    monkeypatch.setattr(experiment, "recover_children", lambda: None)

    def wait(_seconds):
        raise RuntimeError("baseline still running")

    monkeypatch.setattr(experiment, "wait_for_baseline", wait)
    with pytest.raises(RuntimeError, match="baseline still running"):
        experiment.run()
    assert calls == [{"freeze_implementation": False}]


def test_changed_loaded_code_blocks_before_gpu_handoff(experiment):
    experiment.protocol["data_files"] = {}
    experiment.startup_implementation["dealroom/learning_experiment.py"] = "previously-loaded-code"
    with pytest.raises(RuntimeError, match="changed while this process was queued"):
        experiment.verify_inputs()
    assert not (experiment.directory / "implementation.json").exists()


def test_coverage_does_not_treat_compute_cutoffs_as_failure():
    attempts = [
        {"id": "a", "status": "success", "reward": 1, "total_tokens": 21, "duration_seconds": 2.5},
        {"id": "b", "status": "failure", "reward": 0},
        {"id": "c", "status": "limit", "reward": None},
        {"id": "d", "status": "incomplete", "reward": None},
        {"id": "e", "status": "running"},
    ]
    result = aggregate_attempts(attempts, "validation", 32, "v1")
    assert result["success"] == result["failure"] == result["limit"] == result["incomplete"] == 1
    assert result["error"] == 0 and result["planned"] == 32
    assert result["tokens"] == 21 and result["seconds"] == 2.5
    bad = copy.deepcopy(attempts)
    bad[2]["reward"] = 0
    with pytest.raises(ValueError, match="Unscored"):
        aggregate_attempts(bad, "validation", 32, "v1")
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_attempts(attempts + [attempts[0]], "validation", 32, "v1")


def test_baseline_completion_requires_all_96_unique_resolved_attempts():
    report = {"id": BASELINE_ID, "status": "complete", "attempts": [{"id": str(i), "status": "limit"} for i in range(96)]}
    assert baseline_is_complete(report)
    report["attempts"][-1]["status"] = "running"
    assert not baseline_is_complete(report)
    report["attempts"][-1] = report["attempts"][0]
    assert not baseline_is_complete(report)
    with pytest.raises(ValueError, match="different experiment"):
        baseline_is_complete({**report, "id": "other"})


def test_no_signal_for_reused_or_changed_pid(monkeypatch):
    monkeypatch.setattr(experiment_module, "process_identity", lambda pid: {"pid": pid, "description": "a different command and start time"})
    monkeypatch.setattr(experiment_module.os, "kill", lambda *args: pytest.fail("unowned process was signalled"))
    stop_owned_process({"identity": {"pid": 123, "description": "original owned process"}})


def test_server_provenance_detects_dropped_adapter_and_wrong_policy():
    adapted = {**base_checkpoint(), "path": "/fixture/adapter", "policy_fingerprint": "adapter-hash"}
    validate_server_receipt(server_receipt(adapted), adapted, Path("/fixture/base"))
    response = server_receipt(adapted)
    response["dealroom_server"]["adapter_path"] = None
    with pytest.raises(ValueError, match="adapter"):
        validate_server_receipt(response, adapted, Path("/fixture/base"))
    with pytest.raises(ValueError, match="fingerprint"):
        validate_server_receipt(server_receipt(), adapted, Path("/fixture/base"))


def test_test_release_requires_full_validation_and_blocks_further_training(experiment):
    with pytest.raises(PermissionError):
        experiment.freeze_test_release()
    for stage in STAGES:
        experiment.state["stage_status"][stage] = "complete"
        experiment.state["checkpoints"][stage] = base_checkpoint()
        for case_id in experiment.protocol["cases"]["validation"]:
            key = f"{stage}-validation-{case_id}"
            experiment.state["attempts"][key] = {"id": key, "stage": stage, "split": "validation",
                                                    "case_id": case_id, "status": "failure", "reward": 0}
    experiment.freeze_test_release()
    assert experiment.state["test_status"] == "opened"
    release = json.loads((experiment.directory / "test_release.json").read_text())
    assert release["protocol_sha256"] == fingerprint(experiment.protocol)
    with pytest.raises(PermissionError, match="Training is forbidden"):
        experiment.train("late-training", mode="sft", data_path=Path("never-read"))
    experiment.state["checkpoints"]["rl"] = {**base_checkpoint(), "digest": "changed"}
    with pytest.raises(ValueError, match="different frozen checkpoints"):
        experiment.freeze_test_release()


def test_unopened_test_results_cannot_be_published(protocol):
    state = initial_state(protocol)
    state["attempts"]["forbidden"] = {"id": "forbidden", "stage": "original", "split": "sealed_test",
                                       "case_id": protocol["cases"]["sealed_test"][0], "status": "success", "reward": 1}
    with pytest.raises(ValueError, match="before release"):
        report_from_state(protocol, state)


def completed_training_fixture(experiment, tmp_path, *, rate=2e-5):
    output = tmp_path / "checkpoint"
    output.mkdir()
    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"split":"train"}\n')
    (output / "adapters.safetensors").write_bytes(b"unit-test-only adapter bytes")
    (output / "optimizer.safetensors").write_bytes(b"unit-test-only optimizer bytes")
    atomic_json(output / "adapter_config.json", {"num_layers": 8, "lora_parameters": {"rank": 8, "keys": experiment.protocol["sft"]["lora_keys"]}})
    receipt = {"mode": "sft", "data_digest": file_digest(data_path), "parent_digest": None,
               "digest": file_digest(output / "adapters.safetensors"), "optimizer_updates": 128,
               "learning_rate": rate, "seed": 20260928, "max_length": 24576, "history": []}
    atomic_json(output / "training.json", receipt)
    experiment.state["checkpoints"]["original"] = base_checkpoint()
    return output, data_path


def test_checkpoint_receipt_validates_training_config_and_all_behavior_files(experiment, tmp_path):
    output, data = completed_training_fixture(experiment, tmp_path)
    checkpoint = experiment.commit_training("sft-final", output, "sft", data, None)
    assert checkpoint["policy_fingerprint"] == checkpoint["digest"]
    experiment.verify_checkpoint(checkpoint)
    (output / "adapter_config.json").write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="adapter configuration"):
        experiment.verify_checkpoint(checkpoint)


def test_wrong_training_rate_cannot_be_committed(experiment, tmp_path):
    output, data = completed_training_fixture(experiment, tmp_path, rate=0.1)
    with pytest.raises(ValueError, match="learning rate"):
        experiment.commit_training("sft-final", output, "sft", data, None)


@pytest.mark.parametrize("kind", ["success", "failure", "limit"])
def test_native_inspect_canonical_replay_and_unscored_limits(protocol, tmp_path, kind):
    case_id = protocol["cases"]["train"][0]
    actions = training_witness(case_id) if kind == "success" else ([{
        "type": "finish", "claims": {"disposition": "unresolved"}, "report": "No disposition committed.",
    }] if kind == "failure" else [{"type": "read"}])
    task = learning_dealroom(case_id, workflow="clarified", action_limit=256, compaction_threshold=20000)
    model = get_model("mockllm/model", custom_outputs=[demonstration_output(action, i) for i, action in enumerate(actions)], memoize=False)
    planned = {"id": f"native-{kind}", "stage": "workflow", "split": "train", "case_id": case_id, "workflow": "clarified"}
    log = inspect_eval(task, model=model, log_dir=str(tmp_path / "source"), display="none", time_limit=1800,
                       token_limit=1 if kind == "limit" else 2_000_000,
                       metadata={"protocol_sha256": fingerprint(protocol), "attempt_id": planned["id"],
                                 "policy_fingerprint": "base-fingerprint"})[0]
    assert log.eval.config.time_limit == 1800
    # Only provenance is mocked here. Actual native tools, transitions, scorer,
    # limitations, and canonical score serialization run unchanged.
    log.eval.model = MODEL_ID
    for event in log.samples[0].events:
        if isinstance(event, ModelEvent):
            event.call = ModelCall(request={}, response=server_receipt())
    location = tmp_path / "canonical.eval"
    write_eval_log(log, location)
    restored = read_eval_log(location, resolve_attachments="full")
    result = summarize_learning_log(restored, planned, base_checkpoint(), protocol)
    assert result["status"] == kind
    assert result["reward"] == (None if kind == "limit" else int(kind == "success"))
    assert result["log_sha256"] == file_digest(location)
    if kind == "limit":
        assert restored.results.scores[0].scored_samples == 0
        assert restored.samples[0].scores["learning_success"].reason == "sample_limit"
    # Native cancellations/cutoffs can retain a pending generation without a
    # server response. They must stay unscored and must not force a retry.
    pending = next(event for event in restored.samples[0].events if isinstance(event, ModelEvent)).model_copy(deep=True)
    pending.pending = True
    pending.call = None
    restored.samples[0].events.append(pending)
    if kind != "limit":
        restored.status = "cancelled"
    pending_path = tmp_path / "interrupted.eval"
    write_eval_log(restored, pending_path)
    interrupted = read_eval_log(pending_path, resolve_attachments="full")
    result = summarize_learning_log(interrupted, planned, base_checkpoint(), protocol)
    assert result["status"] == ("limit" if kind == "limit" else "incomplete")
    assert result["reward"] is None
