"""Resumable local learning experiment, using native Inspect for every rollout.

Prepare is CPU-only. Run first waits for the original benchmark to finish, then
owns one MLX server or trainer at a time. Fixed protocol, canonical logs, immutable
checkpoint receipts, and the test-release record determine what can be resumed.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from inspect_ai import eval as inspect_eval
from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log

from dealroom.curriculum import DEFAULT_LEARNING_DIR, load_learning_case, load_manifest
from dealroom.domain import reset
from dealroom.learning_task import (
    LEARNING_WORKFLOW_VERSION,
    learning_case_for_record,
    learning_dealroom,
    learning_source_fingerprints,
    replay_learning_record,
)
from dealroom.learning_train import TEMPLATE_ARGS
from dealroom.score import evaluate
from dealroom.task import fixture_fingerprint

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENT_DIR = ROOT / "training" / "experiment-v1"
TRAINING_PYTHON = ROOT / "training" / ".venv" / "bin" / "python"
BASELINE_REPORT = ROOT / "web" / "public" / "benchmark.json"
BASELINE_ID = "dealroom-local-48-text-tools-v4"
MODEL_ID = "openai-api/mlx/default_model"
BASE_URL = "http://127.0.0.1:8087/v1"
PROTOCOL_VERSION = "dealroom-learning-protocol-v1"
STAGES = ("original", "workflow", "sft", "rl")
LABELS = {"original": "Original model", "workflow": "Clearer workflow",
          "sft": "Supervised fine-tuning", "rl": "Reinforcement learning"}
TERMINAL_STATUSES = frozenset({"success", "failure", "limit", "error", "incomplete"})
EVALUATION_CONFIG = {
    "temperature": 0, "max_turns": 128, "max_messages": 512,
    "max_tokens": 2_000_000, "output_tokens_per_turn": 2048,
    "time_limit_seconds": 1800, "domain_action_limit": 256,
    "compaction_threshold": 20000, "compaction_preserve": 0.5,
    "max_training_sequence_length": 24576, "end_on_expiry": True,
    "seed": 20260928, "tool_calling": "inspect_text_emulation",
    "score_policy": "canonical_unscored_limits",
}
IMPLEMENTATION_FILES = (
    "dealroom/domain.py", "dealroom/score.py", "dealroom/task.py", "dealroom/curriculum.py",
    "dealroom/learning_task.py", "dealroom/learning_data.py", "dealroom/learning_train.py",
    "dealroom/learning_policy.py", "dealroom/learning_serve.py", "dealroom/learning_experiment.py",
    "dealroom/learning_analysis.py",
    "training/pyproject.toml", "training/uv.lock",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def planned_rl_groups(curriculum: dict) -> list[dict]:
    """Choose cases by frozen manifest order, never by model outcomes or test data."""
    train = [item for item in curriculum["tasks"] if item["split"] == "train"]
    primary = []
    for stage in ("commit_ready", "route_ready"):
        primary.extend(item for item in train if item["stage"] == stage)
        # Exactly two of each warm-start stage.
        primary = primary[:2] if stage == "commit_ready" else primary[:4]
    full = [item for item in train if item["stage"] == "coordination"]
    for family in ("credit_execution", "quote_credit", "authority_credit", "extension_credit"):
        primary.append(next(item for item in full if item["family"] == family))
    additional = []
    used = {item["id"] for item in primary}
    for family in curriculum["training_families"]:
        additional.append(next(item for item in full if item["family"] == family and item["id"] not in used))
    if len(primary) != 8 or len(additional) != 8:
        raise ValueError("Expected eight primary and eight conditional training groups.")
    return [{"index": index, "case_id": entry["id"], "stage": entry["stage"],
             "family": entry["family"], "conditional": index >= 8}
            for index, entry in enumerate(primary + additional)]


def build_protocol(curriculum: dict, data_manifest: dict, model_source: dict,
                   experiment_id: str = "dealroom-llama-learning-v1") -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,99}", experiment_id):
        raise ValueError("Experiment ID must use lowercase letters, numbers, dots, underscores, or hyphens.")
    if data_manifest["curriculum_fingerprint"] != curriculum["fingerprint"]:
        raise ValueError("SFT data and curriculum fingerprints differ.")
    return {
        "version": PROTOCOL_VERSION, "experiment_id": experiment_id,
        "chat_template_args": dict(TEMPLATE_ARGS),
        "workflow_version": LEARNING_WORKFLOW_VERSION,
        "curriculum_directory": str(DEFAULT_LEARNING_DIR),
        "curriculum_fingerprint": curriculum["fingerprint"],
        "dataset_version": curriculum["suite_id"], "split_counts": curriculum["split_counts"],
        "cases": {split: [item["id"] for item in curriculum["tasks"] if item["split"] == split]
                  for split in ("train", "validation", "sealed_test")},
        "source_fingerprints": learning_source_fingerprints(),
        "data_files": {split: entry["sha256"] for split, entry in data_manifest["splits"].items()},
        "model_source": model_source, "evaluation": dict(EVALUATION_CONFIG),
        "training_memory": {"attention_forward": "causal query blocks",
                            "attention_query_block_tokens": 256,
                            "feed_forward_block_tokens": 256,
                            "recompute_attention_blocks": True,
                            "serial_block_gradients": True, "compiled_optimizer_step": False,
                            "frozen_prefix_outside_autodiff": True,
                            "layer_gradients": "materialized one layer at a time; no full-model gradient trace",
                            "attention_forward_inputs": "host-backed constants with explicit custom VJP",
                            "context_and_objective": "unchanged; exact causal attention"},
        "sft": {"steps": 128, "learning_rate": 2e-5, "seed": 20260928,
                "selection": "fixed final checkpoint", "lora_rank": 8, "last_layers": 8,
                "lora_keys": ["self_attn.q_proj", "self_attn.v_proj"]},
        "preflight": {"steps": 2, "selection": "Shortest and longest tokenized TRAIN rows; disposable infrastructure checkpoint, never selected for evaluation.",
                      "learning_rate": 2e-5, "validation": False},
        "rl": {
            "algorithm": "population-standardized group-relative REINFORCE",
            "primary_groups": 8, "conditional_additional_groups": 8, "rollouts_per_group": 4,
            "additional_trigger": "After eight primary groups, run all eight additional groups only if fewer than two optimizer updates occurred.",
            "groups": planned_rl_groups(curriculum), "learning_rate": 1e-6,
            "temperature": 1, "top_p": 1, "top_k": 0, "min_p": 0,
            "sampling_penalties": "none", "updates_per_nonconstant_group": 1,
            "reuse_rollouts": False, "selection": "last completed update after predeclared groups",
            "reward": "binary_terminal_plus_semantic_potential_delta_v1",
            "shaping": True, "max_potential": 0.2,
            "shaping_scope": "train only; semantic records, not activity counts; no policy-invariance claim",
            "equal_reward_groups": "skip; no fabricated optimizer step",
            "unscored_outcomes": "excluded from rewards, advantages, and scoring denominators",
        },
        "test_policy": "Open once after all validation evaluations and final checkpoint hashes are frozen; never train or select checkpoints using test results.",
        "baseline_precondition": {"id": BASELINE_ID, "report": str(BASELINE_REPORT), "planned": 96},
    }


def initial_state(protocol: dict) -> dict:
    return {"protocol_sha256": fingerprint(protocol), "status": "preparing", "created_at": now(),
            "attempts": {}, "checkpoints": {}, "training_jobs": {}, "rl_groups": {},
            "curves": [], "events": [], "stage_status": {stage: "pending" for stage in STAGES},
            "test_status": "sealed", "current": None}


def aggregate_attempts(attempts: list[dict], split: str, planned: int, dataset_version: str) -> dict:
    if len({attempt["id"] for attempt in attempts}) != len(attempts):
        raise ValueError("Duplicate evaluation attempt IDs.")
    for attempt in attempts:
        status, reward = attempt["status"], attempt.get("reward")
        if status not in TERMINAL_STATUSES | {"pending", "running"}:
            raise ValueError("Unknown attempt status.")
        if status in {"success", "failure"}:
            if type(reward) is not int or reward != int(status == "success"):
                raise ValueError("Scored status and canonical binary reward disagree.")
        elif reward is not None:
            raise ValueError("Unscored outcomes cannot be assigned a zero reward.")
    counts = Counter(attempt["status"] for attempt in attempts if attempt["status"] in TERMINAL_STATUSES)
    if sum(counts.values()) > planned:
        raise ValueError("Recorded evaluation exceeds its frozen plan.")
    result = {"split": split, "planned": planned, "dataset_version": dataset_version,
              **{status: counts[status] for status in ("success", "failure", "limit", "error", "incomplete")}}
    tokens = [item["total_tokens"] for item in attempts if item.get("total_tokens") is not None]
    seconds = [item["duration_seconds"] for item in attempts if item.get("duration_seconds") is not None]
    if tokens:
        result["tokens"] = sum(tokens)
    if seconds:
        result["seconds"] = sum(seconds)
    return result


def report_from_state(protocol: dict, state: dict) -> dict:
    stages = []
    for stage in STAGES:
        evaluations = []
        for split in ("validation", "sealed_test"):
            selected = [item for item in state["attempts"].values()
                        if item.get("stage") == stage and item.get("split") == split]
            if selected:
                if split == "sealed_test" and state["test_status"] != "opened":
                    raise ValueError("Cannot publish sealed-test outcomes before release.")
                evaluations.append(aggregate_attempts(selected, split, protocol["split_counts"][split], protocol["dataset_version"]))
        item = {"id": stage, "label": LABELS[stage], "status": state["stage_status"][stage],
                "evaluations": evaluations}
        if stage in state["checkpoints"]:
            item["checkpoint"] = state["checkpoints"][stage]
        if stage == "rl":
            groups = list(state["rl_groups"].values())
            updates = sum(group.get("optimizer_updates", 0) for group in groups)
            item["notes"] = [f"Recorded {len(groups)} completed rollout groups and {updates} optimizer updates.",
                             "Training uses bounded semantic progress shaping; reported evaluation scores remain binary."]
        stages.append(item)
    source = protocol["model_source"]
    return {
        "schema_version": 1, "id": protocol["experiment_id"],
        "title": "Llama learning in DealRoom", "status": state["status"], "updated_at": now(),
        "model": {"id": source["repo_id"], "label": "Llama 3.1 8B Instruct",
                  "base_revision": source["revision"], "quantization": "MLX 4-bit LoRA"},
        "dataset": {"version": protocol["dataset_version"], "train_count": protocol["split_counts"]["train"],
                    "validation_count": protocol["split_counts"]["validation"], "test_count": protocol["split_counts"]["sealed_test"],
                    "fingerprint": protocol["curriculum_fingerprint"], "test_status": state["test_status"]},
        "stages": stages, "curves": state["curves"], "events": state["events"],
        "attempts": [{"id": item["id"], "stage": item["stage"], "split": item["split"],
                      "case_id": item["case_id"], "status": item["status"], "reward": item.get("reward"),
                      **({"inspect_log": str(Path(item["native_log"]).relative_to(ROOT / "logs"))} if item.get("native_log") else {}),
                      "tokens": item.get("total_tokens"), "seconds": item.get("duration_seconds"),
                      **({"reason": item["reason"]} if item.get("reason") else {})}
                     for item in state["attempts"].values()],
        "limitations": [
            "This is a small local adapter-training pilot on synthetic transactions; improvement is measured, not guaranteed.",
            "The MLX base uses a different 4-bit representation from the earlier Ollama benchmark; all four learning conditions share the identical MLX base.",
            "Compute limits, errors, and incomplete attempts are unscored. Compare success rates together with scored coverage.",
            "RL training adds a bounded semantic potential difference to binary terminal reward. It does not claim reward invariance; held-out success uses the unchanged binary verifier.",
            "Fixed final checkpoints are selected without sealed-test feedback. Validation loss is diagnostic; equal-reward rollout groups produce no optimizer update.",
        ],
        "config": {"protocol_sha256": state["protocol_sha256"], "evaluation": protocol["evaluation"],
                   "training_memory": protocol.get("training_memory", {"strategy": "original MLX trainer"}),
                   "sft": protocol["sft"], "rl": protocol["rl"], "preflight": protocol["preflight"], "current": state.get("current"),
                   "chat_template_args": protocol["chat_template_args"],
                   "training_runs": [{"job_id": job_id, "status": job["status"],
                                      **{key: job["training"][key] for key in ("seconds", "peak_memory_bytes", "optimizer_updates") if key in job.get("training", {})}}
                                     for job_id, job in state["training_jobs"].items()],
                   "baseline_precondition": protocol["baseline_precondition"],
                   "selection": protocol["test_policy"]},
    }


def prepare_experiment(directory: str | Path = DEFAULT_EXPERIMENT_DIR,
                       experiment_id: str | None = None, *, revise_unstarted: bool = False) -> dict:
    output = Path(directory)
    protocol_path, state_path = output / "protocol.json", output / "state.json"
    old_protocol = read_json(protocol_path) if protocol_path.exists() else None
    experiment_id = experiment_id or (old_protocol["experiment_id"] if old_protocol else "dealroom-llama-learning-v1")
    data = read_json(ROOT / "training" / "data" / "manifest.json")
    protocol = build_protocol(load_manifest(), data, read_json(ROOT / "training" / "model_source.json"), experiment_id)
    tokenized = ROOT / "training" / "tokenized"
    if not (tokenized / "manifest.json").exists():
        raise ValueError("Prepare/tokenize supervised data before freezing the experiment protocol.")
    token_manifest = read_json(tokenized / "manifest.json")
    if token_manifest["template_args"] != protocol["chat_template_args"]:
        raise ValueError("Tokenization template arguments differ from the frozen serving template.")
    if any(token_manifest["splits"][split]["source_sha256"] != expected for split, expected in protocol["data_files"].items()):
        raise ValueError("Tokenized data does not match the corrected supervised export.")
    protocol["tokenized_files"] = {split: file_digest(tokenized / f"{split}.jsonl") for split in ("train", "validation")}
    if protocol_path.exists():
        state = read_json(state_path)
        if old_protocol != protocol:
            if not revise_unstarted:
                raise ValueError("Frozen protocol differs; prepare a new --directory and --id, or explicitly --revise-unstarted before any execution.")
            with (output / "experiment.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError("Stop the active orchestrator before revising an unstarted protocol.") from error
                state = read_json(state_path)
                if state["attempts"] or state["training_jobs"] or (output / "implementation.json").exists() or state["test_status"] != "sealed":
                    raise ValueError("An experiment with execution records cannot revise its protocol.")
                atomic_json(output / "protocol_history" / f"{fingerprint(old_protocol)}.json", old_protocol)
                state["protocol_sha256"] = fingerprint(protocol)
                state["events"].append({"at": now(), "message": "Protocol amended before any evaluation or training; prior protocol hash archived. Serving and training use the same frozen chat template."})
                atomic_json(protocol_path, protocol)
                atomic_json(state_path, state)
        if state["protocol_sha256"] != fingerprint(protocol):
            raise ValueError("State is bound to a different protocol.")
    else:
        output.mkdir(parents=True, exist_ok=True)
        state = initial_state(protocol)
        state["events"].append({"at": now(), "message": "Protocol frozen. Waiting for the original two-model benchmark before using the GPU."})
        atomic_json(protocol_path, protocol)
        atomic_json(state_path, state)
    report = report_from_state(protocol, state)
    publish_report(output, report)
    return report


def publish_report(directory: Path, report: dict) -> None:
    atomic_json(directory / "learning.json", report)
    atomic_json(ROOT / "web" / "public" / "learning.json", report)
    if (ROOT / "web" / "dist").is_dir():
        atomic_json(ROOT / "web" / "dist" / "learning.json", report)


def process_identity(pid: int) -> dict | None:
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
                            capture_output=True, text=True, check=False)
    line = result.stdout.strip()
    return {"pid": pid, "description": line} if result.returncode == 0 and line else None


def stop_owned_process(record: dict) -> None:
    """Do not signal a reused PID or a process whose command/start time changed."""
    identity = record.get("identity")
    if not identity or process_identity(identity["pid"]) != identity:
        return
    os.kill(identity["pid"], signal.SIGTERM)
    for _ in range(50):
        if process_identity(identity["pid"]) != identity:
            return
        time.sleep(0.1)
    if process_identity(identity["pid"]) == identity:
        os.kill(identity["pid"], signal.SIGKILL)


def stop_child(process: subprocess.Popen) -> None:
    """Reap our live child even if macOS rewrites its Python command after exec."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def baseline_is_complete(report: dict) -> bool:
    if report.get("id") != BASELINE_ID:
        raise ValueError("The baseline report is from a different experiment.")
    attempts = report.get("attempts", [])
    return (report.get("status") == "complete" and len(attempts) == 96
            and len({attempt["id"] for attempt in attempts}) == 96
            and all(attempt.get("status") in TERMINAL_STATUSES for attempt in attempts))


def baseline_runner_active() -> bool:
    listing = subprocess.run(["ps", "-Ao", "command="], capture_output=True, text=True, check=True).stdout
    return any("-m dealroom.benchmark" in line and BASELINE_ID in line for line in listing.splitlines())


def api_request(route: str, payload: dict | None = None, *, origin: str = "http://127.0.0.1:8087") -> dict:
    request = Request(origin + route, data=json.dumps(payload).encode() if payload is not None else None,
                      headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def validate_server_receipt(response: dict, checkpoint: dict, model_path: Path) -> None:
    receipt = response.get("dealroom_server")
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise ValueError("MLX serving provenance is missing; do not score an unverified checkpoint.")
    expected_adapter = checkpoint.get("path")
    actual_adapter = receipt.get("adapter_path")
    if receipt.get("policy_fingerprint") != checkpoint["policy_fingerprint"]:
        raise ValueError("Response came from a different policy fingerprint.")
    if Path(receipt["model_path"]).resolve() != model_path.resolve():
        raise ValueError("MLX actually loaded a different base model.")
    if bool(expected_adapter) != bool(actual_adapter) or (expected_adapter and
            Path(expected_adapter).resolve() != Path(actual_adapter).resolve()):
        raise ValueError("MLX actually loaded a different adapter (or dropped the requested adapter).")
    if receipt.get("draft_model_path"):
        raise ValueError("Speculative decoding is outside the frozen sampling protocol.")


def summarize_learning_log(log, planned: dict, checkpoint: dict, protocol: dict) -> dict:
    """Replay canonical scores; interrupted or limited computation has no reward."""
    result = {**planned, "native_log": str(log.location), "log_sha256": file_digest(Path(log.location))}
    metadata = log.eval.metadata or {}
    if (metadata.get("protocol_sha256") != fingerprint(protocol)
            or metadata.get("attempt_id") != planned["id"]
            or metadata.get("policy_fingerprint") != checkpoint["policy_fingerprint"]
            or log.eval.model != MODEL_ID):
        raise ValueError("Canonical attempt does not match the frozen plan/checkpoint.")
    if len(log.samples or []) != 1:
        return {**result, "status": "error" if log.error else "incomplete", "reward": None,
                "reason": log.error.message if log.error else "No complete canonical sample."}
    sample = log.samples[0]
    if str(sample.id) != planned["case_id"]:
        raise ValueError("Canonical sample ID differs from the planned case.")
    usage = list((sample.model_usage or {}).values())
    result.update(total_tokens=sum(item.total_tokens for item in usage) if usage else None,
                  duration_seconds=sample.total_time, completed_at=sample.completed_at)
    pending_generation = False
    for event in sample.events:
        if isinstance(event, ModelEvent) and event.pending:
            pending_generation = True
        if isinstance(event, ModelEvent) and not event.error and not event.pending and not (event.call and event.call.error):
            if event.call is None or not isinstance(event.call.response, dict):
                raise ValueError("Successful generation lacks canonical serving provenance.")
            validate_server_receipt(event.call.response, checkpoint, Path(protocol["model_source"]["path"]))
    if log.status == "cancelled":
        return {**result, "status": "incomplete", "reward": None, "reason": "Operator interrupted; no outcome retry."}
    if sample.error:
        return {**result, "status": "error", "reward": None, "reason": sample.error.message}
    if sample.limit:
        score = (sample.scores or {}).get("learning_success")
        if score is not None and not (isinstance(score.value, float) and math.isnan(score.value)):
            raise ValueError("A canonical compute limit must be unscored.")
        return {**result, "status": "limit", "reward": None, "reason": str(sample.limit)}
    if pending_generation:
        return {**result, "status": "incomplete", "reward": None,
                "reason": "Native log retains an unfinished model generation."}
    record = (sample.metadata or {}).get("dealroom")
    if not record:
        return {**result, "status": "incomplete", "reward": None, "reason": "No engine action record."}
    config = protocol["evaluation"]
    if (record["workflow"] != planned["workflow"] or record["action_limit"] != config["domain_action_limit"]
            or record["compaction_threshold"] != config["compaction_threshold"]
            or record["source_fingerprints"] != protocol["source_fingerprints"]):
        raise ValueError("Canonical episode differs from frozen environment or workflow.")
    engine = replay_learning_record(record)
    if record["base_fixture_sha256"] != fixture_fingerprint(load_learning_case(planned["case_id"])):
        raise ValueError("Recorded episode fixture differs from the frozen curriculum fixture.")
    evaluation = evaluate(engine, end_on_expiry=record.get("termination_reason") == "contract_deadline_expired")
    score = (sample.scores or {}).get("learning_success")
    if evaluation.outcome in {"budget_exhausted", "incomplete"}:
        if score is None or not (isinstance(score.value, float) and math.isnan(score.value)):
            raise ValueError("An incomplete canonical attempt must be unscored.")
        return {**result, "status": "limit" if evaluation.outcome == "budget_exhausted" else "incomplete",
                "reward": None, "reason": evaluation.outcome}
    if score is None or score.value != evaluation.reward or score.metadata != evaluation.model_dump(mode="json"):
        raise ValueError("Canonical score/diagnostics differ from deterministic replay.")
    return {**result, "status": "success" if evaluation.success else "failure", "reward": evaluation.reward,
            "reason": "All verifier predicates passed." if evaluation.success else " ".join(
                item.reason for item in evaluation.diagnostics if not item.passed)}


class Experiment:
    def __init__(self, directory: Path):
        self.directory = directory
        self.protocol = read_json(directory / "protocol.json")
        self.state = read_json(directory / "state.json")
        if self.state["protocol_sha256"] != fingerprint(self.protocol):
            raise ValueError("State belongs to another protocol.")
        self.model_path = Path(self.protocol["model_source"]["path"])
        self.startup_implementation = {str(path): file_digest(ROOT / path) for path in IMPLEMENTATION_FILES}

    def save(self) -> None:
        atomic_json(self.directory / "state.json", self.state)
        publish_report(self.directory, report_from_state(self.protocol, self.state))

    def event(self, message: str) -> None:
        self.state["events"].append({"at": now(), "message": message})
        print(message, flush=True)
        self.save()

    def verify_inputs(self, *, freeze_implementation: bool = True) -> None:
        if load_manifest()["fingerprint"] != self.protocol["curriculum_fingerprint"]:
            raise ValueError("Curriculum changed after protocol freeze.")
        if learning_source_fingerprints() != self.protocol["source_fingerprints"]:
            raise ValueError("Environment changed after protocol freeze.")
        for split, digest in self.protocol["data_files"].items():
            if file_digest(ROOT / "training" / "data" / f"{split}.jsonl") != digest:
                raise ValueError("Supervised data changed after protocol freeze.")
        if freeze_implementation:
            implementation = {str(path): file_digest(ROOT / path) for path in IMPLEMENTATION_FILES}
            if implementation != self.startup_implementation:
                raise RuntimeError("Implementation changed while this process was queued; restart the queued process before GPU work.")
            frozen_path = self.directory / "implementation.json"
            if frozen_path.exists() and read_json(frozen_path) != implementation:
                raise ValueError("Implementation changed after the first run; use a new experiment version.")
            if not frozen_path.exists():
                atomic_json(frozen_path, implementation)
            registry = ROOT / "logs" / "learning" / self.protocol["experiment_id"] / "experiment.json"
            owner = {"directory": str(self.directory.resolve()), "protocol_sha256": self.state["protocol_sha256"]}
            if registry.exists() and read_json(registry) != owner:
                raise ValueError("Experiment ID already belongs to another directory or protocol; use a new --id.")
            if not registry.exists():
                atomic_json(registry, owner)
        inventory = {str(path.relative_to(self.model_path)): file_digest(path)
                     for path in sorted(self.model_path.iterdir())
                     if path.suffix in {".json", ".safetensors"} and path.is_file()}
        if not inventory or "config.json" not in inventory or not any(name.endswith(".safetensors") for name in inventory):
            raise ValueError("Downloaded base weights/configuration are incomplete.")
        frozen_model = self.directory / "base_model.json"
        if frozen_model.exists() and read_json(frozen_model) != inventory:
            raise ValueError("Base model weights or tokenizer changed.")
        if not frozen_model.exists():
            atomic_json(frozen_model, inventory)
        digest = fingerprint(inventory)
        base = {"id": "mlx-base", "digest": digest, "step": 0, "policy_fingerprint": digest}
        for stage in ("original", "workflow"):
            if stage in self.state["checkpoints"] and self.state["checkpoints"][stage] != base:
                raise ValueError("Base checkpoint changed.")
            self.state["checkpoints"][stage] = base
        self.save()

    def wait_for_baseline(self, poll_seconds: float = 20) -> None:
        if not BASELINE_REPORT.exists():
            raise RuntimeError("Original benchmark report is missing; cannot establish GPU handoff.")
        while not baseline_is_complete(read_json(BASELINE_REPORT)):
            if not baseline_runner_active():
                raise RuntimeError("Original benchmark is incomplete and its runner is not active; resume it first.")
            self.state["status"] = "preparing"
            self.state["current"] = {"phase": "waiting_for_original_benchmark", "benchmark": BASELINE_ID}
            self.save()
            time.sleep(min(60, max(1, poll_seconds)))
        # Completion publication can precede the benchmark process exit slightly.
        while baseline_runner_active():
            time.sleep(min(10, max(1, poll_seconds)))
        self.event("Original benchmark complete. Acquiring the GPU for the predeclared learning pipeline.")
        try:
            loaded = api_request("/api/ps", origin="http://127.0.0.1:11434").get("models", [])
            for item in loaded:
                if item.get("name") in {"llama3.1:8b", "qwen3:8b"}:
                    api_request("/api/generate", {"model": item["name"], "keep_alive": 0}, origin="http://127.0.0.1:11434")
        except OSError:
            pass  # An already stopped Ollama server holds no weights.

    def recover_children(self) -> None:
        path = self.directory / "child.json"
        if path.exists():
            stop_owned_process(read_json(path))
            path.unlink()

    @contextlib.contextmanager
    def server(self, checkpoint: dict):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", 8087)) == 0:
                raise RuntimeError("Port 8087 is occupied by a server this runner does not own.")
        command = [str(TRAINING_PYTHON), "-u", "-m", "dealroom.learning_serve", "--model", str(self.model_path),
                   "--host", "127.0.0.1", "--port", "8087", "--decode-concurrency", "1", "--prompt-concurrency", "1",
                   "--chat-template-args", json.dumps(self.protocol["chat_template_args"])]
        if checkpoint.get("path"):
            command += ["--adapter-path", checkpoint["path"]]
        environment = {**os.environ, "DEALROOM_POLICY_FINGERPRINT": checkpoint["policy_fingerprint"]}
        log_path = self.directory / "server.log"
        with log_path.open("a") as output:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT)
            child = {"kind": "server", "identity": process_identity(process.pid), "command": command}
            atomic_json(self.directory / "child.json", child)
            try:
                deadline = time.monotonic() + 300
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"MLX server exited; inspect {log_path}.")
                    try:
                        health = api_request("/health")
                        if health.get("status") == "ok":
                            break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError("MLX server did not become ready in five minutes.")
                    time.sleep(2)
                # Verify the actual loaded adapter; upstream default_model adapter
                # lookup can otherwise silently serve the base checkpoint.
                canary = {"model": "default_model", "messages": [{"role": "user", "content": "Reply OK."}],
                          "temperature": 0, "max_tokens": 1, "stream": False}
                if checkpoint.get("path"):
                    canary["adapters"] = checkpoint["path"]
                receipt = api_request("/v1/chat/completions", canary)
                validate_server_receipt(receipt, checkpoint, self.model_path)
                atomic_json(self.directory / f"server-canary-{checkpoint['id']}.json", receipt)
                yield
            finally:
                stop_child(process)
                (self.directory / "child.json").unlink(missing_ok=True)

    def attempt(self, stage: str, split: str, case_id: str, checkpoint: dict,
                *, group: int | None = None, rollout: int | None = None) -> dict:
        workflow = "original" if stage == "original" else "clarified"
        attempt_id = f"{stage}-{split}-{case_id}" if group is None else f"rl-group-{group:02d}-rollout-{rollout:02d}"
        planned = {"id": attempt_id, "stage": stage, "split": split, "case_id": case_id, "workflow": workflow}
        if split == "sealed_test" and self.state["test_status"] != "opened":
            raise PermissionError("Test release is required before a sealed attempt.")
        if case_id not in self.protocol["cases"][split]:
            raise PermissionError("Case is outside the planned split.")
        log_dir = ROOT / "logs" / "learning" / self.protocol["experiment_id"] / attempt_id
        existing = sorted(log_dir.glob("*.eval")) if log_dir.exists() else []
        if len(existing) > 1:
            raise ValueError("Multiple canonical logs for one planned attempt; refusing outcome selection.")
        previous = self.state["attempts"].get(attempt_id)
        if existing:
            if previous and previous.get("log_sha256") and previous["log_sha256"] != file_digest(existing[0]):
                raise ValueError("A recorded canonical log changed.")
            log = read_eval_log(existing[0], resolve_attachments="full")
        elif previous:
            if previous["status"] in TERMINAL_STATUSES:
                return previous
            result = {**planned, "status": "incomplete", "reward": None,
                      "reason": "Interrupted before a canonical log was finalized; no outcome retry."}
            self.state["attempts"][attempt_id] = result
            self.save()
            return result
        else:
            self.state["attempts"][attempt_id] = {**planned, "status": "running", "started_at": now()}
            self.state["current"] = {"phase": "rollout" if group is not None else "evaluation", **planned}
            self.save()
            config = self.protocol["evaluation"]
            task = learning_dealroom(case_id, workflow=workflow, action_limit=config["domain_action_limit"],
                                     end_on_expiry=True, compaction_threshold=config["compaction_threshold"])
            extra = {"top_k": 0, "min_p": 0, "repetition_penalty": 0.0}
            if checkpoint.get("path"):
                extra["adapters"] = checkpoint["path"]
            logs = inspect_eval(
                task, model=MODEL_ID, model_base_url=BASE_URL,
                model_args={"api_key": "local", "emulate_tools": True, "stream": False},
                log_dir=str(log_dir), display="none", log_model_api=True,
                metadata={"run_kind": "learning_rollout" if group is not None else "learning_evaluation",
                          "protocol_sha256": self.state["protocol_sha256"], "attempt_id": attempt_id,
                          "policy_fingerprint": checkpoint["policy_fingerprint"], "stage": stage, "split": split},
                max_samples=1, max_connections=1, max_retries=0, cache=False,
                temperature=1 if group is not None else 0, top_p=1,
                frequency_penalty=0, presence_penalty=0,
                logprobs=True if group is not None else False, extra_body=extra,
                parallel_tool_calls=False, max_tokens=config["output_tokens_per_turn"],
                token_limit=config["max_tokens"], turn_limit=config["max_turns"], message_limit=config["max_messages"],
                timeout=config["time_limit_seconds"], time_limit=config["time_limit_seconds"],
                seed=config["seed"] + (group * 4 + rollout if group is not None else 0),
                fail_on_error=False, score_on_error=False,
            )
            if len(logs) != 1:
                raise RuntimeError("Expected one canonical attempt log.")
            log = read_eval_log(logs[0].location, resolve_attachments="full")
        result = summarize_learning_log(log, planned, checkpoint, self.protocol)
        self.state["attempts"][attempt_id] = result
        self.save()
        print(f"{attempt_id}: {result['status']}", flush=True)
        return result

    def evaluate_split(self, stage: str, split: str) -> None:
        checkpoint = self.state["checkpoints"][stage]
        self.state["stage_status"][stage] = "running"
        self.save()
        for case_id in self.protocol["cases"][split]:
            self.attempt(stage, split, case_id, checkpoint)
        self.state["stage_status"][stage] = "complete"
        selected = [item for item in self.state["attempts"].values() if item.get("stage") == stage and item.get("split") == split]
        scored = [item for item in selected if item["status"] in {"success", "failure"}]
        if scored:
            point = {"stage": stage, "step": checkpoint.get("step", 0), "metric": "success_rate",
                     "value": sum(item["reward"] for item in scored) / len(scored), "split": split}
            if point not in self.state["curves"]:
                self.state["curves"].append(point)
        self.event(f"{LABELS[stage]} {split} evaluation recorded: {len(scored)}/{len(selected)} tasks scored.")

    def verify_tokenized_data(self) -> None:
        manifest_path = ROOT / "training" / "tokenized" / "manifest.json"
        if not manifest_path.exists():
            command = [str(TRAINING_PYTHON), "-m", "dealroom.learning_train", "prepare",
                       "--model", str(self.model_path), "--data", str(ROOT / "training" / "data"),
                       "--output", str(manifest_path.parent), "--max-length", "24576"]
            subprocess.run(command, cwd=ROOT, check=True)
        manifest = read_json(manifest_path)
        if manifest["template_args"] != self.protocol["chat_template_args"]:
            raise ValueError("Tokenized chat template differs from serving and policy-gradient tokenization.")
        for split, expected in self.protocol["data_files"].items():
            if manifest["splits"][split]["source_sha256"] != expected:
                raise ValueError("Tokenized data comes from different source demonstrations.")
        if manifest["max_length"] != self.protocol["evaluation"]["max_training_sequence_length"]:
            raise ValueError("Tokenization maximum length differs from protocol.")
        for split, digest in self.protocol["tokenized_files"].items():
            if file_digest(manifest_path.parent / f"{split}.jsonl") != digest:
                raise ValueError("Tokenized training or validation rows changed after protocol freeze.")

    def train(self, job_id: str, *, mode: str, data_path: Path, parent: dict | None = None) -> dict:
        if self.state["test_status"] != "sealed":
            raise PermissionError("Training is forbidden after the sealed test has opened.")
        known = self.state["training_jobs"].get(job_id)
        if known and known.get("status") == "complete":
            if known["training"]["data_digest"] != file_digest(data_path) or known["training"]["parent_digest"] != (parent["digest"] if parent else None):
                raise ValueError("Completed training job has different data or parent.")
            self.verify_checkpoint(known["checkpoint"])
            return known["checkpoint"]
        jobs = self.directory / "checkpoints" / job_id
        # A interrupted optimizer process has no committed training receipt. Keep
        # its files, then repeat the identical planned job from its fixed parent.
        candidates = sorted(jobs.glob("attempt-*")) if jobs.exists() else []
        completed = [candidate for candidate in candidates if (candidate / "training.json").exists()]
        if len(completed) > 1:
            raise ValueError("Multiple completed training receipts for one job; refusing checkpoint selection.")
        if completed:
            return self.commit_training(job_id, completed[0], mode, data_path, parent)
        output = jobs / f"attempt-{len(candidates) + 1:03d}"
        output.mkdir(parents=True, exist_ok=False)
        if candidates:
            self.event(f"Preserving {len(candidates)} interrupted {job_id} output(s); restarting the identical job from its fixed parent.")
        args = self.protocol["sft"] if mode == "sft" else self.protocol["rl"]
        steps = self.protocol["preflight"]["steps"] if job_id == "preflight" else self.protocol["sft"]["steps"]
        command = [str(TRAINING_PYTHON), "-u", "-m", "dealroom.learning_train", mode,
                   "--model", str(self.model_path), "--data", str(data_path), "--output", str(output),
                   "--steps", str(steps if mode == "sft" else 1),
                   "--learning-rate", str(args["learning_rate"]), "--max-length", "24576",
                   "--seed", str(self.protocol["evaluation"]["seed"])]
        if mode == "sft" and job_id != "preflight":
            command += ["--validation", str(ROOT / "training" / "tokenized" / "validation.jsonl")]
        if parent:
            self.verify_checkpoint(parent)
            command += ["--parent", parent["path"]]
            optimizer = Path(parent["path"]) / "optimizer.safetensors"
            if mode == "policy" and parent["id"].startswith("rl-group-") and optimizer.exists():
                command += ["--optimizer-state", str(optimizer)]
        self.state["training_jobs"][job_id] = {"status": "running", "output": str(output), "started_at": now(),
                                              "data_digest": file_digest(data_path), "parent": parent, "command": command}
        self.state["current"] = {"phase": "training", "job_id": job_id, "output": str(output)}
        self.save()
        with (output / "process.log").open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            child = {"kind": "trainer", "identity": process_identity(process.pid), "command": command}
            atomic_json(self.directory / "child.json", child)
            try:
                while process.poll() is None:
                    if job_id != "preflight":
                        self.collect_training_curves(output, "sft" if mode == "sft" else "rl")
                    self.save()
                    time.sleep(10)
                if process.returncode:
                    raise RuntimeError(f"Trainer exited with code {process.returncode}; inspect {output / 'process.log'}.")
            finally:
                if process.poll() is None:
                    stop_child(process)
                (self.directory / "child.json").unlink(missing_ok=True)
        return self.commit_training(job_id, output, mode, data_path, parent)

    def collect_training_curves(self, output: Path, stage: str) -> None:
        if stage == "rl":
            # Policy segment microsteps are not optimizer updates. Publish one
            # actual trainer measurement only after the accumulated update commits.
            return
        path = output / "metrics.json"
        if not path.exists():
            return
        try:
            metrics = json.loads(path.read_text())
        except json.JSONDecodeError:
            return  # Trainer may be in the middle of writing this observational file.
        for item in metrics:
            step = item.get("iteration", item.get("step"))
            value = item.get("train_loss" if item.get("kind") == "train" else "val_loss", item.get("loss"))
            if type(step) is int and isinstance(value, (int, float)) and math.isfinite(value):
                point = {"stage": stage, "step": step, "metric": "policy_loss" if stage == "rl" else "cross_entropy",
                         "value": value, "split": "train" if item.get("kind") == "train" else "validation"}
                if point not in self.state["curves"]:
                    self.state["curves"].append(point)

    def commit_training(self, job_id: str, output: Path, mode: str, data_path: Path, parent: dict | None) -> dict:
        result = read_json(output / "training.json")
        if result["mode"] != mode or result["data_digest"] != file_digest(data_path):
            raise ValueError("Training receipt does not match the planned job/data.")
        if result["parent_digest"] != (parent["digest"] if parent else None):
            raise ValueError("Training checkpoint descended from a different parent.")
        if result["digest"] != file_digest(output / "adapters.safetensors"):
            raise ValueError("Training checkpoint bytes differ from the committed receipt.")
        expected_updates = (self.protocol["preflight"]["steps"] if job_id == "preflight" else self.protocol["sft"]["steps"]) if mode == "sft" else 1
        training_config = self.protocol["sft"] if mode == "sft" else self.protocol["rl"]
        if (result["learning_rate"] != training_config["learning_rate"]
                or result["seed"] != self.protocol["evaluation"]["seed"]
                or result["max_length"] != self.protocol["evaluation"]["max_training_sequence_length"]):
            raise ValueError("Training receipt learning rate, seed, or sequence budget differs from protocol.")
        adapter_config = read_json(output / "adapter_config.json")
        if (adapter_config["num_layers"] != self.protocol["sft"]["last_layers"]
                or adapter_config["lora_parameters"]["rank"] != self.protocol["sft"]["lora_rank"]
                or adapter_config["lora_parameters"]["keys"] != self.protocol["sft"]["lora_keys"]):
            raise ValueError("Adapter layout differs from the frozen LoRA protocol.")
        if result["optimizer_updates"] != expected_updates:
            raise ValueError("Optimizer update count differs from frozen training plan.")
        step = expected_updates if mode == "sft" else (parent.get("step", 0) + 1)
        checkpoint = {"id": job_id, "path": str(output), "digest": result["digest"], "step": step,
                      "created_at": now(), "parent": parent["id"] if parent else "mlx-base",
                      "policy_fingerprint": result["digest"],
                      "training_receipt_sha256": file_digest(output / "training.json"),
                      "adapter_config_sha256": file_digest(output / "adapter_config.json"),
                      "optimizer_sha256": file_digest(output / "optimizer.safetensors")}
        self.state["training_jobs"][job_id] = {"status": "complete", "checkpoint": checkpoint,
                                              "training": result, "completed_at": now()}
        if job_id != "preflight":
            self.collect_training_curves(output, "sft" if mode == "sft" else "rl")
        if mode == "policy":
            measurements = [item for item in result.get("history", []) if item.get("kind") == "train" and isinstance(item.get("train_loss"), (int, float))]
            if measurements and math.isfinite(measurements[-1]["train_loss"]):
                self.state["curves"].append({"stage": "rl", "step": step - self.protocol["sft"]["steps"],
                                             "metric": "reported_policy_training_loss", "value": measurements[-1]["train_loss"], "split": "train"})
        self.event(f"Committed {job_id}: {expected_updates} optimizer update(s), adapter SHA256 {result['digest']}.")
        return checkpoint

    def verify_checkpoint(self, checkpoint: dict) -> None:
        if checkpoint.get("path"):
            path = Path(checkpoint["path"])
            if file_digest(path / "adapters.safetensors") != checkpoint["digest"]:
                raise ValueError("Committed adapter checkpoint changed.")
            if file_digest(path / "training.json") != checkpoint["training_receipt_sha256"]:
                raise ValueError("Committed training receipt changed.")
            if file_digest(path / "adapter_config.json") != checkpoint["adapter_config_sha256"]:
                raise ValueError("Committed adapter configuration changed.")
            if file_digest(path / "optimizer.safetensors") != checkpoint["optimizer_sha256"]:
                raise ValueError("Committed optimizer state changed.")

    def preflight(self) -> None:
        if self.state["test_status"] != "sealed":
            return
        rows = [json.loads(line) for line in (ROOT / "training" / "tokenized" / "train.jsonl").read_text().splitlines()]
        ordered = sorted(rows, key=lambda row: len(row["prompt_ids"]) + len(row["completion_ids"]))
        selected = [ordered[0], ordered[-1]]
        if any(row["case_id"] not in self.protocol["cases"]["train"] or row["split"] != "train" for row in selected):
            raise PermissionError("Preflight consumes training rows only.")
        path = self.directory / "preflight.jsonl"
        payload = "".join(json.dumps(row) + "\n" for row in selected)
        if path.exists() and path.read_text() != payload:
            raise ValueError("Frozen preflight rows changed.")
        path.write_text(payload)
        self.event("Running disposable 8B memory/trainer preflight on two training examples; this checkpoint is never selected or counted as performance.")
        self.train("preflight", mode="sft", data_path=path)

    def rl_group(self, group: dict, parent: dict) -> dict:
        from transformers import AutoTokenizer

        from dealroom.learning_policy import (
            extract_policy_segments,
            make_policy_segments,
            training_reward,
        )
        from dealroom.learning_train import prompt_tokens

        key = str(group["index"])
        if key in self.state["rl_groups"]:
            recorded = self.state["rl_groups"][key]
            if recorded["parent_policy"] != parent["policy_fingerprint"]:
                raise ValueError("Resumed RL group uses a different sampled parent.")
            group_dir = self.directory / "rollout_groups" / f"group-{group['index']:02d}"
            if recorded["plan"] != group or file_digest(group_dir / "statistics.json") != recorded["statistics_sha256"]:
                raise ValueError("Committed rollout group plan/statistics changed.")
            if recorded.get("policy_data_sha256") and file_digest(group_dir / "policy.jsonl") != recorded["policy_data_sha256"]:
                raise ValueError("Committed on-policy optimization rows changed.")
            checkpoint = recorded["checkpoint"]
            self.verify_checkpoint(checkpoint)
            return checkpoint
        attempts = []
        with self.server(parent):
            for rollout in range(self.protocol["rl"]["rollouts_per_group"]):
                attempts.append(self.attempt("rl", "train", group["case_id"], parent, group=group["index"], rollout=rollout))
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        trajectories = []
        for attempt in attempts:
            trajectory = {"trajectory_id": attempt["id"], "case_id": group["case_id"],
                          "status": attempt["status"], "reward": None, "segments": []}
            if attempt["status"] in {"success", "failure"}:
                sample = read_eval_log(attempt["native_log"], resolve_attachments="full").samples[0]
                record = sample.metadata["dealroom"]
                initial, final = reset(learning_case_for_record(record)), replay_learning_record(record)
                trajectory["reward"] = training_reward(attempt["status"], attempt["reward"], initial, final,
                                                        shaping=self.protocol["rl"]["shaping"], split="train")
                trajectory["segments"] = extract_policy_segments(
                    sample, lambda messages: prompt_tokens(tokenizer, messages),
                    expected_policy_fingerprint=parent["policy_fingerprint"],
                )
            trajectories.append(trajectory)
        prepared = make_policy_segments(trajectories)
        group_dir = self.directory / "rollout_groups" / f"group-{group['index']:02d}"
        group_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(group_dir / "statistics.json", prepared["statistics"])
        checkpoint = parent
        if prepared["segments"]:
            data_path = group_dir / "policy.jsonl"
            if group["case_id"] not in self.protocol["cases"]["train"]:
                raise PermissionError("RL updates must use the training split.")
            data_path.write_text("".join(json.dumps({**row, "split": "train"}) + "\n" for row in prepared["segments"]))
            checkpoint = self.train(f"rl-group-{group['index']:02d}", mode="policy", data_path=data_path, parent=parent)
        statistics = prepared["statistics"]
        self.state["rl_groups"][key] = {"plan": group, "parent_policy": parent["policy_fingerprint"],
                                       "statistics": statistics, "checkpoint": checkpoint,
                                       "statistics_sha256": file_digest(group_dir / "statistics.json"),
                                       "policy_data_sha256": file_digest(group_dir / "policy.jsonl") if prepared["segments"] else None,
                                       "optimizer_updates": 1 if prepared["segments"] else 0,
                                       "completed_at": now()}
        if statistics["mean_reward"] is not None:
            self.state["curves"].append({"stage": "rl", "step": group["index"] + 1,
                                         "metric": "shaped_training_reward", "value": statistics["mean_reward"], "split": "train"})
        self.event(f"RL group {group['index'] + 1}: {statistics['scored_count']}/4 scored; "
                   + (f"skipped ({statistics['skip_reason']})." if statistics["skip_reason"] else "one on-policy optimizer update committed."))
        return checkpoint

    def freeze_test_release(self) -> None:
        if any(self.state["stage_status"][stage] != "complete" for stage in STAGES):
            raise PermissionError("All four validation stages must complete before opening test.")
        for stage in STAGES:
            expected = {f"{stage}-validation-{case_id}" for case_id in self.protocol["cases"]["validation"]}
            if any(key not in self.state["attempts"] or self.state["attempts"][key]["status"] not in TERMINAL_STATUSES for key in expected):
                raise PermissionError("Validation plan is incomplete.")
            self.verify_checkpoint(self.state["checkpoints"][stage])
        release = {"protocol_sha256": self.state["protocol_sha256"], "checkpoints": self.state["checkpoints"],
                   "case_ids": self.protocol["cases"]["sealed_test"], "selection": "fixed final checkpoints"}
        path = self.directory / "test_release.json"
        if path.exists() and read_json(path) != release:
            raise ValueError("Test was already opened with different frozen checkpoints.")
        if not path.exists():
            atomic_json(path, release)
        self.state["test_status"] = "opened"
        self.event("Final checkpoint hashes frozen. Opening the sealed test once for all four predeclared conditions.")

    def run(self, poll_seconds: float = 20) -> None:
        self.verify_inputs(freeze_implementation=False)
        self.recover_children()
        self.wait_for_baseline(poll_seconds)
        self.verify_inputs()
        self.verify_tokenized_data()
        if self.state["status"] == "complete":
            for checkpoint in self.state["checkpoints"].values():
                self.verify_checkpoint(checkpoint)
            self.export_analysis()
            return
        if self.state["test_status"] == "sealed":
            self.preflight()
            self.state["status"] = "baseline"
            with self.server(self.state["checkpoints"]["original"]):
                self.evaluate_split("original", "validation")
                self.evaluate_split("workflow", "validation")
            self.state["status"] = "sft"
            self.state["stage_status"]["sft"] = "running"
            self.save()
            self.state["checkpoints"]["sft"] = self.train("sft-final", mode="sft", data_path=ROOT / "training" / "tokenized" / "train.jsonl")
            with self.server(self.state["checkpoints"]["sft"]):
                self.evaluate_split("sft", "validation")
            self.state["status"] = "rl"
            self.state["stage_status"]["rl"] = "running"
            self.save()
            checkpoint = self.state["checkpoints"]["sft"]
            for group in self.protocol["rl"]["groups"][:8]:
                checkpoint = self.rl_group(group, checkpoint)
            updates = sum(item["optimizer_updates"] for key, item in self.state["rl_groups"].items() if int(key) < 8)
            trigger = updates < 2
            if "additional_rl_groups" in self.state and self.state["additional_rl_groups"] != trigger:
                raise ValueError("Recorded conditional curriculum trigger disagrees with primary update count.")
            self.state["additional_rl_groups"] = trigger
            self.save()
            if trigger:
                for group in self.protocol["rl"]["groups"][8:]:
                    checkpoint = self.rl_group(group, checkpoint)
            self.state["checkpoints"]["rl"] = checkpoint
            self.save()
            with self.server(checkpoint):
                self.evaluate_split("rl", "validation")
            self.freeze_test_release()
        else:
            released = read_json(self.directory / "test_release.json")
            if released["protocol_sha256"] != self.state["protocol_sha256"] or released["checkpoints"] != self.state["checkpoints"]:
                raise ValueError("Checkpoints/protocol changed after test release.")
        self.state["status"] = "evaluating"
        self.save()
        for stage in STAGES:
            self.verify_checkpoint(self.state["checkpoints"][stage])
            with self.server(self.state["checkpoints"][stage]):
                self.evaluate_split(stage, "sealed_test")
        self.state["status"] = "complete"
        self.state["current"] = None
        self.event("All predeclared learning stages and sealed-test attempts are recorded. Inspect scored coverage and measured gains together.")
        self.export_analysis()

    def export_analysis(self) -> None:
        from dealroom.learning_analysis import export_learning

        export_learning(report_from_state(self.protocol, self.state), self.directory / "analysis")


def run_experiment(directory: str | Path = DEFAULT_EXPERIMENT_DIR, poll_seconds: float = 20,
                   experiment_id: str | None = None) -> None:
    directory = Path(directory)
    if not (directory / "protocol.json").exists():
        prepare_experiment(directory, experiment_id)
    elif experiment_id is not None and read_json(directory / "protocol.json")["experiment_id"] != experiment_id:
        raise ValueError("Requested --id differs from the saved protocol.")
    with (directory / "experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This experiment already has a running orchestrator.") from error
        experiment = Experiment(directory)
        try:
            experiment.run(poll_seconds)
        except BaseException as error:
            experiment.recover_children()
            experiment.state["status"] = "blocked"
            experiment.state["current"] = {"phase": "needs_attention", "reason": str(error) or type(error).__name__}
            experiment.event(f"Pipeline stopped without fabricating results: {str(error) or type(error).__name__}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--directory", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--id", dest="experiment_id")
    parser.add_argument("--revise-unstarted", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20)
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare_experiment(args.directory, args.experiment_id, revise_unstarted=args.revise_unstarted), indent=2))
    else:
        if Path(sys.prefix).resolve() != TRAINING_PYTHON.parent.parent.resolve():
            os.execv(str(TRAINING_PYTHON), [str(TRAINING_PYTHON), "-u", "-m", "dealroom.learning_experiment", *sys.argv[1:]])
        run_experiment(args.directory, args.poll_seconds, args.experiment_id)


if __name__ == "__main__":
    main()
