"""Run a matched local-model evaluation using Inspect, with resumable live exports.

This is orchestration and reporting around the existing Inspect task. No model
responses are synthesized, and no private witness is supplied to a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog, read_eval_log

from dealroom.demo import ROOT, export_logs, inspect_log_path
from dealroom.domain import reset, step
from dealroom.score import evaluate
from dealroom.task import case_for_record, dealroom, fixture_fingerprint

DEFAULT_MODELS = ["ollama/llama3.1:8b", "ollama/qwen3:8b"]
DEFAULT_ID = "dealroom-local-48-native-v4"
FINAL_STATUSES = {"success", "failure", "error", "limit", "incomplete"}
DEFAULT_CONFIG = {
    "temperature": 0,
    "max_messages": 512,
    "max_turns": 128,
    "max_tokens": 2_000_000,
    "output_tokens_per_turn": 2048,
    "time_limit_seconds": 900,
    "context_window": 40960,
    "domain_action_limit": 256,
    "compaction_threshold": 28672,
    "compaction_preserve": 0.5,
    "end_on_expiry": True,
    "score_policy": "canonical_unscored_limits",
    "thinking": False,
    "seed": 20260927,
    "notes": (
        "Matched experiment with expanded budgets for both models. "
        "One new attempt per task/model; identical tools, prompt "
        "and limits. Qwen thinking disabled. No generation cache or outcome retries within "
        "this experiment. Native deterministic history trimming preserves system and initial "
        "task messages and half of later messages at the declared token threshold. "
        "Irreversible contractual expiry ends an episode as an objective task failure; "
        "compute limits are canonically unscored in Inspect and excluded from the "
        "scored-task success rate. "
        "Tokens include repeated input context. Duration includes local serving overhead; "
        "this is not a controlled hardware-speed benchmark. Planned coverage and all "
        "unscored outcomes are reported separately. Synthetic variants are not independent "
        "real-world tasks."
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_metadata() -> dict:
    source = Path(__file__).parent
    return {
        "inspect_ai_version": version("inspect-ai"),
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("domain.py", "task.py", "score.py")
        },
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def local_api(base_url: str, route: str, payload: dict | None = None) -> dict:
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("This runner uses a loopback Ollama server for downloaded models.")
    origin = f"{parsed.scheme}://{parsed.netloc}"
    request = Request(
        origin + route,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def model_metadata(model_ids: list[str], base_url: str) -> list[dict]:
    if not model_ids or len(model_ids) != len(set(model_ids)):
        raise ValueError("Model identifiers must be nonempty and unique.")
    available = {item["name"]: item for item in local_api(base_url, "/api/tags")["models"]}
    result = []
    for model_id in model_ids:
        if not model_id.startswith("ollama/"):
            raise ValueError(f"Expected a downloaded ollama/model identifier: {model_id}")
        name = model_id.removeprefix("ollama/")
        if name not in available:
            raise ValueError(f"Model {name} is not downloaded. Run ollama pull {name} first.")
        metadata = available[name]
        details = local_api(base_url, "/api/show", {"model": name})
        if "tools" not in details.get("capabilities", []):
            raise ValueError(f"Model {name} does not advertise native tool support.")
        result.append(
            {
                "id": model_id,
                "label": {"qwen3:8b": "Qwen3 8B", "llama3.1:8b": "Llama 3.1 8B"}.get(name, name),
                "provider": "Ollama · local",
                "parameters": metadata.get("details", {}).get("parameter_size", ""),
                "quantization": metadata.get("details", {}).get("quantization_level", ""),
                "digest": metadata["digest"],
                "size_bytes": metadata["size"],
                "capabilities": details.get("capabilities", []),
            }
        )
    return result


def create_report(manifest: dict, models: list[dict], benchmark_id: str, config: dict) -> dict:
    # Interleave families, ensuring early progress covers each behavior.
    tasks = sorted(manifest["tasks"], key=lambda item: (item["variant"], item["id"]))
    if not tasks or len({item["id"] for item in tasks}) != len(tasks):
        raise ValueError("Task identifiers must be nonempty and unique.")
    if not models or len({item["id"] for item in models}) != len(models):
        raise ValueError("Model identifiers must be nonempty and unique.")
    tasks = [
        {key: item[key] for key in ("id", "title", "split", "family", "variant")} for item in tasks
    ]
    return {
        "schema_version": 1,
        "id": benchmark_id,
        "title": (
            "DealRoom · Inspect built-in text-tool evaluation"
            if config.get("tool_calling") == "inspect_text_emulation"
            else "DealRoom · expanded-budget model evaluation"
        ),
        "generated_at": now(),
        "status": "running",
        "seed": manifest["seed"],
        "suite_fingerprint": manifest["fingerprint"],
        "environment": environment_metadata(),
        "task_count": len(tasks),
        "models": models,
        "tasks": tasks,
        "config": config,
        "attempts": [
            {
                "id": f"{model['id']}::{task['id']}",
                "task_id": task["id"],
                "model_id": model["id"],
                "status": "pending",
            }
            for model in models
            for task in tasks
        ],
    }


def summarize_log(log: EvalLog, attempt: dict) -> dict:
    # Rebuild derived values exclusively from the canonical log on every resume.
    result = {key: attempt[key] for key in ("id", "task_id", "model_id")}
    result["inspect_log"] = inspect_log_path(Path(log.location))
    result["completed_at"] = log.stats.completed_at or now()
    if log.eval.model != attempt["model_id"]:
        raise ValueError("The Inspect log does not match the planned model.")
    samples = log.samples or []
    if len(samples) != 1:
        result.update(
            status="error" if log.error else "incomplete",
            reason=log.error.message if log.error else "No complete sample record was written.",
            reward=None,
        )
        return result
    sample = samples[0]
    if sample.started_at:
        result["started_at"] = sample.started_at
    if sample.completed_at:
        result["completed_at"] = sample.completed_at
    if str(sample.id) != attempt["task_id"]:
        raise ValueError("The Inspect log does not match the planned task.")
    usage = list((sample.model_usage or {}).values())
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        result[name] = sum(getattr(item, name) for item in usage) if usage else None
    result["duration_seconds"] = sample.total_time
    result["turns"] = sample.turn_count
    if sample.events is not None:
        tool_events = [event for event in sample.events if event.event == "tool"]
        result["tool_calls"] = len(tool_events)
        result["tool_errors"] = sum(event.error is not None for event in tool_events)
        result["generations"] = sum(event.event == "model" for event in sample.events)
    record = (sample.metadata or {}).get("dealroom")
    if record and isinstance(record.get("actions"), list):
        result["actions"] = len(record["actions"])
    if record and record.get("case_id") != attempt["task_id"]:
        raise ValueError("The DealRoom action record does not match the planned task.")
    declared = (log.eval.metadata or {}).get("benchmark_config", {})
    if record and "domain_action_limit" in declared:
        if (
            record.get("action_limit") != declared["domain_action_limit"]
            or record.get("end_on_expiry", False) != declared["end_on_expiry"]
            or record.get("compaction_threshold") != declared["compaction_threshold"]
        ):
            raise ValueError("The recorded episode policy differs from its declared configuration.")
    if sample.error:
        result.update(status="error", reason=sample.error.message, reward=None)
    elif sample.limit:
        result.update(status="limit", reason=str(sample.limit), reward=None)
    elif not record:
        result.update(status="incomplete", reason="No DealRoom action record.", reward=None)
    else:
        case = case_for_record(record)
        if record.get("fixture_sha256") != fixture_fingerprint(case):
            raise ValueError("The fixture changed after this evaluation was recorded.")
        engine = reset(case)
        for action in record["actions"]:
            step(engine, action)
        terminal_expiry = (
            record.get("end_on_expiry") is True
            and record.get("termination_reason") == "contract_deadline_expired"
            and engine.business_status == "expired"
        )
        evaluated = evaluate(engine, end_on_expiry=terminal_expiry)
        if terminal_expiry:
            result["termination_reason"] = "contract_deadline_expired"
        if evaluated.outcome == "budget_exhausted":
            result.update(status="limit", reason="Domain action limit exhausted.", reward=None)
        elif not engine.finished and not terminal_expiry:
            result.update(status="incomplete", reason="No accepted structured finish.", reward=None)
        else:
            score = (sample.scores or {}).get("transaction_success")
            if score is None or score.value not in (0, 1) or score.value != evaluated.reward:
                raise ValueError("Replay outcome does not match the canonical Inspect score.")
            result.update(
                status="success" if evaluated.success else "failure",
                reward=int(evaluated.success),
                run_id=(log.eval.metadata or {})["run_id"],
                reason=(
                    "Every verifier predicate passed."
                    if evaluated.success
                    else (
                        "Contractual deadline expired; the episode can no longer succeed. "
                        if terminal_expiry
                        else ""
                    )
                    + " ".join(item.reason for item in evaluated.diagnostics if not item.passed)
                ),
            )
    return result


def refresh_replays(report: dict, output: Path) -> None:
    paths = sorted((ROOT / "logs" / "demo").glob("*.eval"))
    paths += [
        ROOT / "logs" / item["inspect_log"]
        for item in report["attempts"]
        if item.get("inspect_log")
    ]
    temporary = output.with_suffix(".building.json")
    export_logs(list(dict.fromkeys(paths)), temporary)
    temporary.replace(output)


def publish(report: dict, output: Path) -> None:
    report["generated_at"] = now()
    atomic_json(output, report)
    # Preview serves dist; preserve live progress without repeated Vite builds.
    if output.parent == ROOT / "web" / "public" and (ROOT / "web" / "dist").is_dir():
        atomic_json(ROOT / "web" / "dist" / output.name, report)


def restore_served_context(candidate: dict, persisted: dict, config: dict) -> None:
    """Keep the report's serving observation separate from canonical outcome replay.

    Inspect records the declared context configuration, not the context measured
    from the serving process after inference. A missing historical observation
    cannot be reconstructed by querying whichever model happens to be loaded now.
    """
    if not candidate["model_id"].startswith("ollama/") or not candidate.get("total_tokens"):
        return
    measured = persisted.get("served_context_window")
    if measured is None:
        raise ValueError(
            f"No persisted served-context observation for {candidate['id']}. "
            "The canonical log cannot establish the historical serving context. "
            "Restore the original report observation before resuming; this attempt will not be rerun."
        )
    if type(measured) is not int or measured != config["context_window"]:
        raise ValueError(
            f"Persisted served context for {candidate['id']} differs from the frozen "
            "benchmark configuration. This attempt will not be rerun."
        )
    candidate["served_context_window"] = measured


def run_benchmark(args) -> dict:
    from dealroom.synthesis import verify_manifest

    manifest = json.loads(args.manifest.read_text())
    # The Inspect task resolves IDs against project cases/generated, regardless
    # of where a copy of the manifest was supplied from.
    verify_manifest(manifest)
    models = model_metadata(args.models, args.base_url)
    config = {**DEFAULT_CONFIG, "time_limit_seconds": args.time_limit}
    emulate_tools = getattr(args, "emulate_tools", False)
    if emulate_tools:
        config["tool_calling"] = "inspect_text_emulation"
        config["notes"] += (
            " This experiment uses Inspect's built-in text-tool format instead of "
            "the model provider's native tool-call format. Inspect supplies the tool "
            "instructions and parses the model's textual tool calls."
        )
    expected = create_report(manifest, models, args.id, config)
    if args.output.exists():
        report = json.loads(args.output.read_text())
        if report["id"] != args.id or report["suite_fingerprint"] != manifest["fingerprint"]:
            raise ValueError(
                "Existing report belongs to a different suite. Use a new --output and --id."
            )
        if report["models"] != models or report["config"] != config:
            raise ValueError("Model digests or evaluation settings changed. Use a new report.")
        if report.get("environment") != expected["environment"]:
            raise ValueError(
                "The task, engine, scorer, or Inspect version changed. Use a new report."
            )
        if report["tasks"] != expected["tasks"] or report["seed"] != expected["seed"]:
            raise ValueError("The report task matrix or seed differs from the validated suite.")
        planned = [(item["id"], item["task_id"], item["model_id"]) for item in expected["attempts"]]
        actual = [(item["id"], item["task_id"], item["model_id"]) for item in report["attempts"]]
        if actual != planned:
            raise ValueError("The report attempt matrix differs from the planned evaluation.")
    else:
        report = expected
    runtime = {"ollama_version": local_api(args.base_url, "/api/version")["version"]}
    if report.get("runtime", runtime) != runtime:
        raise ValueError("The local serving version changed. Use a new report.")
    report["runtime"] = runtime
    log_dir = ROOT / "logs" / "benchmark" / args.id
    log_dir.mkdir(parents=True, exist_ok=True)
    # Recover a finished canonical record if the UI publication was interrupted.
    recovered = {}
    planned_ids = {item["id"] for item in report["attempts"]}
    for path in log_dir.glob("*.eval"):
        log = read_eval_log(path)
        key = (log.eval.metadata or {}).get("attempt_id")
        if key:
            if key not in planned_ids:
                raise ValueError("Canonical log references an unknown attempt_id.")
            if (log.eval.metadata or {}).get("benchmark_id") != args.id:
                raise ValueError("Canonical log belongs to another benchmark.")
            metadata = log.eval.metadata or {}
            if (log.eval.model_args or {}).get("emulate_tools", False) is not emulate_tools:
                raise ValueError("Canonical log has a different tool-calling mode.")
            if (
                any(
                    metadata.get(name) != report[name]
                    for name in ("environment", "suite_fingerprint")
                )
                or metadata.get("benchmark_config") != config
            ):
                raise ValueError("Canonical log has a different environment or evaluation config.")
            expected_model = next((m for m in models if m["id"] == log.eval.model), None)
            if expected_model is None or metadata.get("model_digest") != expected_model.get(
                "digest"
            ):
                raise ValueError("Canonical log has a different model digest.")
            if key in recovered:
                raise ValueError(
                    f"Duplicate canonical attempts for {key}; inspect before resuming."
                )
            recovered[key] = log
    for index, attempt in enumerate(report["attempts"]):
        if attempt["id"] in recovered:
            candidate = summarize_log(recovered[attempt["id"]], attempt)
            restore_served_context(candidate, attempt, config)
            report["attempts"][index] = candidate
        elif attempt["status"] in FINAL_STATUSES:
            raise ValueError(
                f"Canonical Inspect log is missing for completed attempt {attempt['id']}."
            )
        elif attempt["status"] == "running":
            attempt["status"] = "pending"
    report["status"] = "running"
    refresh_replays(report, args.runs_output)
    if args.runs_output.parent == ROOT / "web" / "public":
        publish(json.loads(args.runs_output.read_text()), args.runs_output)
    publish(report, args.output)
    if args.initialize_only:
        return report
    # Small model batches avoid reloading weights for every task while exposing
    # progress for both models early. Each model receives the same task order.
    order = []
    for offset in range(0, len(report["tasks"]), 8):
        for model in models:
            order += [
                f"{model['id']}::{task['id']}" for task in report["tasks"][offset : offset + 8]
            ]
    by_id = {item["id"]: index for index, item in enumerate(report["attempts"])}
    executed = 0
    try:
        for key in order:
            index = by_id[key]
            attempt = report["attempts"][index]
            if attempt["status"] in FINAL_STATUSES:
                continue
            if args.max_attempts is not None and executed >= args.max_attempts:
                break
            attempt.update(status="running", started_at=now())
            publish(report, args.output)
            slug = re.sub(r"[^a-zA-Z0-9_-]", "-", attempt["model_id"])
            run_id = f"{args.id}-{slug}-{attempt['task_id']}"
            started = time.monotonic()
            print(f"START {attempt['model_id']} {attempt['task_id']}", flush=True)
            task = dealroom(
                suite="synthetic",
                case_id=attempt["task_id"],
                action_limit=config["domain_action_limit"],
                end_on_expiry=config["end_on_expiry"],
                compaction_threshold=config["compaction_threshold"],
            )
            logs = inspect_eval(
                task,
                model=attempt["model_id"],
                model_base_url=args.base_url,
                model_args={"emulate_tools": True} if emulate_tools else {},
                log_dir=str(log_dir),
                display="none",
                metadata={
                    "benchmark_id": args.id,
                    "attempt_id": attempt["id"],
                    "environment": report["environment"],
                    "suite_fingerprint": report["suite_fingerprint"],
                    "benchmark_config": config,
                    "model_digest": next(
                        m["digest"] for m in models if m["id"] == attempt["model_id"]
                    ),
                    "run_id": run_id,
                    "run_kind": "recorded_model",
                    "run_label": next(m["label"] for m in models if m["id"] == attempt["model_id"]),
                    "run_description": (
                        "Real local-model inference on a generated task. Canonical Inspect "
                        "record; no scripted actions or private witness supplied."
                        + (
                            " Tool calling uses Inspect's built-in text-tool format."
                            if emulate_tools
                            else ""
                        )
                    ),
                },
                max_samples=1,
                max_connections=1,
                temperature=0,
                seed=config["seed"],
                max_tokens=config["output_tokens_per_turn"],
                extra_body={"reasoning_effort": "none"},
                parallel_tool_calls=False,
                cache=False,
                max_retries=0,
                timeout=config["time_limit_seconds"],
                time_limit=config["time_limit_seconds"],
                token_limit=config["max_tokens"],
                message_limit=config["max_messages"],
                turn_limit=config["max_turns"],
                fail_on_error=False,
                score_on_error=False,
                log_model_api=True,
            )
            if len(logs) != 1:
                raise RuntimeError("Inspect did not return exactly one evaluation log.")
            candidate = summarize_log(read_eval_log(logs[0].location), attempt)
            if attempt["model_id"].startswith("ollama/") and candidate.get("total_tokens"):
                loaded = local_api(args.base_url, "/api/ps").get("models", [])
                served = next(
                    (m for m in loaded if m["name"] == attempt["model_id"].removeprefix("ollama/")),
                    None,
                )
                measured = served.get("context_length") if served is not None else None
                if type(measured) is not int or measured != config["context_window"]:
                    raise ValueError(
                        "Ollama's served context is unknown or differs from the declared "
                        "benchmark configuration. The attempt remains unvalidated; its "
                        "canonical log is retained and it will not be rerun on resume."
                    )
                candidate["served_context_window"] = measured
            # Do not publish a terminal result until its serving observation is valid.
            report["attempts"][index] = candidate
            executed += 1
            refresh_replays(report, args.runs_output)
            if args.runs_output.parent == ROOT / "web" / "public":
                publish(json.loads(args.runs_output.read_text()), args.runs_output)
            publish(report, args.output)
            counts = Counter(item["status"] for item in report["attempts"])
            print(
                f"DONE {key}: {report['attempts'][index]['status']} ({time.monotonic() - started:.1f}s) {dict(counts)}",
                flush=True,
            )
    finally:
        report["status"] = (
            "complete"
            if all(item["status"] in FINAL_STATUSES for item in report["attempts"])
            else "interrupted"
        )
        publish(report, args.output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--id", default=DEFAULT_ID)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--manifest", type=Path, default=ROOT / "cases/generated/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "web/public/benchmark.json")
    parser.add_argument("--runs-output", type=Path, default=ROOT / "web/public/runs.json")
    parser.add_argument("--time-limit", type=int, default=DEFAULT_CONFIG["time_limit_seconds"])
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument(
        "--emulate-tools",
        action="store_true",
        help="Use Inspect's built-in text-tool format in a separate named experiment.",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.id):
        parser.error("Benchmark ID must contain only letters, digits, underscores, or hyphens.")
    if args.time_limit <= 0 or (args.max_attempts is not None and args.max_attempts < 0):
        parser.error("Limits must be positive (max-attempts may be zero).")
    if args.emulate_tools and args.id == DEFAULT_ID:
        parser.error("--emulate-tools requires a new experiment --id; use a separate --output.")
    report = run_benchmark(args)
    print(json.dumps(Counter(item["status"] for item in report["attempts"]), indent=2))


if __name__ == "__main__":
    main()
