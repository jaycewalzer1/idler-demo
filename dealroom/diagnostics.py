"""Controlled public-input ablations; never overwrite the matched benchmark.

These selected-case probes diagnose interface sensitivity, not model rankings.
They use the unchanged Inspect solver, tools, engine, scorer and expanded budgets.
Run with exclusive access to the local Ollama server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log

from dealroom.benchmark import (
    DEFAULT_CONFIG,
    ROOT,
    atomic_json,
    environment_metadata,
    local_api,
    model_metadata,
    summarize_log,
)
from dealroom.domain import observe, reset
from dealroom.task import case_for_record, dealroom

MODEL = "ollama/llama3.1:8b"
CASES = ("synth-007", "synth-025")
READ_GUIDANCE = """Resource access instructions:
The resources field is an index, not the contents. Read the listed documents and
buyer/seller messages before deciding. For each read, supply both resource
(document, message, amendment, or envelope) and resource_id copied from that
resource's index entry. Calling read with resource='summary' returns the index
again. Reads consume no simulated time.
"""
TOOL_GUIDANCE = READ_GUIDANCE + """
Tool effects:
- wait_until advances all the way to the supplied timestamp. It does not stop
  when a response arrives earlier. Obtain timing from public messages and leave
  enough simulated time for subsequent actions before the contractual deadline.
- Requests and drafts do not execute amendments. Routing requests signatures;
  check the exact revision's execution before relying on its terms.
- set_disposition commits proceed or cancel when authorized and feasible.
- finish only submits a report about the recorded state. It does not commit a
  disposition, execute an amendment, or make its claims true. Verify the state
  and the relevant tool receipts before submitting claims.
"""
CONDITIONS = ("explicit_read_guidance", "preloaded_public_evidence", "explicit_tool_semantics")


def diagnostic_task(case_id: str, condition: str):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown diagnostic condition: {condition}")
    task = dealroom(
        suite="synthetic",
        case_id=case_id,
        action_limit=DEFAULT_CONFIG["domain_action_limit"],
        end_on_expiry=DEFAULT_CONFIG["end_on_expiry"],
        compaction_threshold=DEFAULT_CONFIG["compaction_threshold"],
    )
    sample = task.dataset[0]
    baseline_input = sample.input
    if condition in {"explicit_read_guidance", "explicit_tool_semantics"}:
        supplement = READ_GUIDANCE if condition == "explicit_read_guidance" else TOOL_GUIDANCE
    else:
        # Only fields emitted by the public observation boundary, at episode start.
        public = observe(reset(case_for_record({
            "case_id": case_id, "action_limit": DEFAULT_CONFIG["domain_action_limit"]
        })))
        supplement = "Initially available public resource contents:\n" + json.dumps(
            {key: public[key] for key in ("documents", "messages", "amendments", "envelopes")},
            ensure_ascii=False,
        )
    sample.input = baseline_input + "\n\n" + supplement
    task.metadata["input_ablation"] = {
        "condition": condition,
        "baseline_input_sha256": hashlib.sha256(baseline_input.encode()).hexdigest(),
        "input_sha256": hashlib.sha256(sample.input.encode()).hexdigest(),
        "supplement": supplement,
        "private_information_supplied": False,
    }
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", choices=CONDITIONS, nargs="+", default=list(CONDITIONS[:2]))
    args = parser.parse_args()
    base_url = "http://127.0.0.1:11434/v1"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "logs" / "diagnostics" / f"llama-interface-{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    model = model_metadata([MODEL], base_url)[0]
    config = {
        **DEFAULT_CONFIG,
        "tool_calling": "inspect_text_emulation",
        "notes": "Selected-case diagnostic input ablations. Model, tools, solver, engine, scorer "
        "and numeric budgets match the recorded baseline; supplemental public input varies "
        "by the declared condition. Never merge these outcomes into the matched benchmark.",
    }
    report = {
        "id": directory.name,
        "purpose": "Selected-case public-input ablations, not a model comparison",
        "baseline_experiment": "dealroom-local-48-text-tools-v4",
        "hypotheses": {
            "explicit_read_guidance": "Does clearer resource-access documentation improve evidence retrieval?",
            "preloaded_public_evidence": "Does removing initial retrieval improve coordination?",
            "explicit_tool_semantics": "Does a plain-language restatement of public tool effects improve coordination?",
        },
        "selection": "Two failed baseline tasks: straightforward execution and deadline extension.",
        "cases": list(CASES),
        "conditions": args.conditions,
        "config": config,
        "model": model,
        "environment": environment_metadata(),
        "attempts": [],
        "status": "running",
    }
    atomic_json(directory / "report.json", report)
    for condition in args.conditions:
        for case_id in CASES:
            run_id = f"{directory.name}-{condition}-{case_id}"
            task = diagnostic_task(case_id, condition)
            print(f"DIAGNOSTIC START {condition} {case_id}", flush=True)
            logs = inspect_eval(
                task, model=MODEL, model_base_url=base_url,
                model_args={"emulate_tools": True}, log_dir=str(directory), display="none",
                metadata={
                    "run_id": run_id, "run_kind": "diagnostic_ablation",
                    "condition": condition, "model_digest": model["digest"],
                    "environment": report["environment"], "benchmark_config": config,
                    "baseline_experiment": report["baseline_experiment"],
                },
                max_samples=1, max_connections=1, temperature=0, seed=config["seed"],
                max_tokens=config["output_tokens_per_turn"],
                extra_body={"reasoning_effort": "none"}, parallel_tool_calls=False,
                cache=False, max_retries=0, timeout=config["time_limit_seconds"],
                time_limit=config["time_limit_seconds"], token_limit=config["max_tokens"],
                message_limit=config["max_messages"], turn_limit=config["max_turns"],
                fail_on_error=False, score_on_error=False, log_model_api=True,
            )
            if len(logs) != 1:
                raise RuntimeError("Expected one canonical diagnostic log.")
            log = read_eval_log(logs[0].location)
            attempt = summarize_log(log, {"id": run_id, "task_id": case_id, "model_id": MODEL})
            sample = log.samples[0]
            record = sample.metadata.get("dealroom", {})
            attempt["condition"] = condition
            attempt["resource_reads"] = [
                action for action in record.get("actions", []) if action["type"] == "read"
            ]
            attempt["waits"] = [
                action for action in record.get("actions", []) if action["type"] == "wait_until"
            ]
            loaded = local_api(base_url, "/api/ps")["models"]
            served = next(item for item in loaded if item["digest"] == model["digest"])
            attempt["served_context_window"] = served["context_length"]
            if served["context_length"] != config["context_window"]:
                raise ValueError("Diagnostic serving context differs from baseline.")
            report["attempts"].append(attempt)
            atomic_json(directory / "report.json", report)
            print("DIAGNOSTIC DONE " + json.dumps(attempt), flush=True)
    report["status"] = "complete"
    atomic_json(directory / "report.json", report)
    print(f"DIAGNOSTIC REPORT {directory / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
