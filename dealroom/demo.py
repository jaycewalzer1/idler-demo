"""Offline fixture validation and exports from Inspect's canonical log files."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput, get_model
from inspect_ai.tool import ToolCall

from dealroom.domain import load_case, observe, reset, step
from dealroom.score import evaluate
from dealroom.task import case_for_record, dealroom, fixture_fingerprint

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPORT = ROOT / "web" / "public" / "runs.json"


def dump(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dump(item) for item in value]
    return value


def tool_output(action: dict, index: int) -> ModelOutput:
    arguments = deepcopy(action)
    name = arguments.pop("type")
    return ModelOutput(
        model="mockllm/fixture-validation",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content="Scripted fixture validation; this is not a model decision.",
                    tool_calls=[ToolCall(id=f"action-{index}", function=name, arguments=arguments)],
                ),
                stop_reason="tool_calls",
            )
        ],
    )


def run_script(case_id: str, actions: list[dict], metadata: dict, log_dir: Path):
    """Drive the real built-in ReAct loop, seven tools and scorer with mock outputs."""
    task = dealroom(case_id=case_id)
    task.metadata.update(metadata)
    model = get_model(
        "mockllm/model",
        custom_outputs=[tool_output(a, i) for i, a in enumerate(actions)],
        memoize=False,
    )
    logs = inspect_eval(
        task,
        model=model,
        log_dir=str(log_dir),
        display="none",
        log_model_api=True,
    )
    log = logs[0]
    if log.status != "success" or not log.samples or any(s.error or s.limit for s in log.samples):
        raise RuntimeError(f"Fixture validation failed in Inspect: {log.location}")
    return log


def diagnostics(evaluation) -> list[dict]:
    data = dump(evaluation)
    result = data["diagnostics"]
    return [
        {
            "predicate": item["name"],
            "passed": item["passed"],
            "reason": item["reason"],
            "evidence": item["evidence_ids"] + item["timestamps"],
        }
        for item in result
    ]


def snapshot(engine) -> dict:
    """Map engine-authored public records to the small viewer schema."""
    public = observe(engine)
    credit, cost = public["effective_credit_cents"], public["eligible_cost_cents"]
    return {
        "time": public["now"],
        "deadline": public["deadline"],
        "disposition": public["business_status"],
        "effective_credit_cents": credit,
        "residual_cents": max(0, cost - credit) if cost is not None else None,
        "documents": [
            {"id": d["id"], "title": d["title"], "body": d["body"]} for d in public["documents"]
        ],
        "messages": [
            {
                "id": m["id"],
                "title": f"Message from {m['sender']}",
                "body": m["body"],
                "time": m["at"],
            }
            for m in public["messages"]
        ],
        "amendments": [
            {
                "id": r["id"],
                "kind": r["terms"]["type"],
                "credit_cents": r["terms"].get("amount_cents"),
                "deadline": r["terms"].get("deadline"),
                "status": r["status"],
                "required_signers": r["required_signers"],
                "signatures": [
                    {"party": s["party"], "time": s["signed_at"]}
                    for s in r["signers"]
                    if s["signed"]
                ],
            }
            for r in public["amendments"]
        ],
        "authority": public["authority"] or {},
    }


def action_label(action: dict | None, result: dict | None = None) -> str:
    if action is None:
        return "Case opened"
    kind = action["type"]
    if kind == "request":
        label = {
            "specialist_evidence": "Request specialist evidence",
            "buyer_authorization": "Request buyer authority",
            "seller_negotiation": "Negotiate with seller",
        }.get(action["request"]["intent"], "Send request")
    elif kind == "send_for_signature":
        label = f"Route {action['revision_id']} for signatures"
    elif kind == "set_disposition":
        label = f"Attempt to {action['disposition']}"
    elif kind == "draft_amendment":
        label = f"Draft {action['terms']['type']} amendment"
    else:
        label = {
            "read": "Read public evidence",
            "wait_until": "Wait for counterpart events",
            "finish": "Submit final report",
        }.get(kind, kind)
    if result and not result.get("accepted", True):
        label += " · rejected"
    return label


def make_step(
    engine, index: int, action: dict | None, result=None, *, end_on_expiry: bool = False
) -> dict:
    data = dump(result) if result is not None else None
    evaluation = dump(evaluate(engine, end_on_expiry=end_on_expiry))
    checks = diagnostics(evaluation)
    public = snapshot(engine)
    if evaluation["success"]:
        note = (
            f"Authorized {public['disposition']} is recorded by the deadline. "
            f"Effective executed credit: ${public['effective_credit_cents'] / 100:,.2f}. "
            "Every hard constraint and structured final claim is supported."
        )
    elif end_on_expiry and public["disposition"] == "expired":
        note = (
            "Terminal reward is 0: the contractual deadline expired without an authorized "
            "disposition. The episode ended on irreversible expiry; no finish call or final "
            "claims were fabricated."
        )
    elif engine.finished:
        failed = [check["reason"] for check in checks if not check["passed"]]
        note = "Terminal reward is 0. " + " ".join(failed[:3])
    elif data and not data["accepted"]:
        note = "This attempted action was rejected: " + data["reason"]
    elif public["amendments"]:
        latest = public["amendments"][-1]
        signed = {s["party"] for s in latest["signatures"]}
        missing = [p for p in latest["required_signers"] if p not in signed]
        note = (
            f"{latest['id']} is {latest['status'].replace('_', ' ')}. "
            + (
                f"Missing exact-revision signatures: {', '.join(missing)}. "
                if missing
                else "All required signatures are present. "
            )
            + f"Effective executed credit is ${public['effective_credit_cents'] / 100:,.2f}; a draft or informal agreement creates no credit."
        )
    else:
        note = "No amendment is executed at this step. Public evidence and authority must support a timely disposition; the attempt is still unfinished."
    return {
        "index": index,
        "label": action_label(action, data),
        "action": action,
        "result": data,
        "state": public,
        "reviewer": {
            "note": note,
            "diagnostics": checks,
        },
    }


def export_logs(paths: list[Path], output: Path) -> dict:
    runs, issues = [], []
    counts = dict(
        total_samples=0,
        completed_domain_outcomes=0,
        successes=0,
        failures=0,
        execution_errors=0,
        budget_exhaustions=0,
        incomplete=0,
        run_errors=0,
        cancelled_runs=0,
    )
    for path in paths:
        log = read_eval_log(path)
        meta = log.eval.metadata or {}
        if log.status in {"error", "cancelled"}:
            counts["run_errors" if log.status == "error" else "cancelled_runs"] += 1
            issues.append(
                {
                    "sample_id": path.name,
                    "status": f"run_{log.status}",
                    "reason": log.error.message if log.error else "Inspect run was cancelled.",
                }
            )
        dataset = log.eval.dataset
        sample_count = (
            len(dataset.sample_ids) if dataset.sample_ids is not None else (dataset.samples or 0)
        )
        planned = (
            log.results.total_samples
            if log.results
            else sample_count * (log.eval.config.epochs or 1)
        )
        missing = max(0, planned - len(log.samples or []))
        if missing:
            counts["total_samples"] += missing
            counts["incomplete"] += missing
            issues.append(
                {
                    "sample_id": path.name,
                    "status": "incomplete",
                    "reason": f"{missing} planned sample(s) have no completed sample record.",
                }
            )
        for sample in log.samples or []:
            counts["total_samples"] += 1
            record = (sample.metadata or {}).get("dealroom")
            problem = None
            if sample.error:
                problem = ("execution_errors", str(sample.error.message))
            elif sample.limit:
                problem = ("budget_exhaustions", str(sample.limit))
            elif not record:
                problem = ("incomplete", "No DealRoom action record in this sample.")
            if problem:
                counts[problem[0]] += 1
                issues.append(
                    {"sample_id": str(sample.id), "status": problem[0], "reason": problem[1]}
                )
                continue
            case = case_for_record(record)
            if record.get("fixture_sha256") != fixture_fingerprint(case):
                raise ValueError(
                    f"Fixture has changed since this log was recorded: {path} sample {sample.id}"
                )
            engine = reset(case)
            end_on_expiry = (
                record.get("end_on_expiry") is True
                and record.get("termination_reason") == "contract_deadline_expired"
            )
            steps = [make_step(engine, 0, None, end_on_expiry=end_on_expiry)]
            for i, action in enumerate(record["actions"], 1):
                result = step(engine, action)
                steps.append(make_step(engine, i, action, result, end_on_expiry=end_on_expiry))
            outcome = dump(evaluate(engine, end_on_expiry=end_on_expiry))
            terminal_expiry = end_on_expiry and engine.business_status == "expired"
            if outcome["outcome"] == "budget_exhausted":
                counts["budget_exhaustions"] += 1
                issues.append(
                    {
                        "sample_id": str(sample.id),
                        "status": "budget_exhaustions",
                        "reason": "Domain action limit exhausted.",
                    }
                )
                continue
            if not getattr(engine, "finished", False) and not terminal_expiry:
                counts["incomplete"] += 1
                issues.append(
                    {
                        "sample_id": str(sample.id),
                        "status": "incomplete",
                        "reason": "No structured finish report.",
                    }
                )
                continue
            recorded = (sample.scores or {}).get("transaction_success")
            if (
                recorded is None
                or recorded.value not in (0, 1)
                or recorded.value != int(outcome["success"])
            ):
                raise ValueError(
                    f"Replay score differs from canonical Inspect score: {path} sample {sample.id}"
                )
            counts["completed_domain_outcomes"] += 1
            counts["successes" if outcome["success"] else "failures"] += 1
            run_id = meta.get("run_id", f"{log.eval.eval_id}-{sample.id}-{sample.epoch}")
            is_mock = log.eval.model.startswith("mockllm/")
            initial_observation = json.loads(sample.input) if isinstance(sample.input, str) else {}
            if not isinstance(initial_observation, dict):
                initial_observation = {}
            first_request = next((event for event in sample.events if event.event == "model"), None)
            # Task inspection uses the public prompt and tool schemas recorded by
            # Inspect, not a new frontend description of the environment contract.
            task_details = {
                "brief": initial_observation.get("brief", case.brief),
                "policy": initial_observation.get("policy", case.public_policy),
                "system_prompt": "\n\n".join(
                    message.text for message in sample.messages if message.role == "system"
                ),
                "initial_observation": initial_observation,
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters.model_dump(mode="json", exclude_none=True),
                    }
                    for tool in (first_request.tools if first_request else [])
                ],
            }
            runs.append(
                {
                    "id": run_id,
                    "case_id": record["case_id"],
                    "case_title": case.title,
                    "split": case.split,
                    "kind": meta.get(
                        "run_kind", "fixture_witness" if is_mock else "recorded_model"
                    ),
                    "label": meta.get(
                        "run_label",
                        "Mock model validation"
                        if is_mock
                        else f"Recorded model · {log.eval.model}",
                    ),
                    "description": meta.get(
                        "run_description",
                        "Mock model validation, not agent performance."
                        if is_mock
                        else "Recorded model run exported from Inspect; no transcript is duplicated here.",
                    ),
                    **({"branch_step": meta["branch_step"]} if "branch_step" in meta else {}),
                    **({"paired_run_id": meta["paired_run_id"]} if "paired_run_id" in meta else {}),
                    "score": {"success": outcome["success"], "diagnostics": diagnostics(outcome)},
                    "steps": steps,
                    **(
                        {
                            "episode": {
                                "termination_reason": "contract_deadline_expired",
                                "finished": engine.finished,
                                "action_limit": case.action_limit,
                            }
                        }
                        if terminal_expiry
                        else {}
                    ),
                    "task": task_details,
                    "provenance": {
                        "inspect_log": inspect_log_path(path),
                        "model": log.eval.model,
                        "usage": dump(sample.model_usage),
                        "duration_seconds": sample.total_time,
                        "source": "Inspect public log API",
                    },
                }
            )
    bundle = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "summary": counts,
        "issues": issues,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n")
    return bundle


def inspect_log_path(path: Path) -> str:
    """Link logs beneath the project's shared native viewer root."""
    try:
        return path.resolve().relative_to((ROOT / "logs").resolve()).as_posix()
    except ValueError:
        return path.name


def offline(output: Path, log_dir: Path) -> dict:
    from dealroom.witnesses import (
        FLAGSHIP_FLAWED_SUFFIX,
        FLAGSHIP_PREFIX,
        FLAGSHIP_REPAIRED_SUFFIX,
        WITNESSES,
    )

    first, second = reset(load_case("case-01")), reset(load_case("case-01"))
    for action in FLAGSHIP_PREFIX:
        step(first, deepcopy(action))
        step(second, deepcopy(action))
    assert dump(first) == dump(second), "Branch prefix does not reproduce the same state"
    paths = []
    scripts = [
        (
            "case-01",
            FLAGSHIP_PREFIX + FLAGSHIP_FLAWED_SUFFIX,
            {
                "run_id": "flagship-failure",
                "run_kind": "scripted_failure",
                "run_label": "Scripted failure",
                "run_description": "A deliberately flawed script treats informal agreement as valid execution. Not a model failure or a fair baseline.",
                "branch_step": len(FLAGSHIP_PREFIX),
                "paired_run_id": "flagship-repaired",
            },
            False,
        ),
        (
            "case-01",
            FLAGSHIP_PREFIX + FLAGSHIP_REPAIRED_SUFFIX,
            {
                "run_id": "flagship-repaired",
                "run_kind": "repaired_script",
                "run_label": "Repaired script",
                "run_description": "Same fixture and exact action prefix, reset and replayed before the alternate signature-completing suffix.",
                "branch_step": len(FLAGSHIP_PREFIX),
                "paired_run_id": "flagship-failure",
            },
            True,
        ),
    ]
    scripts.extend(
        (
            case_id,
            actions,
            {
                "run_id": f"witness-{case_id}",
                "run_kind": "fixture_witness",
                "run_label": "Fixture witness",
                "run_description": "Fixture validation script with access to the complete case. This is not agent performance.",
            },
            True,
        )
        for case_id, actions in WITNESSES.items()
    )
    for case_id, actions, metadata, expected in scripts:
        log = run_script(case_id, actions, metadata, log_dir)
        assert bool(log.samples[0].scores["transaction_success"].value) == expected, log.location
        paths.append(Path(log.location))
        print(f"{metadata['run_id']}: {'success' if expected else 'expected failure'}", flush=True)
    bundle = export_logs(paths, output)
    print(f"Exported {len(bundle['runs'])} runs to {output}")
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Generate eight offline traces through Inspect's mock model"
    )
    demo.add_argument("--output", type=Path, default=DEFAULT_EXPORT)
    demo.add_argument("--log-dir", type=Path, default=ROOT / "logs" / "demo")
    export = commands.add_parser(
        "export", help="Export completed samples from canonical Inspect logs"
    )
    export.add_argument("logs", type=Path, nargs="+")
    export.add_argument("--output", type=Path, default=DEFAULT_EXPORT)
    args = parser.parse_args()
    if args.command == "demo":
        offline(args.output, args.log_dir)
    else:
        bundle = export_logs(args.logs, args.output)
        print(json.dumps(bundle["summary"], indent=2))


if __name__ == "__main__":
    main()
