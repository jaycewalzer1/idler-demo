"""Case-paired analysis of recorded learning attempts; never loads task answers.

Rates use scored attempts only. Limits, errors, and unfinished computation remain
visible in coverage and costs. Synthetic-case bootstrap intervals describe this
suite's paired scored subset, not a general model ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

STAGES = ("original", "workflow", "sft", "rl")
SPLITS = ("train", "validation", "sealed_test")
SCORED_STATUSES = frozenset({"success", "failure"})
STATUSES = ("success", "failure", "limit", "error", "incomplete", "running", "pending")
COMPARISONS = (("original", "workflow"), ("workflow", "sft"), ("sft", "rl"))
DEFAULT_BOOTSTRAP_SEED = 20260928
BOOTSTRAP_ASSUMPTIONS = (
    "Percentile bootstrap resamples paired synthetic cases as exchangeable observations. "
    "Shared generation templates can violate independence. The interval is conditional on "
    "this synthetic suite and the subset scored by both conditions; it is not a general "
    "model ranking and does not include training-seed or sampling-seed uncertainty."
)


def _attempts(report: Mapping[str, Any]) -> list[dict]:
    raw = report.get("attempts", [])
    if not isinstance(raw, list):
        raise ValueError("Learning report attempts must be a list of actual records")
    result, ids, evaluation_keys = [], set(), set()
    for attempt in raw:
        if not isinstance(attempt, dict):
            raise ValueError("Every attempt must be an object")
        attempt = dict(attempt)
        for source, target in (
            ("tokens", "total_tokens"),
            ("seconds", "duration_seconds"),
            ("inspect_log", "native_log"),
        ):
            if source in attempt and target not in attempt:
                attempt[target] = attempt[source]
            elif source in attempt and target in attempt and attempt[source] != attempt[target]:
                raise ValueError("Conflicting public and raw attempt field aliases")
        attempt_id, case_id = attempt.get("id"), attempt.get("case_id")
        if not isinstance(attempt_id, str) or not attempt_id or attempt_id in ids:
            raise ValueError("Attempt IDs must be unique nonempty strings")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every attempt requires a case ID")
        ids.add(attempt_id)
        stage, split, status = attempt.get("stage"), attempt.get("split"), attempt.get("status")
        if stage not in STAGES or split not in SPLITS or status not in STATUSES:
            raise ValueError("Unknown learning stage, split, or attempt status")
        key = (stage, split, case_id)
        if split != "train":
            if key in evaluation_keys:
                raise ValueError(
                    "Duplicate evaluation case in one stage/split; pairing is ambiguous"
                )
            evaluation_keys.add(key)
        reward = attempt.get("reward")
        if status in SCORED_STATUSES:
            expected = int(status == "success")
            if reward is not None and (isinstance(reward, bool) or reward != expected):
                raise ValueError("Scored attempt reward disagrees with its status")
        elif reward is not None:
            raise ValueError("Unscored attempt cannot have a numeric reward")
        tokens, seconds = attempt.get("total_tokens"), attempt.get("duration_seconds")
        if tokens is not None and (
            isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0
        ):
            raise ValueError("Recorded token counts must be nonnegative integers")
        if seconds is not None and (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError("Recorded durations must be finite nonnegative numbers")
        result.append(attempt)
    if report.get("dataset", {}).get("test_status") != "opened" and any(
        attempt["split"] == "sealed_test" for attempt in result
    ):
        raise ValueError("Cannot analyze or export sealed-test outcomes before test release")
    return result


def _cost_metric(attempts: Sequence[dict], key: str) -> dict:
    values = [attempt[key] for attempt in attempts if attempt.get(key) is not None]
    return {
        "recorded_total": sum(values) if values else None,
        "recorded_attempts": len(values),
        "missing_attempts": len(attempts) - len(values),
        "mean_per_recorded_attempt": sum(values) / len(values) if values else None,
    }


def _costs(attempts: Sequence[dict]) -> dict:
    return {
        "attempt_count": len(attempts),
        "tokens": _cost_metric(attempts, "total_tokens"),
        "seconds": _cost_metric(attempts, "duration_seconds"),
    }


def _planned_cases(report: Mapping[str, Any], split: str) -> int | None:
    key = {"train": "train_count", "validation": "validation_count", "sealed_test": "test_count"}[
        split
    ]
    count = report.get("dataset", {}).get(key)
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
        raise ValueError("Planned case counts must be nonnegative integers")
    return count


def summarize_attempts(attempts: Sequence[dict], *, planned_cases: int | None = None) -> dict:
    """Summarize validated actual records; absent attempts are not pending rows."""
    counts = Counter(attempt["status"] for attempt in attempts)
    scored = [attempt for attempt in attempts if attempt["status"] in SCORED_STATUSES]
    case_count = len({attempt["case_id"] for attempt in scored})
    observed_case_count = len({attempt["case_id"] for attempt in attempts})
    if planned_cases is not None and observed_case_count > planned_cases:
        raise ValueError("Recorded cases exceed the frozen dataset size")
    return {
        "recorded_attempts": len(attempts),
        "counts": {status: counts[status] for status in STATUSES},
        "scored_attempts": len(scored),
        "success_rate": counts["success"] / len(scored) if scored else None,
        "scored_cases": case_count,
        "recorded_cases": observed_case_count,
        "planned_cases": planned_cases,
        "scored_case_coverage": case_count / planned_cases if planned_cases else None,
        "costs": {"all_attempts": _costs(attempts), "scored_attempts": _costs(scored)},
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def paired_comparison(
    attempts: Sequence[dict],
    control: str,
    treatment: str,
    *,
    planned_cases: int | None,
    seed: int,
    bootstrap_samples: int,
) -> dict:
    """Compare one binary result per case only when both conditions are scored."""
    selected = {stage: {} for stage in (control, treatment)}
    for attempt in attempts:
        if attempt["split"] != "sealed_test" or attempt["stage"] not in selected:
            continue
        stage = selected[attempt["stage"]]
        if attempt["case_id"] in stage:
            raise ValueError("Duplicate evaluation case makes paired comparison ambiguous")
        stage[attempt["case_id"]] = attempt
    paired_ids = sorted(
        case_id
        for case_id in set(selected[control]) & set(selected[treatment])
        if selected[control][case_id]["status"] in SCORED_STATUSES
        and selected[treatment][case_id]["status"] in SCORED_STATUSES
    )
    if planned_cases is not None and len(paired_ids) > planned_cases:
        raise ValueError("Paired cases exceed the frozen test size")
    differences = [
        int(selected[treatment][case_id]["status"] == "success")
        - int(selected[control][case_id]["status"] == "success")
        for case_id in paired_ids
    ]
    control_successes = sum(
        selected[control][case_id]["status"] == "success" for case_id in paired_ids
    )
    treatment_successes = sum(
        selected[treatment][case_id]["status"] == "success" for case_id in paired_ids
    )
    interval = None
    if differences:
        rng = random.Random(f"{seed}:{control}:{treatment}")
        values = sorted(
            100 * sum(rng.choices(differences, k=len(differences))) / len(differences)
            for _ in range(bootstrap_samples)
        )
        interval = [_quantile(values, 0.025), _quantile(values, 0.975)]
    return {
        "id": f"{treatment}_vs_{control}",
        "control": control,
        "treatment": treatment,
        "split": "sealed_test",
        "planned_cases": planned_cases,
        "paired_scored_cases": len(paired_ids),
        "paired_coverage": len(paired_ids) / planned_cases if planned_cases else None,
        "paired_case_ids": paired_ids,
        "control_scored_cases": sum(
            attempt["status"] in SCORED_STATUSES for attempt in selected[control].values()
        ),
        "treatment_scored_cases": sum(
            attempt["status"] in SCORED_STATUSES for attempt in selected[treatment].values()
        ),
        "treatment_wins": differences.count(1),
        "treatment_losses": differences.count(-1),
        "ties": differences.count(0),
        "both_success": sum(
            selected[control][case_id]["status"]
            == selected[treatment][case_id]["status"]
            == "success"
            for case_id in paired_ids
        ),
        "both_failure": sum(
            selected[control][case_id]["status"]
            == selected[treatment][case_id]["status"]
            == "failure"
            for case_id in paired_ids
        ),
        "control_success_rate_paired": control_successes / len(paired_ids) if paired_ids else None,
        "treatment_success_rate_paired": treatment_successes / len(paired_ids)
        if paired_ids
        else None,
        "delta_percentage_points": 100 * sum(differences) / len(differences)
        if differences
        else None,
        "bootstrap_95_interval_percentage_points": interval,
        "bootstrap": {
            "method": "paired percentile",
            "samples": bootstrap_samples,
            "seed": seed,
            "assumptions": BOOTSTRAP_ASSUMPTIONS,
        },
    }


def summarize_learning(
    report: Mapping[str, Any],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_samples: int = 10000,
) -> dict:
    """Analyze the public learning report without opening any fixture or witness."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Bootstrap seed must be an integer")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 1
    ):
        raise ValueError("Bootstrap sample count must be a positive integer")
    attempts = _attempts(report)
    opened = report.get("dataset", {}).get("test_status") == "opened"
    groups = []
    for stage in STAGES:
        for split in SPLITS:
            selected = [
                attempt
                for attempt in attempts
                if attempt["stage"] == stage and attempt["split"] == split
            ]
            if split == "train" and not selected:
                continue
            if split == "sealed_test" and not opened:
                continue
            groups.append(
                {
                    "stage": stage,
                    "split": split,
                    **summarize_attempts(
                        selected,
                        planned_cases=_planned_cases(report, split),
                    ),
                }
            )
    return {
        "schema_version": 1,
        "experiment_id": report.get("id"),
        "source_updated_at": report.get("updated_at"),
        "source_status": report.get("status"),
        "test_status": report.get("dataset", {}).get("test_status", "unknown"),
        "attempt_records_available": "attempts" in report,
        "groups": groups,
        "costs": {
            "all_attempts": _costs(attempts),
            "scored_attempts": _costs(
                [attempt for attempt in attempts if attempt["status"] in SCORED_STATUSES]
            ),
        },
        "comparisons": [
            paired_comparison(
                attempts,
                control,
                treatment,
                planned_cases=_planned_cases(report, "sealed_test"),
                seed=seed,
                bootstrap_samples=bootstrap_samples,
            )
            for control, treatment in COMPARISONS
        ]
        if opened
        else [],
        "notes": [
            "Success rates use success / (success + failure); unscored attempts are excluded, not assigned zero.",
            "Paired deltas use exactly the same cases scored by both conditions; inspect paired coverage alongside each delta.",
            "Recorded cost totals include only available measurements. All-attempt costs include unscored and active attempts when measured; scored costs exclude them. Optimization compute is not an attempt cost.",
            "Training groups may include repeated stochastic rollouts of one case; training success rates are per rollout and case coverage counts unique cases.",
            BOOTSTRAP_ASSUMPTIONS,
        ],
    }


def _percentage(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _cost_text(cost: dict) -> str:
    total = cost["recorded_total"]
    if total is None:
        return "Not recorded"
    return (
        f"{total:,.1f} ({cost['recorded_attempts']} recorded; {cost['missing_attempts']} missing)"
    )


def analysis_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# DealRoom learning results",
        "",
        f"Experiment: `{summary.get('experiment_id') or 'unspecified'}`. Test state: `{summary['test_status']}`.",
        "",
        "Rates use scored attempts only. Every count below comes from a recorded attempt.",
        "",
        "| Condition | Split | Success / scored | Success rate | Scored cases / planned | Limits / errors / incomplete |",
        "|---|---|---:|---:|---:|---:|",
    ]
    if not summary["attempt_records_available"]:
        lines[4:4] = [
            "Attempt-level records are unavailable. Rates and paired comparisons cannot be recovered from stage aggregates alone.",
            "",
        ]
    for group in summary["groups"]:
        counts = group["counts"]
        planned = "unknown" if group["planned_cases"] is None else group["planned_cases"]
        lines.append(
            f"| {group['stage']} | {group['split']} | {counts['success']} / {group['scored_attempts']} "
            f"| {_percentage(group['success_rate'])} | {group['scored_cases']} / {planned} "
            f"| {counts['limit']} / {counts['error']} / {counts['incomplete']} |"
        )
    lines += ["", "## Paired held-out changes", ""]
    if summary["test_status"] != "opened":
        lines.append("The test set remains sealed. No held-out comparison is available.")
    else:
        lines += [
            "Positive changes favor the later condition. Both conditions must have scored the case.",
            "",
            "| Comparison | Paired cases / planned | Wins / losses / ties | Change (percentage points) | Bootstrap 95% interval |",
            "|---|---:|---:|---:|---:|",
        ]
        for pair in summary["comparisons"]:
            delta = pair["delta_percentage_points"]
            interval = pair["bootstrap_95_interval_percentage_points"]
            delta_text = "—" if delta is None else f"{delta:+.1f}"
            interval_text = "—" if interval is None else f"[{interval[0]:+.1f}, {interval[1]:+.1f}]"
            lines.append(
                f"| {pair['treatment']} − {pair['control']} | {pair['paired_scored_cases']} / "
                f"{pair['planned_cases'] if pair['planned_cases'] is not None else 'unknown'} | "
                f"{pair['treatment_wins']} / {pair['treatment_losses']} / {pair['ties']} | "
                f"{delta_text} | {interval_text} |"
            )
        lines += ["", BOOTSTRAP_ASSUMPTIONS]
    lines += [
        "",
        "## Recorded inference costs",
        "",
        "Costs include native agent attempts, excluding model optimization. Missing measurements remain missing.",
        "",
        "| Condition | Split | All-attempt tokens | Scored-attempt tokens | All-attempt seconds | Scored-attempt seconds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group in summary["groups"]:
        all_cost = group["costs"]["all_attempts"]
        scored_cost = group["costs"]["scored_attempts"]
        lines.append(
            f"| {group['stage']} | {group['split']} | {_cost_text(all_cost['tokens'])} "
            f"| {_cost_text(scored_cost['tokens'])} | {_cost_text(all_cost['seconds'])} "
            f"| {_cost_text(scored_cost['seconds'])} |"
        )
    lines += ["", *summary["notes"], ""]
    return "\n".join(lines)


def export_learning(
    report_or_path: Mapping[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_samples: int = 10000,
) -> dict[str, str]:
    """Export a summary, raw observed-attempt CSV, and readable analysis."""
    report = (
        json.loads(Path(report_or_path).read_text())
        if isinstance(report_or_path, (str, Path))
        else dict(report_or_path)
    )
    summary = summarize_learning(report, seed=seed, bootstrap_samples=bootstrap_samples)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_path": output / "learning_analysis.json",
        "csv_path": output / "learning_attempts.csv",
        "markdown_path": output / "learning_analysis.md",
    }
    paths["summary_path"].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    paths["markdown_path"].write_text(analysis_markdown(summary))
    fields = [
        "id",
        "stage",
        "split",
        "case_id",
        "status",
        "reward",
        "total_tokens",
        "duration_seconds",
        "started_at",
        "completed_at",
        "native_log",
        "reason",
    ]
    with paths["csv_path"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for attempt in _attempts(report):
            writer.writerow({key: attempt.get(key) for key in fields})
    return {key: str(path) for key, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    print(
        json.dumps(
            export_learning(
                args.report, args.output, seed=args.seed, bootstrap_samples=args.bootstrap_samples
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
