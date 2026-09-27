"""Coverage-aware pairing, missing-cost handling, and sealed-test protection."""

import csv
import json

import pytest

from dealroom.learning_analysis import analysis_markdown, export_learning, summarize_learning


def attempt(stage, case, status, *, split="sealed_test", tokens=None, seconds=None, suffix=""):
    return {
        "id": f"{stage}-{split}-{case}{suffix}",
        "stage": stage,
        "split": split,
        "case_id": case,
        "status": status,
        "reward": 1 if status == "success" else 0 if status == "failure" else None,
        "tokens": tokens,
        "seconds": seconds,
        "inspect_log": f"learning/example/{stage}-{case}.eval",
    }


def report(attempts=(), *, opened=True, test_count=8):
    return {
        "id": "experiment",
        "status": "evaluating",
        "updated_at": "2026-09-27T22:00:00Z",
        "dataset": {
            "version": "learning-v1",
            "train_count": 256,
            "validation_count": 32,
            "test_count": test_count,
            "test_status": "opened" if opened else "sealed",
        },
        "attempts": list(attempts),
    }


def test_pairing_uses_only_cases_scored_by_both_not_unmatched_rates():
    attempts = [
        attempt("original", "a", "success", tokens=10, seconds=1),
        attempt("workflow", "a", "success", tokens=20, seconds=2),
        attempt("original", "b", "failure", tokens=30, seconds=3),
        attempt("workflow", "b", "success", tokens=40, seconds=4),
        attempt("original", "c", "success", tokens=50, seconds=5),
        attempt("workflow", "c", "failure", tokens=60, seconds=6),
        attempt("original", "d", "failure"),
        attempt("workflow", "d", "failure"),
        attempt("original", "e", "success"),
        attempt("workflow", "e", "limit", tokens=1000, seconds=100),
        attempt("workflow", "f", "success"),
        attempt("original", "f", "error", tokens=2000, seconds=200),
        attempt("workflow", "g", "pending"),
        attempt("original", "h", "running"),
    ]
    summary = summarize_learning(report(attempts), bootstrap_samples=1000)
    pair = summary["comparisons"][0]
    assert pair["id"] == "workflow_vs_original"
    assert pair["paired_case_ids"] == ["a", "b", "c", "d"]
    assert pair["paired_scored_cases"] == 4
    assert pair["paired_coverage"] == 0.5
    assert pair["treatment_wins"] == pair["treatment_losses"] == 1
    assert pair["ties"] == 2
    assert pair["both_success"] == pair["both_failure"] == 1
    assert pair["delta_percentage_points"] == 0
    assert pair["control_success_rate_paired"] == pair["treatment_success_rate_paired"] == 0.5
    workflow = next(
        group
        for group in summary["groups"]
        if group["stage"] == "workflow" and group["split"] == "sealed_test"
    )
    assert workflow["scored_attempts"] == 5
    assert workflow["success_rate"] == 3 / 5
    assert workflow["counts"]["pending"] == 1
    assert workflow["counts"]["limit"] == 1
    assert workflow["costs"]["all_attempts"]["tokens"] == {
        "recorded_total": 1120,
        "recorded_attempts": 4,
        "missing_attempts": 3,
        "mean_per_recorded_attempt": 280,
    }
    assert workflow["costs"]["scored_attempts"]["tokens"]["recorded_total"] == 120
    assert workflow["costs"]["all_attempts"]["seconds"]["recorded_total"] == 112
    assert workflow["costs"]["scored_attempts"]["seconds"]["recorded_total"] == 12


def test_bootstrap_is_seeded_case_paired_and_does_not_depend_on_attempt_order():
    attempts = []
    for index in range(12):
        attempts += [
            attempt("workflow", str(index), "success" if index % 3 else "failure"),
            attempt("sft", str(index), "success" if index % 2 else "failure"),
        ]
    first = summarize_learning(report(attempts, test_count=12), seed=47, bootstrap_samples=1234)
    second = summarize_learning(
        report(list(reversed(attempts)), test_count=12), seed=47, bootstrap_samples=1234
    )
    assert first["comparisons"] == second["comparisons"]
    pair = first["comparisons"][1]
    assert pair["id"] == "sft_vs_workflow"
    assert pair["paired_scored_cases"] == 12
    assert pair["delta_percentage_points"] == pytest.approx(-100 / 6)
    lower, upper = pair["bootstrap_95_interval_percentage_points"]
    assert -100 <= lower <= pair["delta_percentage_points"] <= upper <= 100
    assert pair["bootstrap"]["seed"] == 47
    assert "not a general model ranking" in pair["bootstrap"]["assumptions"]


def test_zero_scored_or_missing_attempt_records_never_become_zero_success_rates():
    summary = summarize_learning(
        report(
            [
                attempt("sft", "a", "limit"),
                attempt("rl", "a", "error"),
                attempt("rl", "b", "incomplete"),
            ]
        ),
        bootstrap_samples=10,
    )
    assert all(group["success_rate"] is None for group in summary["groups"])
    for pair in summary["comparisons"]:
        assert pair["paired_scored_cases"] == 0
        assert pair["delta_percentage_points"] is None
        assert pair["bootstrap_95_interval_percentage_points"] is None
        assert pair["control_success_rate_paired"] is None
    assert summary["costs"]["all_attempts"]["tokens"]["recorded_total"] is None
    assert summary["costs"]["all_attempts"]["tokens"]["missing_attempts"] == 3
    empty = report()
    empty.pop("attempts")
    summary = summarize_learning(empty)
    assert summary["attempt_records_available"] is False
    assert all(group["recorded_attempts"] == 0 for group in summary["groups"])
    assert all(group["counts"]["pending"] == 0 for group in summary["groups"])


def test_tied_all_failure_pairs_are_real_zero_delta_not_missing_data():
    attempts = [attempt(stage, "a", "failure") for stage in ("sft", "rl")]
    pair = summarize_learning(report(attempts), bootstrap_samples=20)["comparisons"][2]
    assert pair["id"] == "rl_vs_sft"
    assert pair["delta_percentage_points"] == 0
    assert pair["bootstrap_95_interval_percentage_points"] == [0, 0]
    assert pair["both_failure"] == 1
    assert pair["treatment_wins"] == pair["treatment_losses"] == 0


def test_sealed_guard_prevents_comparison_and_export_of_unreleased_outcomes(tmp_path):
    sealed = report([attempt("sft", "a", "success", split="validation")], opened=False)
    summary = summarize_learning(sealed)
    assert summary["comparisons"] == []
    assert all(group["split"] != "sealed_test" for group in summary["groups"])
    assert "remains sealed" in analysis_markdown(summary)
    leaked = report([attempt("sft", "a", "success")], opened=False)
    with pytest.raises(ValueError, match="before test release"):
        export_learning(leaked, tmp_path)
    assert not list(tmp_path.iterdir())


def test_duplicate_attempt_and_evaluation_case_ids_are_rejected_but_training_repeats_allowed():
    item = attempt("workflow", "a", "success")
    with pytest.raises(ValueError, match="Attempt IDs must be unique"):
        summarize_learning(report([item, item]))
    duplicate = {**item, "id": "a-different-attempt"}
    with pytest.raises(ValueError, match="Duplicate evaluation case"):
        summarize_learning(report([item, duplicate]))
    train = [
        attempt("rl", "a", "success", split="train", suffix="-1"),
        attempt("rl", "a", "failure", split="train", suffix="-2"),
    ]
    group = next(
        group
        for group in summarize_learning(report(train, opened=False))["groups"]
        if group["split"] == "train"
    )
    assert group["scored_attempts"] == 2 and group["scored_cases"] == 1
    assert group["success_rate"] == 0.5


def test_recorded_zero_cost_differs_from_missing_and_exports_only_real_records(tmp_path):
    source = report(
        [
            attempt("workflow", "a", "failure", tokens=0, seconds=0),
            attempt("workflow", "b", "limit"),
        ]
    )
    paths = export_learning(source, tmp_path, bootstrap_samples=10)
    summary = json.loads((tmp_path / "learning_analysis.json").read_text())
    cost = summary["costs"]["all_attempts"]["tokens"]
    assert cost["recorded_total"] == 0
    assert cost["recorded_attempts"] == cost["missing_attempts"] == 1
    with (tmp_path / "learning_attempts.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["reward"] == "0" and rows[1]["reward"] == ""
    assert rows[0]["total_tokens"] == "0" and rows[1]["total_tokens"] == ""
    assert rows[0]["native_log"] == source["attempts"][0]["inspect_log"]
    assert set(paths) == {"summary_path", "csv_path", "markdown_path"}
    markdown = (tmp_path / "learning_analysis.md").read_text()
    assert "1 recorded; 1 missing" in markdown
    assert "Optimization compute is not an attempt cost" in markdown


@pytest.mark.parametrize(
    "field,value", [("tokens", -1), ("tokens", 1.5), ("seconds", float("nan")), ("seconds", -1)]
)
def test_invalid_costs_are_not_silently_aggregated(field, value):
    item = attempt("sft", "a", "failure")
    item[field] = value
    with pytest.raises(ValueError, match="Recorded"):
        summarize_learning(report([item]))
