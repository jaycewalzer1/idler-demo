"""Synthetic suite checks validate fixtures, never assert model capability."""

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from dealroom.domain import effective_credit, load_case, observe, reset, step
from dealroom.score import evaluate
from dealroom.synthesis import (
    DEFAULT_CASE_DIR,
    DEFAULT_SEED,
    FAMILIES,
    generate_suite,
    suite_case_ids,
    validation_witness,
    verify_manifest,
)


@pytest.mark.parametrize("case_id", suite_case_ids())
def test_synthetic_case_has_an_executable_successful_witness(case_id):
    state = reset(case_id)
    for action in validation_witness(case_id):
        result = step(state, action)
        assert result.accepted, result.reason
    score = evaluate(state)
    assert score.reward == 1
    assert score.finished and score.invalid_attempts == 0
    assert all(d.passed for d in score.diagnostics)


def test_generation_is_byte_reproducible_and_seed_changes_task_parameters(tmp_path):
    first = generate_suite(tmp_path / "first")
    second = generate_suite(tmp_path / "second")
    assert first == second
    for filename in ["manifest.json", *(f"{cid}.json" for cid in suite_case_ids())]:
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()
        assert (tmp_path / "first" / filename).read_bytes() == (
            DEFAULT_CASE_DIR / filename
        ).read_bytes()
    alternate = generate_suite(tmp_path / "alternate", seed=DEFAULT_SEED + 1)
    assert alternate["fingerprint"] != first["fingerprint"]
    original = load_case(tmp_path / "first" / "synth-001.json")
    changed = load_case(tmp_path / "alternate" / "synth-001.json")
    assert (
        original.counterpart.specialist_document.eligible_cost_cents
        != changed.counterpart.specialist_document.eligible_cost_cents
    )


def _keys(value):
    if isinstance(value, dict):
        yield from value
        for item in value.values():
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def test_private_generation_and_verification_data_never_enter_public_observations():
    manifest = json.loads((DEFAULT_CASE_DIR / "manifest.json").read_text())
    prohibited = {
        "counterpart",
        "acceptable_dispositions",
        "prelude",
        "pending_events",
        "buyer_grants",
        "seller_max_credit_cents",
        "validation",
        "witness",
        "witness_fingerprint",
        "fingerprint",
        "family",
        "seed",
    }
    for case_id in suite_case_ids():
        state = reset(case_id)
        for action in [None, *validation_witness(case_id)]:
            if action is not None:
                step(state, action)
            assert prohibited.isdisjoint(_keys(observe(state)))
    # Validation results are clearly separated and scripts are not shipped in the manifest.
    assert manifest["provenance"].endswith("These validation scores are not model results.")
    assert all(t["validation"]["reward"] == 1 for t in manifest["tasks"])
    assert "action" not in set(_keys(manifest))


def test_suite_has_eight_behavioral_families_and_varied_constraints():
    manifest = json.loads((DEFAULT_CASE_DIR / "manifest.json").read_text())
    cases = [load_case(case_id) for case_id in suite_case_ids()]
    assert len(cases) == 48
    assert Counter(t["family"] for t in manifest["tasks"]) == dict.fromkeys(FAMILIES, 6)
    assert Counter(c.split for c in cases) == {"development": 16, "evaluation": 32}
    assert len({t["fingerprint"] for t in manifest["tasks"]}) == 48
    assert len({c.counterpart.seller_max_credit_cents for c in cases}) == 48
    assert len({c.authorities[0].maximum_residual_cents for c in cases}) == 6
    assert {len(c.counterpart.required_evidence_ids) for c in cases} == {2, 3}
    assert len({tuple(c.counterpart.signature_minutes.values()) for c in cases}) >= 12

    discovery = reset("synth-001")
    straightforward = reset("synth-007")
    partial = reset("synth-013")
    revised = reset("synth-019")
    extension = reset("synth-025")
    cancellation = reset("synth-031")
    authority = reset("synth-037")
    boundary = reset("synth-043")
    assert observe(discovery)["eligible_cost_cents"] is None
    assert observe(straightforward)["eligible_cost_cents"] is not None
    assert not straightforward.revisions
    assert len(partial.signatures) == 1 and not partial.executed_at
    assert set(revised.executed_at) == {"revision-001"}
    assert set(revised.revisions) == {"revision-001", "revision-002"}
    assert revised.revisions["revision-002"].terms.amount_cents > effective_credit(revised)
    assert extension.case.counterpart.maximum_extension > extension.deadline
    assert cancellation.case.acceptable_dispositions == ("cancel",)
    assert "cancel" not in observe(cancellation)["authority"]["permissions"]
    assert "credit" not in observe(authority)["authority"]["permissions"]
    assert (boundary.deadline - boundary.now).total_seconds() == 25 * 60


@pytest.mark.parametrize("case_id", ["synth-001", "synth-013", "synth-019", "synth-031"])
def test_opening_records_cannot_be_treated_as_completed_deals(case_id):
    state = reset(case_id)
    assert not step(
        state,
        {"type": "set_disposition", "disposition": "proceed", "reason": "Premature completion"},
    ).accepted
    assert state.disposition is None


def test_new_families_require_authority_and_completion_aware_timing():
    authority = reset("synth-037")
    draft = next(a for a in validation_witness("synth-037") if a["type"] == "draft_amendment")
    assert not step(authority, draft).accepted

    cancellation = reset("synth-031")
    assert not step(
        cancellation,
        {"type": "set_disposition", "disposition": "cancel", "reason": "Without a grant"},
    ).accepted

    boundary = reset("synth-043")
    witness = validation_witness("synth-043")
    for action in witness[:3]:
        assert step(boundary, action).accepted
    # Waiting until signatures arrive leaves no time to commit the disposition.
    assert step(boundary, {"type": "wait_until", "at": boundary.deadline.isoformat()}).accepted
    assert boundary.business_status == "expired"
    assert len(boundary.signatures) == 2
    assert not step(boundary, witness[-2]).accepted
    assert not evaluate(boundary).success


def test_original_six_fixtures_are_retained():
    cases_dir = Path(__file__).resolve().parent.parent / "cases"
    assert len(list(cases_dir.glob("case-*.json"))) == 6
    assert load_case("case-01").id == "case-01"


@pytest.mark.parametrize("field", ["seed", "fingerprint", "task_id", "task_title", "validation"])
def test_manifest_verification_rejects_changed_provenance(field):
    manifest = json.loads((DEFAULT_CASE_DIR / "manifest.json").read_text())
    verify_manifest(manifest)
    changed = deepcopy(manifest)
    if field == "seed":
        changed["seed"] += 1
    elif field == "fingerprint":
        changed["fingerprint"] = "not-the-suite"
    elif field == "task_id":
        changed["tasks"][0]["id"] = changed["tasks"][1]["id"]
    elif field == "task_title":
        changed["tasks"][0]["title"] = "A different task"
    else:
        changed["tasks"][0]["validation"]["reward"] = 0
    with pytest.raises(ValueError):
        verify_manifest(changed)


def test_manifest_verification_rejects_modified_fixture_despite_unchanged_manifest(tmp_path):
    manifest = generate_suite(tmp_path)
    verify_manifest(manifest, tmp_path)
    path = tmp_path / "synth-001.json"
    case = json.loads(path.read_text())
    case["counterpart"]["seller_max_credit_cents"] += 100
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="differs from its declared seed"):
        verify_manifest(manifest, tmp_path)
