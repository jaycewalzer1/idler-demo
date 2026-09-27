"""Training-data isolation and executable held-out dependency checks."""

import json
from collections import Counter

import pytest

from dealroom.curriculum import (
    DEFAULT_LEARNING_DIR,
    DEFAULT_SEED,
    SEALED_FAMILIES,
    SPLIT_COUNTS,
    TRAIN_FAMILIES,
    _case_and_witness,
    _fingerprint,
    _specs,
    generate_curriculum,
    learning_case_ids,
    load_learning_case,
    load_manifest,
    training_witness,
    verify_curriculum,
)
from dealroom.domain import observe, reset, step
from dealroom.score import evaluate


def test_all_fixtures_have_successful_private_engine_witnesses():
    manifest = load_manifest()
    assert Counter(task["split"] for task in manifest["tasks"]) == Counter(SPLIT_COUNTS)
    for spec in _specs():
        case, witness, metadata = _case_and_witness(*spec)
        assert metadata in manifest["tasks"]
        assert case == load_learning_case(case.id)
        state = reset(case)
        for action in witness:
            receipt = step(state, action)
            assert receipt.accepted, (case.id, receipt.reason)
        score = evaluate(state)
        assert score.reward == 1
        assert score.invalid_attempts == 0
        assert all(diagnostic.passed for diagnostic in score.diagnostics)


def test_reproducible_bytes_and_seed_changes(tmp_path):
    actual = generate_curriculum(tmp_path / "same")
    assert actual == load_manifest()
    for path in DEFAULT_LEARNING_DIR.rglob("*.json"):
        assert (
            path.read_bytes()
            == (tmp_path / "same" / path.relative_to(DEFAULT_LEARNING_DIR)).read_bytes()
        )
    alternate = generate_curriculum(tmp_path / "alternate", seed=DEFAULT_SEED + 1)
    assert alternate["fingerprint"] != actual["fingerprint"]
    assert {t["id"] for t in alternate["tasks"]}.isdisjoint(t["id"] for t in actual["tasks"])
    verify_curriculum(tmp_path / "same")


def test_structural_holdout_and_no_old_case_overlap():
    manifest = load_manifest()
    training = [task for task in manifest["tasks"] if task["split"] == "train"]
    validation = [task for task in manifest["tasks"] if task["split"] == "validation"]
    sealed = [task for task in manifest["tasks"] if task["split"] == "sealed_test"]
    assert set(task["family"] for task in training) == set(TRAIN_FAMILIES)
    assert set(task["family"] for task in validation) == set(TRAIN_FAMILIES)
    assert set(task["family"] for task in sealed) == set(SEALED_FAMILIES)
    assert {_fingerprint(task["dependency_edges"]) for task in sealed}.isdisjoint(
        _fingerprint(task["dependency_edges"]) for task in training + validation
    )
    old = json.loads((DEFAULT_LEARNING_DIR.parent / "generated" / "manifest.json").read_text())
    for key in ("id", "fingerprint", "public_initial_fingerprint"):
        assert {task[key] for task in manifest["tasks"]}.isdisjoint(
            task[key] for task in old["tasks"]
        )
    assert Counter(task["stage"] for task in training) == {
        "commit_ready": 8,
        "route_ready": 8,
        "coordination": 240,
    }
    assert all(
        case_id.startswith("learn-") and len(case_id) == 22 for case_id in learning_case_ids()
    )


def test_only_training_witnesses_are_exportable():
    for split in ("validation", "sealed_test"):
        for case_id in learning_case_ids(split):
            with pytest.raises(PermissionError, match="fixture validation only"):
                training_witness(case_id)
    for case_id in learning_case_ids():
        witness = training_witness(case_id)
        assert witness[-1]["type"] == "finish"
    with pytest.raises(ValueError, match="Unknown learning split"):
        learning_case_ids("bogus")
    with pytest.raises(ValueError, match="Unknown learning case"):
        load_learning_case("../../generated/synth-001")


def _keys(value):
    if isinstance(value, dict):
        yield from value
        for child in value.values():
            yield from _keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _keys(child)


def test_public_observations_do_not_contain_private_curriculum_fields():
    forbidden = {
        "counterpart",
        "prelude",
        "acceptable_dispositions",
        "pending_events",
        "buyer_grants",
        "seller_max_credit_cents",
        "witness",
        "witness_fingerprint",
        "validation",
        "family",
        "stage",
        "dependency_edges",
        "fingerprint",
        "seed",
        "split",
    }
    # Every task and every witness transition, including newly delivered evidence.
    for spec in _specs():
        case, witness, _ = _case_and_witness(*spec)
        raw_fixture = json.loads((DEFAULT_LEARNING_DIR / f"{case.id}.json").read_text())
        assert {"witness", "target", "family", "dependency_edges"}.isdisjoint(raw_fixture)
        state = reset(case)
        assert forbidden.isdisjoint(_keys(observe(state)))
        for action in witness:
            receipt = step(state, action)
            assert forbidden.isdisjoint(_keys(receipt.observation))


@pytest.mark.parametrize("family", SEALED_FAMILIES)
def test_heldout_compositions_really_require_the_added_dependency(family):
    case, witness, _ = _case_and_witness("sealed_test", SEALED_FAMILIES.index(family), family, 2)
    state = reset(case)
    if family in {"extension_authority_credit", "revised_extension_credit"}:
        # Remove the extension draft, route, and response wait, then replay the
        # credit sequence at original absolute times. The old deadline wins.
        remove = next(
            i
            for i, a in enumerate(witness)
            if a["type"] == "draft_amendment" and a["terms"]["type"] == "extension"
        )
        selected = witness[:remove] + witness[remove + 3 :]
        extension_number = 2 if family == "revised_extension_credit" else 1
        selected = json.loads(json.dumps(selected))
        for action in selected:
            if action["type"] == "send_for_signature":
                number = int(action["revision_id"].split("-")[1])
                if number > extension_number:
                    action["revision_id"] = f"revision-{number - 1:03}"
    elif family == "quote_retry_credit":
        # Keep quote retrieval and the first route; remove the reroute. Merely
        # waiting until availability never produces the missing signature.
        routes = [i for i, a in enumerate(witness) if a["type"] == "send_for_signature"]
        selected = [action for i, action in enumerate(witness) if i != routes[-1]]
    else:
        selected = [
            action
            for action in witness
            if not (
                action["type"] == "request" and action["request"]["intent"] == "buyer_authorization"
            )
        ]
    for action in selected:
        step(state, action)
    assert not evaluate(state).success


def test_changed_fixture_and_witness_are_rejected(tmp_path):
    generate_curriculum(tmp_path)
    case_id = learning_case_ids(directory=tmp_path)[0]
    path = tmp_path / f"{case_id}.json"
    data = json.loads(path.read_text())
    data["title"] += " changed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_learning_case(case_id, tmp_path)
    generate_curriculum(tmp_path)
    path = tmp_path / "private" / f"{case_id}.json"
    data = json.loads(path.read_text())
    data["witness"][-1]["claims"]["effective_credit_cents"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="witness fingerprint mismatch"):
        training_witness(case_id, tmp_path)
    with pytest.raises(ValueError, match="Private validation witness changed"):
        verify_curriculum(tmp_path)
