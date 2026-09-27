"""Domain behavior checks. Witness scores validate fixtures, not agent performance."""

from copy import deepcopy
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from dealroom.domain import (
    Case,
    ExtensionTerms,
    FinalClaims,
    evaluate,
    list_cases,
    load_case,
    observe,
    reset,
    step,
)
from dealroom.witnesses import (
    FLAGSHIP_FLAWED_SUFFIX,
    FLAGSHIP_PREFIX,
    FLAGSHIP_REPAIRED_SUFFIX,
    WITNESSES,
    draft,
    finish,
    proceed,
    route,
    wait,
)


def run(case_id, actions):
    state = reset(case_id)
    results = [step(state, action) for action in actions]
    return state, results


@pytest.mark.parametrize("case_id", sorted(WITNESSES))
def test_each_fixture_has_successful_witness(case_id):
    state, results = run(case_id, WITNESSES[case_id])
    assert all(result.accepted for result in results), [r.reason for r in results if not r.accepted]
    score = evaluate(state)
    assert score.success and score.reward == 1
    assert all(diagnostic.passed for diagnostic in score.diagnostics)
    assert state.finished


def test_six_validated_fixtures_with_two_development_cases():
    cases = [load_case(case_id) for case_id in list_cases()]
    assert len(cases) == 6
    assert sum(c.split == "development" for c in cases) == 2
    assert sum(c.split == "evaluation" for c in cases) == 4
    assert all(len(c.documents) >= 3 and len(c.messages) >= 2 for c in cases)
    assert all(c.start_at < c.deadline and c.start_at.utcoffset() is not None for c in cases)


def test_flagship_exact_prefix_replay_and_actual_failed_repaired_results():
    flawed, _ = run("case-01", FLAGSHIP_PREFIX)
    repaired, _ = run("case-01", FLAGSHIP_PREFIX)
    assert flawed.model_dump(mode="json") == repaired.model_dump(mode="json")
    boundary = deepcopy(observe(flawed))
    assert boundary["effective_credit_cents"] == 0
    assert boundary["amendments"][0]["status"] == "draft"
    assert any("Informal agreement" in m["body"] for m in boundary["messages"])
    for action in FLAGSHIP_FLAWED_SUFFIX:
        step(flawed, action)
    for action in FLAGSHIP_REPAIRED_SUFFIX:
        step(repaired, action)
    assert not evaluate(flawed).success
    missing = next(d for d in evaluate(flawed).diagnostics if d.name == "revision_signatures")
    assert not missing.passed and "buyer" in missing.reason and "seller" in missing.reason
    assert evaluate(repaired).success
    assert repaired.disposition.residual_cents == 400_000
    assert repaired.history[: len(FLAGSHIP_PREFIX)] == flawed.history[: len(FLAGSHIP_PREFIX)]


def test_all_witnesses_replay_to_identical_records():
    for case_id, actions in WITNESSES.items():
        first, _ = run(case_id, actions)
        second, _ = run(case_id, deepcopy(actions))
        assert first.model_dump(mode="json") == second.model_dump(mode="json")
        assert evaluate(first) == evaluate(second)


def test_terminal_allows_reads_and_final_claims_but_no_business_mutation():
    state, _ = run("case-02", WITNESSES["case-02"][:-1])
    terminal_time = state.now
    sigs = state.signatures.copy()
    assert step(
        state, {"type": "read", "resource": "envelope", "resource_id": "envelope-001"}
    ).accepted
    rejected = step(state, draft(325_000))
    assert not rejected.accepted
    assert state.now == terminal_time and state.signatures == sigs
    assert step(state, finish(350_000, [1])).accepted
    assert evaluate(state).success
    assert step(state, {"type": "read", "resource": "summary"}).accepted
    assert not step(state, finish(350_000, [1])).accepted


def test_finish_cannot_manufacture_a_credit_or_disposition():
    state = reset("case-02")
    assert step(state, finish(350_000, [1])).accepted
    assert state.disposition is None and not state.signatures and not state.executed_at
    score = evaluate(state)
    assert not score.success
    assert not next(d for d in score.diagnostics if d.name == "supported_final_claims").passed
    assert not next(d for d in score.diagnostics if d.name == "revision_signatures").passed


def test_missing_signature_has_public_evidence_and_later_retry_succeeds():
    actions = WITNESSES["case-03"]
    state, results = run("case-03", actions[:2])
    envelope = observe(state, "envelope", "envelope-001")
    assert envelope["revision"]["status"] == "partially_signed"
    assert {s["party"] for s in envelope["signatures"]} == {"buyer"}
    assert state.executed_at == {}
    assert not step(state, proceed()).accepted
    for action in actions[2:]:
        assert step(state, action).accepted
    assert evaluate(state).success
    assert len(state.signatures) == 2


def test_exact_deadline_extension_events_precede_expiry():
    case = load_case("case-05")
    data = case.model_dump(mode="json")
    # Route completes 09:10. Both signatures arrive at the original 10:00 deadline.
    data["counterpart"]["extension_signature_minutes"] = {"buyer": 50, "seller": 50}
    state = reset(Case.model_validate(data))
    assert step(state, WITNESSES["case-05"][1]).accepted
    assert step(state, route()).accepted
    assert step(state, wait("2026-10-07T10:00:00-04:00")).accepted
    assert state.deadline.isoformat() == "2026-10-07T14:00:00-04:00"
    assert state.business_status == "open"
    at_deadline = [event for event in state.events if event["at"] == "2026-10-07T10:00:00-04:00"]
    assert [e["kind"] for e in at_deadline] == [
        "signature_received",
        "signature_received",
        "amendment_executed",
    ]
    assert [e.get("party") for e in at_deadline[:2]] == ["buyer", "seller"]


def test_proceed_can_complete_with_final_signature_at_exact_deadline():
    data = load_case("case-02").model_dump(mode="json")
    data["deadline"] = "2026-10-06T09:30:00-04:00"
    state = reset(Case.model_validate(data))
    assert step(state, draft(350_000)).accepted
    assert step(state, route()).accepted
    assert step(state, wait("2026-10-06T09:25:00-04:00")).accepted
    assert not state.executed_at  # Seller's event is due at 09:30.
    assert step(state, proceed()).accepted
    assert state.now == state.deadline
    assert state.executed_at["revision-001"] == state.deadline
    assert step(state, finish(350_000, [1])).accepted
    assert evaluate(state).success


def test_unsigned_extension_never_moves_deadline():
    state = reset("case-05")
    assert step(state, WITNESSES["case-05"][1]).accepted
    original = state.deadline
    assert step(state, wait("2026-10-07T10:01:00-04:00")).accepted
    assert state.deadline == original and state.business_status == "expired"
    assert not step(state, route()).accepted


def test_extension_applies_to_credit_envelope_already_in_flight():
    state = reset("case-05")
    # This alternate valid sequence starts the long signature request first.
    assert step(state, draft(850_000)).accepted
    assert step(state, route()).accepted
    assert step(state, WITNESSES["case-05"][1]).accepted
    assert step(state, route(2)).accepted
    assert step(state, wait("2026-10-07T12:15:00-04:00")).accepted
    assert state.executed_at.keys() == {"revision-001", "revision-002"}
    assert step(state, proceed()).accepted
    assert step(state, finish(850_000, [1, 2])).accepted
    assert evaluate(state).success


def test_invalid_attempts_have_no_business_effect_and_consume_budget():
    data = load_case("case-02").model_dump(mode="json")
    data["action_limit"] = 2
    state = reset(Case.model_validate(data))
    before = state.now
    assert not step(state, {"type": "send_for_signature", "revision_id": "unavailable"}).accepted
    assert not step(
        state, {"type": "draft_amendment", "terms": {"type": "credit", "amount_cents": -1}}
    ).accepted
    assert state.attempts == 2 and state.budget_exhausted
    assert state.now == before and state.revisions == {} and state.pending_events == []
    assert not step(state, draft(350_000)).accepted
    assert evaluate(state).outcome == "budget_exhausted"
    assert step(state, {"type": "read"}).accepted
    assert step(state, finish(0, [], "unresolved")).accepted


def test_timestamp_validation_rejects_naive_action_and_optional_claim():
    with pytest.raises(ValidationError):
        ExtensionTerms(deadline=datetime(2026, 10, 9, 17))
    with pytest.raises(ValidationError):
        FinalClaims(disposition="unresolved", deadline=datetime(2026, 10, 9, 17))
    state = reset("case-02")
    assert not step(state, {"type": "wait_until", "at": "2026-10-06T10:00:00"}).accepted


def test_immutable_terms_and_signature_records():
    state, _ = run("case-02", WITNESSES["case-02"])
    revision = state.revisions["revision-001"]
    with pytest.raises(ValidationError):
        revision.terms.amount_cents = 1
    with pytest.raises(ValidationError):
        state.signatures[0].revision_id = "revision-999"


def test_read_is_free_but_coordination_takes_five_minutes():
    state = reset("case-02")
    start = state.now
    assert step(state, {"type": "read"}).accepted
    assert state.now == start
    assert step(state, draft(350_000)).accepted
    assert state.now == start + timedelta(minutes=5)
    before = state.now
    assert not step(state, draft(1)).accepted
    assert state.now == before
