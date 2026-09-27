"""Behavioral regressions at action/event boundaries, using small controlled fixtures."""

import json
from datetime import datetime, timedelta

import pytest

from dealroom.domain import (
    Authority,
    Case,
    CounterpartPolicy,
    CreditTerms,
    effective_credit,
    load_case,
    observe,
    operative_revision,
    reset,
    step,
)
from dealroom.score import evaluate

START = datetime.fromisoformat("2026-10-05T09:00:00-04:00")


def fixture_case(**overrides):
    """A fully public quote and broad initial authority isolate the tested boundary."""
    deadline = START + timedelta(hours=4)
    authority = Authority(
        id="authority-001",
        effective_at=START,
        valid_through=deadline,
        permissions=("credit", "extension", "proceed", "cancel"),
        maximum_residual_cents=500_000,
        extension_limit=deadline + timedelta(hours=2),
        source_id="message-001",
    )
    values = {
        "id": "case-boundary",
        "title": "Maple Avenue",
        "split": "development",
        "brief": "Coordinate the disclosed inspection response.",
        "property": "18 Maple Avenue, Example City",
        "buyer": "Alex Buyer",
        "seller": "Jordan Seller",
        "start_at": START,
        "deadline": deadline,
        "public_policy": "Synthetic rules. Actions take five minutes; both parties must sign.",
        "documents": (
            {
                "id": "document-001",
                "title": "Inspection",
                "body": "Roof repair is recommended.",
                "kind": "inspection",
                "issued_at": START,
            },
            {
                "id": "document-002",
                "title": "Roof quote",
                "body": "Eligible work costs $12,000.",
                "kind": "quote",
                "eligible_cost_cents": 1_200_000,
                "issued_at": START,
            },
        ),
        "messages": (
            {"id": "message-001", "sender": "buyer", "body": "Residual limit $5,000.", "at": START},
        ),
        "authorities": (authority,),
        "counterpart": {
            "required_evidence_ids": ("document-002",),
            "seller_max_credit_cents": 1_000_000,
            "signature_minutes": {"buyer": 0, "seller": 0},
            "maximum_extension": deadline + timedelta(hours=2),
        },
        "acceptable_dispositions": ("proceed",),
    }
    values.update(overrides)
    return Case.model_validate(values)


def accept(state, action):
    result = step(state, action)
    assert result.accepted, result.reason
    return result


def credit_draft(amount=800_000):
    return {
        "type": "draft_amendment",
        "terms": {"type": "credit", "amount_cents": amount},
        "evidence_ids": ["document-002"],
    }


def finish(state, disposition="proceed", credit=800_000, revisions=("revision-001",)):
    accept(
        state,
        {
            "type": "finish",
            "claims": {
                "disposition": disposition,
                "effective_credit_cents": credit,
                "executed_revision_ids": list(revisions),
                "deadline": state.deadline.isoformat(),
            },
        },
    )


def test_authority_response_during_draft_revalidates_before_committing():
    case = fixture_case()
    replacement = case.authorities[0].model_copy(
        update={
            "id": "authority-002",
            "permissions": ("cancel",),
        }
    )
    policy = case.counterpart.model_copy(
        update={
            "buyer_grants": (replacement,),
            "buyer_reply_minutes": 3,
        }
    )
    state = reset(case.model_copy(update={"counterpart": policy}))
    accept(
        state,
        {
            "type": "request",
            "request": {
                "intent": "buyer_authorization",
                "permission": "cancel",
            },
        },
    )

    result = step(state, credit_draft())

    assert not result.accepted
    assert state.now == START + timedelta(minutes=10)
    assert state.revisions == {}
    assert len(state.authorities) == 2
    assert state.history[-1]["accepted"] is False


def test_failed_completion_at_deadline_still_expires_transaction():
    deadline = START + timedelta(minutes=10)
    case = fixture_case(deadline=deadline)
    authority = case.authorities[0].model_copy(
        update={
            "valid_through": deadline - timedelta(minutes=1),
        }
    )
    state = reset(case.model_copy(update={"authorities": (authority,)}))
    accept(state, credit_draft())

    result = step(state, {"type": "send_for_signature", "revision_id": "revision-001"})

    assert not result.accepted
    assert state.now == deadline
    assert state.business_status == "expired"
    assert not state.envelopes
    assert not state.signatures


def test_new_authority_does_not_retroactively_invalidate_accepted_execution():
    case = fixture_case()
    replacement = case.authorities[0].model_copy(
        update={
            "id": "authority-002",
            "permissions": ("proceed",),
        }
    )
    policy = case.counterpart.model_copy(
        update={
            "buyer_grants": (replacement,),
            "buyer_reply_minutes": 1,
        }
    )
    state = reset(case.model_copy(update={"counterpart": policy}))
    accept(state, credit_draft())
    accept(state, {"type": "send_for_signature", "revision_id": "revision-001"})
    accept(
        state,
        {
            "type": "request",
            "request": {
                "intent": "buyer_authorization",
                "permission": "proceed",
            },
        },
    )
    accept(state, {"type": "wait_until", "at": (START + timedelta(minutes=20)).isoformat()})
    accept(state, {"type": "set_disposition", "disposition": "proceed", "reason": "Signed credit."})
    finish(state)

    assert effective_credit(state) == 800_000
    assert evaluate(state).success


def test_new_unsigned_draft_preserves_old_operative_revision_and_supported_report():
    state = reset(fixture_case())
    accept(state, credit_draft())
    accept(state, {"type": "send_for_signature", "revision_id": "revision-001"})
    accept(state, credit_draft(900_000))
    accept(
        state,
        {"type": "set_disposition", "disposition": "proceed", "reason": "Use executed terms."},
    )
    finish(state)

    assert operative_revision(state).id == "revision-001"
    assert state.revisions["revision-002"].terms == CreditTerms(amount_cents=900_000)
    assert evaluate(state).success


def test_replacement_execution_is_not_added_to_or_maximized_with_previous_credit():
    state = reset(fixture_case())
    for amount, revision in [(800_000, "revision-001"), (750_000, "revision-002")]:
        accept(state, credit_draft(amount))
        accept(state, {"type": "send_for_signature", "revision_id": revision})
    accept(
        state,
        {
            "type": "set_disposition",
            "disposition": "proceed",
            "reason": "Latest executed revision.",
        },
    )
    finish(state, credit=750_000, revisions=("revision-001", "revision-002"))

    assert effective_credit(state) == 750_000
    assert state.disposition.residual_cents == 450_000
    assert evaluate(state).success


def test_authorized_cancel_can_be_recorded_but_cannot_pass_a_proceed_only_objective():
    state = reset(fixture_case())
    accept(
        state,
        {
            "type": "set_disposition",
            "disposition": "cancel",
            "reason": "Buyer permitted cancellation.",
        },
    )
    finish(state, disposition="cancel", credit=0, revisions=())

    score = evaluate(state)
    assert not score.success
    assert not next(d for d in score.diagnostics if d.name == "disposition").passed
    assert next(d for d in score.diagnostics if d.name == "authority").passed


def test_terminal_reads_and_report_cannot_change_business_outcome():
    state = reset(fixture_case(acceptable_dispositions=("cancel",)))
    accept(
        state,
        {"type": "set_disposition", "disposition": "cancel", "reason": "Authorized cancellation."},
    )
    original = state.disposition.model_dump()
    for action in [
        credit_draft(),
        {"type": "set_disposition", "disposition": "proceed", "reason": "Reconsider."},
    ]:
        assert not step(state, action).accepted
    accept(state, {"type": "read", "resource": "document", "resource_id": "document-002"})
    finish(state, disposition="cancel", credit=0, revisions=())
    accept(state, {"type": "read", "resource": "summary"})

    assert state.disposition.model_dump() == original
    assert not state.revisions
    assert evaluate(state).success


def test_scheduled_private_payloads_are_not_observable_before_their_public_event():
    case = fixture_case()
    private_document = case.documents[1].model_copy(
        update={
            "id": "document-003",
            "body": "PRIVATE-PAYLOAD-UNTIL-DELIVERY",
            "title": "Specialist follow-up",
        }
    )
    policy = case.counterpart.model_copy(
        update={
            "specialist_document": private_document,
            "specialist_latency_minutes": 30,
        }
    )
    state = reset(case.model_copy(update={"counterpart": policy}))
    result = accept(
        state,
        {
            "type": "request",
            "request": {
                "intent": "specialist_evidence",
                "document_id": "document-001",
            },
        },
    )

    serialized = json.dumps(result.model_dump(mode="json"))
    assert "PRIVATE-PAYLOAD-UNTIL-DELIVERY" not in serialized
    assert "seller_max_credit_cents" not in serialized
    assert "acceptable_dispositions" not in serialized
    assert "pending_events" not in serialized
    with pytest.raises(ValueError, match="Unknown public document"):
        observe(state, "document", "document-003")
    accept(state, {"type": "wait_until", "at": (START + timedelta(minutes=35)).isoformat()})
    assert observe(state, "document", "document-003")["body"] == "PRIVATE-PAYLOAD-UNTIL-DELIVERY"


@pytest.mark.parametrize("invalid_amount", [True, 800_000.5, "800000"])
def test_non_integer_money_inputs_are_rejected_without_clock_or_business_effects(invalid_amount):
    state = reset(fixture_case())
    result = step(state, credit_draft(invalid_amount))
    assert not result.accepted
    assert state.attempts == 1
    assert state.now == START
    assert not state.revisions


def test_duplicate_signature_requests_do_not_reschedule_or_duplicate_counterparts():
    case = fixture_case()
    policy = case.counterpart.model_copy(update={"signature_minutes": {"buyer": 15, "seller": 15}})
    state = reset(case.model_copy(update={"counterpart": policy}))
    accept(state, credit_draft())
    accept(state, {"type": "send_for_signature", "revision_id": "revision-001"})
    original_deliveries = [(event.at, event.payload) for event in state.pending_events]
    accept(state, {"type": "send_for_signature", "revision_id": "revision-001"})
    assert [(event.at, event.payload) for event in state.pending_events] == original_deliveries
    accept(state, {"type": "wait_until", "at": (START + timedelta(minutes=25)).isoformat()})
    accept(state, {"type": "send_for_signature", "revision_id": "revision-001"})

    assert len(state.envelopes) == 1
    assert len(state.signatures) == 2
    assert len(state.executed_at) == 1
    assert not state.pending_events
    assert effective_credit(state) == 800_000


def test_duplicate_requests_receive_one_reply_and_new_drafts_start_unsigned():
    state = reset(fixture_case())
    request = {
        "type": "request",
        "request": {
            "intent": "seller_negotiation",
            "terms": {"type": "credit", "amount_cents": 800_000},
            "evidence_ids": ["document-002"],
        },
    }
    accept(state, request)
    accept(state, request)
    assert len(state.pending_events) == 1
    accept(state, {"type": "wait_until", "at": (START + timedelta(minutes=20)).isoformat()})
    assert len([message for message in state.messages if message.sender == "seller"]) == 1
    assert effective_credit(state) == 0

    accept(state, credit_draft())
    accept(state, {"type": "send_for_signature", "revision_id": "revision-001"})
    accept(state, credit_draft(900_000))
    second = observe(state, "amendment", "revision-002")
    assert all(not signer["signed"] for signer in second["signers"])
    assert second["executed_at"] is None


def test_late_signature_after_expiry_cannot_make_partial_revision_effective():
    deadline = START + timedelta(minutes=20)
    case = fixture_case(deadline=deadline)
    policy = case.counterpart.model_copy(update={"signature_minutes": {"buyer": 5, "seller": 15}})
    state = reset(case.model_copy(update={"counterpart": policy}))
    accept(state, credit_draft())
    accept(state, {"type": "send_for_signature", "revision_id": "revision-001"})
    accept(state, {"type": "wait_until", "at": (START + timedelta(minutes=30)).isoformat()})

    assert state.business_status == "expired"
    assert {signature.party for signature in state.signatures} == {"buyer"}
    assert not state.executed_at
    assert effective_credit(state) == 0
    finish(state, disposition="unresolved", credit=800_000, revisions=("revision-001",))
    score = evaluate(state)
    assert not score.success
    assert not next(d for d in score.diagnostics if d.name == "supported_final_claims").passed


@pytest.mark.parametrize("field", ["seller_available_at", "maximum_extension"])
def test_naive_optional_counterpart_timestamps_are_rejected_at_fixture_validation(field):
    from pydantic import ValidationError

    data = fixture_case().counterpart.model_dump()
    data[field] = datetime(2026, 10, 5, 10)  # noqa: DTZ001 - deliberately invalid input
    with pytest.raises(ValidationError, match="timezone"):
        CounterpartPolicy.model_validate(data)


def test_naive_authority_extension_limit_is_rejected_at_fixture_validation():
    from pydantic import ValidationError

    data = fixture_case().authorities[0].model_dump()
    data["extension_limit"] = datetime(2026, 10, 5, 10)  # noqa: DTZ001 - deliberately invalid input
    with pytest.raises(ValidationError, match="timezone"):
        Authority.model_validate(data)


def test_timely_extension_keeps_already_routed_credit_signature_valid():
    """Credit and extension may be coordinated in either order before original expiry."""
    original_deadline = START + timedelta(minutes=30)
    extended_deadline = START + timedelta(hours=2)
    case = fixture_case(deadline=original_deadline)
    policy = case.counterpart.model_copy(
        update={
            "signature_minutes": {"buyer": 0, "seller": 50},
            "extension_signature_minutes": {"buyer": 5, "seller": 10},
        }
    )
    state = reset(case.model_copy(update={"counterpart": policy}))
    accept(state, credit_draft())
    accept(state, {"type": "send_for_signature", "revision_id": "revision-001"})
    accept(
        state,
        {
            "type": "draft_amendment",
            "terms": {
                "type": "extension",
                "deadline": extended_deadline.isoformat(),
            },
        },
    )
    accept(state, {"type": "send_for_signature", "revision_id": "revision-002"})
    accept(state, {"type": "wait_until", "at": (START + timedelta(hours=1)).isoformat()})
    accept(
        state,
        {
            "type": "set_disposition",
            "disposition": "proceed",
            "reason": "Both amendments executed.",
        },
    )
    finish(state, revisions=("revision-001", "revision-002"))

    assert state.executed_at["revision-002"] == original_deadline
    assert state.executed_at["revision-001"] == START + timedelta(hours=1)
    assert state.deadline == extended_deadline
    assert evaluate(state).success


def test_revision_change_fixture_starts_with_real_old_signatures_and_unsigned_changed_terms():
    """The model, not only the witness script, must encounter the curated revision trap."""
    state = reset(load_case("case-04"))
    old = operative_revision(state)
    assert old is not None, "Changed-revision fixture needs an already executed amendment."
    newer = [revision for revision in state.revisions.values() if revision.version > old.version]
    assert newer, "Changed-revision fixture needs a changed unsigned amendment at reset."
    latest = max(newer, key=lambda revision: revision.version)
    assert latest.terms != old.terms
    assert latest.id not in state.executed_at
    assert not any(signature.revision_id == latest.id for signature in state.signatures)
    assert not step(
        state,
        {
            "type": "set_disposition",
            "disposition": "proceed",
            "reason": "Assume signatures transferred.",
        },
    ).accepted


def test_missing_signature_fixture_opens_with_partial_signing_visible_to_the_model():
    state = reset(load_case("case-03"))
    assert state.revisions
    revision = next(iter(state.revisions.values()))
    public_revision = observe(state, "amendment", revision.id)
    signed = [party for party in public_revision["signers"] if party["signed"]]
    assert 0 < len(signed) < len(revision.required_signers)
    assert not state.executed_at
    assert effective_credit(state) == 0
    assert state.attempts == 0
    assert "prelude" not in json.dumps(observe(state))
