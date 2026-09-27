"""Small deterministic fictional transaction engine; no Inspect or application dependencies."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Frozen(Model):
    model_config = ConfigDict(extra="forbid", frozen=True)


def aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a timezone offset")
    return value


def optional_aware(value: datetime | None) -> datetime | None:
    return aware(value) if value is not None else None


class CreditTerms(Frozen):
    type: Literal["credit"] = "credit"
    amount_cents: int = Field(ge=0, strict=True)


class ExtensionTerms(Frozen):
    type: Literal["extension"] = "extension"
    deadline: datetime
    _aware = field_validator("deadline")(aware)


Terms = Annotated[CreditTerms | ExtensionTerms, Field(discriminator="type")]


class Document(Frozen):
    id: str
    title: str
    body: str
    kind: Literal["contract", "inspection", "quote", "other"]
    eligible_cost_cents: int | None = Field(default=None, ge=0)
    issued_at: datetime
    _aware = field_validator("issued_at")(aware)


class Message(Frozen):
    id: str
    sender: str
    body: str
    at: datetime
    _aware = field_validator("at")(aware)


class Authority(Frozen):
    id: str
    effective_at: datetime
    valid_through: datetime
    permissions: tuple[Literal["credit", "extension", "proceed", "cancel"], ...]
    minimum_credit_cents: int = Field(default=0, ge=0)
    maximum_credit_cents: int = Field(default=100_000_000, ge=0)
    maximum_residual_cents: int = Field(ge=0)
    extension_limit: datetime | None = None
    source_id: str
    _aware = field_validator("effective_at", "valid_through")(aware)
    _optional_aware = field_validator("extension_limit")(optional_aware)


class CounterpartPolicy(Frozen):
    """Private deterministic counterpart implementation, never included in observe()."""

    specialist_document: Document | None = None
    specialist_latency_minutes: int = Field(default=30, ge=0)
    required_evidence_ids: tuple[str, ...]
    seller_max_credit_cents: int = Field(ge=0)
    seller_reply_minutes: int = Field(default=15, ge=0)
    buyer_reply_minutes: int = Field(default=10, ge=0)
    buyer_grants: tuple[Authority, ...] = ()
    signature_minutes: dict[str, int]
    extension_signature_minutes: dict[str, int] = Field(default_factory=dict)
    seller_available_at: datetime | None = None
    maximum_extension: datetime | None = None
    _optional_aware = field_validator("seller_available_at", "maximum_extension")(optional_aware)


class Case(Frozen):
    id: str
    title: str
    split: Literal["development", "evaluation"]
    brief: str
    property: str
    buyer: str
    seller: str
    start_at: datetime
    deadline: datetime
    required_signers: tuple[str, ...] = ("buyer", "seller")
    public_policy: str
    documents: tuple[Document, ...]
    messages: tuple[Message, ...]
    authorities: tuple[Authority, ...]
    counterpart: CounterpartPolicy
    acceptable_dispositions: tuple[Literal["proceed", "cancel"], ...]
    action_limit: int = Field(default=80, ge=1)
    prelude: tuple[dict[str, Any], ...] = ()
    _aware = field_validator("start_at", "deadline")(aware)


class SpecialistRequest(Frozen):
    intent: Literal["specialist_evidence"] = "specialist_evidence"
    document_id: str


class BuyerAuthorityRequest(Frozen):
    intent: Literal["buyer_authorization"] = "buyer_authorization"
    permission: Literal["credit", "extension", "proceed", "cancel"]


class SellerNegotiationRequest(Frozen):
    intent: Literal["seller_negotiation"] = "seller_negotiation"
    terms: Terms
    evidence_ids: tuple[str, ...] = ()


RequestSpec = Annotated[
    SpecialistRequest | BuyerAuthorityRequest | SellerNegotiationRequest,
    Field(discriminator="intent"),
]


class ReadAction(Frozen):
    type: Literal["read"] = "read"
    resource: Literal["summary", "document", "message", "envelope", "amendment"] = "summary"
    resource_id: str | None = None


class RequestAction(Frozen):
    type: Literal["request"] = "request"
    request: RequestSpec


class DraftAmendmentAction(Frozen):
    type: Literal["draft_amendment"] = "draft_amendment"
    terms: Terms
    evidence_ids: tuple[str, ...] = ()


class SendForSignatureAction(Frozen):
    type: Literal["send_for_signature"] = "send_for_signature"
    revision_id: str


class WaitUntilAction(Frozen):
    type: Literal["wait_until"] = "wait_until"
    at: datetime
    _aware = field_validator("at")(aware)


class SetDispositionAction(Frozen):
    type: Literal["set_disposition"] = "set_disposition"
    disposition: Literal["proceed", "cancel"]
    reason: str = Field(min_length=1)


class FinalClaims(Frozen):
    disposition: Literal["proceed", "cancel", "unresolved"]
    effective_credit_cents: int = Field(default=0, ge=0, strict=True)
    executed_revision_ids: tuple[str, ...] = ()
    deadline: datetime | None = None
    _optional_aware = field_validator("deadline")(optional_aware)


class FinishAction(Frozen):
    type: Literal["finish"] = "finish"
    claims: FinalClaims
    report: str = ""


Action = Annotated[
    ReadAction
    | RequestAction
    | DraftAmendmentAction
    | SendForSignatureAction
    | WaitUntilAction
    | SetDispositionAction
    | FinishAction,
    Field(discriminator="type"),
]
ACTION_ADAPTER = TypeAdapter(Action)


def parse_action(value: dict[str, Any] | Any) -> Action:
    return ACTION_ADAPTER.validate_python(value)


class Revision(Frozen):
    id: str
    version: int
    terms: Terms
    evidence_ids: tuple[str, ...]
    created_at: datetime
    authority_id: str
    required_signers: tuple[str, ...]


class Signature(Frozen):
    id: str
    revision_id: str
    envelope_id: str
    party: str
    signed_at: datetime


class Envelope(Model):
    id: str
    revision_id: str
    created_at: datetime
    authority_id: str
    signing_deadline: datetime
    pending_parties: list[str] = Field(default_factory=list)
    attempts: int = 1


class Event(Model):
    id: str
    at: datetime
    sequence: int
    kind: Literal["document", "authority", "negotiation", "signature"]
    payload: dict[str, Any]


class Disposition(Frozen):
    kind: Literal["proceed", "cancel"]
    at: datetime
    authority_id: str
    revision_id: str | None
    effective_credit_cents: int
    residual_cents: int | None
    deadline: datetime
    reason: str


class State(Model):
    case: Case
    now: datetime
    deadline: datetime
    documents: dict[str, Document]
    messages: list[Message]
    authorities: list[Authority]
    revisions: dict[str, Revision] = Field(default_factory=dict)
    envelopes: dict[str, Envelope] = Field(default_factory=dict)
    signatures: list[Signature] = Field(default_factory=list)
    executed_at: dict[str, datetime] = Field(default_factory=dict)
    pending_events: list[Event] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    requests: dict[str, str] = Field(default_factory=dict)
    disposition: Disposition | None = None
    business_status: Literal["open", "proceed", "cancel", "expired"] = "open"
    finished: bool = False
    final_claims: FinalClaims | None = None
    final_report: str = ""
    attempts: int = 0
    budget_exhausted: bool = False
    next_sequence: int = 1


class TransitionResult(Model):
    accepted: bool
    reason: str
    observation: dict[str, Any]
    events: list[dict[str, Any]]


def list_cases() -> list[str]:
    return sorted(
        path.stem for path in (Path(__file__).resolve().parent.parent / "cases").glob("case-*.json")
    )


def load_case(case: str | Path) -> Case:
    path = Path(case)
    if not path.is_file():
        directory = Path(__file__).resolve().parent.parent / "cases"
        path = directory / f"{case}.json"
        if not path.is_file() and str(case).startswith("synth-"):
            path = directory / "generated" / f"{case}.json"
    return Case.model_validate_json(path.read_text())


def reset(case: Case | str | Path) -> State:
    fixture = load_case(case) if isinstance(case, (str, Path)) else case.model_copy(deep=True)
    state = State(
        case=fixture,
        now=fixture.start_at,
        deadline=fixture.deadline,
        documents={d.id: d for d in fixture.documents},
        messages=list(fixture.messages),
        authorities=list(fixture.authorities),
    )
    # Curated opening records are generated by the identical transition path,
    # rather than manually declaring signatures or execution to be true.
    for action in fixture.prelude:
        result = step(state, action)
        if not result.accepted:
            raise ValueError(f"Invalid fixture prelude for {fixture.id}: {result.reason}")
        state.history[-1]["origin"] = "fixture_initialization"
    if state.business_status != "open" or state.finished:
        raise ValueError("Fixture prelude must leave an open transaction for the agent.")
    state.attempts = 0
    state.budget_exhausted = False
    return state


def current_authority(state: State, at: datetime | None = None) -> Authority | None:
    when = at or state.now
    applicable = [a for a in state.authorities if a.effective_at <= when]
    if not applicable:
        return None
    latest = max(enumerate(applicable), key=lambda pair: (pair[1].effective_at, pair[0]))[1]
    return latest if when <= latest.valid_through else None


def authority_for(state: State, permission: str, terms: Terms | None = None) -> Authority | None:
    auth = current_authority(state)
    if not auth or permission not in auth.permissions:
        return None
    if (
        isinstance(terms, CreditTerms)
        and not auth.minimum_credit_cents <= terms.amount_cents <= auth.maximum_credit_cents
    ):
        return None
    if isinstance(terms, ExtensionTerms) and (
        not auth.extension_limit or terms.deadline > auth.extension_limit
    ):
        return None
    return auth


def revision_signers(state: State, revision_id: str) -> set[str]:
    return {s.party for s in state.signatures if s.revision_id == revision_id}


def operative_revision(state: State) -> Revision | None:
    credits = [
        r
        for r in state.revisions.values()
        if r.id in state.executed_at and isinstance(r.terms, CreditTerms)
    ]
    return max(credits, key=lambda r: (state.executed_at[r.id], r.version)) if credits else None


def effective_credit(state: State) -> int:
    operative = operative_revision(state)
    return (
        operative.terms.amount_cents
        if operative and isinstance(operative.terms, CreditTerms)
        else 0
    )


def eligible_cost(state: State) -> int | None:
    values = [
        d.eligible_cost_cents
        for d in state.documents.values()
        if d.kind == "quote" and d.eligible_cost_cents is not None
    ]
    return max(values) if values else None


def _public_revision(state: State, revision: Revision) -> dict[str, Any]:
    data = revision.model_dump(mode="json")
    signed = revision_signers(state, revision.id)
    data.update(
        status="executed"
        if revision.id in state.executed_at
        else "partially_signed"
        if signed
        else "draft",
        executed_at=state.executed_at.get(revision.id).isoformat()
        if revision.id in state.executed_at
        else None,
        signers=[
            {
                "party": p,
                "signed": p in signed,
                "signed_at": next(
                    (
                        s.signed_at.isoformat()
                        for s in state.signatures
                        if s.revision_id == revision.id and s.party == p
                    ),
                    None,
                ),
            }
            for p in revision.required_signers
        ],
    )
    return data


def observe(
    state: State, resource: str = "summary", resource_id: str | None = None
) -> dict[str, Any]:
    """Whitelist public resources; never serialize case, scheduled payloads, or evaluator targets."""
    if resource == "document":
        if resource_id not in state.documents:
            raise ValueError("Unknown public document ID; use the summary resource list.")
        return state.documents[resource_id].model_dump(mode="json")
    if resource == "message":
        message = next((m for m in state.messages if m.id == resource_id), None)
        if message is None:
            raise ValueError("Unknown public message ID; use the summary resource list.")
        return message.model_dump(mode="json")
    if resource == "amendment":
        if resource_id not in state.revisions:
            raise ValueError("Unknown public amendment ID; use the summary resource list.")
        return _public_revision(state, state.revisions[resource_id])
    if resource == "envelope":
        if resource_id not in state.envelopes:
            raise ValueError("Unknown public envelope ID; use the summary resource list.")
        env = state.envelopes[resource_id]
        return {
            **env.model_dump(mode="json"),
            "signatures": [
                s.model_dump(mode="json") for s in state.signatures if s.envelope_id == env.id
            ],
            "revision": _public_revision(state, state.revisions[env.revision_id]),
        }
    if resource != "summary":
        raise ValueError("Unknown resource type.")
    auth = current_authority(state)
    operative = operative_revision(state)
    return {
        "case_id": state.case.id,
        "title": state.case.title,
        "brief": state.case.brief,
        "property": state.case.property,
        "buyer": state.case.buyer,
        "seller": state.case.seller,
        "policy": state.case.public_policy,
        "now": state.now.isoformat(),
        "deadline": state.deadline.isoformat(),
        "business_status": state.business_status,
        "finished": state.finished,
        "attempts": state.attempts,
        "action_limit": state.case.action_limit,
        "budget_exhausted": state.budget_exhausted,
        "authority": auth.model_dump(mode="json") if auth else None,
        "required_signers": list(state.case.required_signers),
        "resources": {
            "documents": [
                {"id": d.id, "title": d.title, "kind": d.kind} for d in state.documents.values()
            ],
            "messages": [
                {"id": m.id, "sender": m.sender, "at": m.at.isoformat()} for m in state.messages
            ],
            "amendments": list(state.revisions),
            "envelopes": list(state.envelopes),
        },
        "documents": [d.model_dump(mode="json") for d in state.documents.values()],
        "messages": [m.model_dump(mode="json") for m in state.messages],
        "amendments": [_public_revision(state, r) for r in state.revisions.values()],
        "envelopes": [observe(state, "envelope", eid) for eid in state.envelopes],
        "operative_revision_id": operative.id if operative else None,
        "effective_credit_cents": effective_credit(state),
        "eligible_cost_cents": eligible_cost(state),
        "disposition": state.disposition.model_dump(mode="json") if state.disposition else None,
        "final_claims": state.final_claims.model_dump(mode="json") if state.final_claims else None,
        "final_report": state.final_report,
    }


def _emit(state: State, kind: str, message: str, **data: Any) -> None:
    state.events.append(
        {
            "id": f"event-{len(state.events) + 1:03}",
            "at": state.now.isoformat(),
            "kind": kind,
            "message": message,
            **data,
        }
    )


def _message(state: State, sender: str, body: str) -> None:
    state.messages.append(
        Message(id=f"message-{len(state.messages) + 1:03}", sender=sender, body=body, at=state.now)
    )


def _schedule(state: State, kind: str, delay: int, payload: dict[str, Any]) -> None:
    seq = state.next_sequence
    state.next_sequence += 1
    state.pending_events.append(
        Event(
            id=f"scheduled-{seq:03}",
            at=state.now + timedelta(minutes=delay),
            sequence=seq,
            kind=kind,
            payload=payload,
        )
    )
    state.pending_events.sort(key=lambda event: (event.at, event.sequence))


def _has_evidence(state: State, ids: tuple[str, ...] | list[str]) -> bool:
    return all(i in state.documents for i in ids) and set(
        state.case.counterpart.required_evidence_ids
    ).issubset(ids)


def _seller_accepts(state: State, terms: Terms, evidence_ids: tuple[str, ...] | list[str]) -> bool:
    policy = state.case.counterpart
    if isinstance(terms, CreditTerms):
        return (
            _has_evidence(state, evidence_ids)
            and terms.amount_cents <= policy.seller_max_credit_cents
        )
    return bool(
        policy.maximum_extension and state.deadline < terms.deadline <= policy.maximum_extension
    )


def _apply_event(state: State, event: Event) -> None:
    # Terminal outcomes are immutable. Responses/signatures arriving afterward have no business effect.
    if state.business_status != "open" or state.finished:
        _emit(
            state,
            "event_ignored",
            "Scheduled response arrived after the transaction became terminal.",
            scheduled_id=event.id,
        )
        return
    p = event.payload
    if event.kind == "document":
        doc = Document.model_validate(p["document"])
        if doc.id not in state.documents:
            state.documents[doc.id] = doc
            _message(
                state, "specialist", f"Specialist evidence is available as {doc.id}: {doc.title}."
            )
            _emit(state, "document_received", f"Received {doc.title}.", evidence_ids=[doc.id])
    elif event.kind == "authority":
        auth = Authority.model_validate(p["authority"]).model_copy(
            update={"effective_at": state.now}
        )
        if auth.id not in {a.id for a in state.authorities}:
            state.authorities.append(auth)
            _message(
                state,
                "buyer",
                f"Authorization {auth.id}: permissions {', '.join(auth.permissions)}; credit range {auth.minimum_credit_cents}–{auth.maximum_credit_cents} cents; residual limit {auth.maximum_residual_cents} cents; valid through {auth.valid_through.isoformat()}.",
            )
            _emit(
                state,
                "authority_received",
                "Buyer provided scoped authority.",
                authority_id=auth.id,
            )
    elif event.kind == "negotiation":
        terms = TypeAdapter(Terms).validate_python(p["terms"])
        accepted = _seller_accepts(state, terms, p["evidence_ids"])
        _message(
            state,
            "seller",
            (
                "Informal agreement to the proposed terms. "
                if accepted
                else "The proposed terms are not accepted. "
            )
            + "No amendment is executed by this message. Route an immutable revision for all required signatures.",
        )
        _emit(
            state,
            "seller_reply",
            "Seller informally accepted terms." if accepted else "Seller declined proposed terms.",
            informally_accepted=accepted,
            terms=p["terms"],
        )
    elif event.kind == "signature":
        env = state.envelopes.get(p["envelope_id"])
        if not env or env.revision_id != p["revision_id"]:
            _emit(
                state, "signature_ignored", "Envelope no longer references the scheduled revision."
            )
            return
        party = p["party"]
        if party in env.pending_parties:
            env.pending_parties.remove(party)
        revision = state.revisions[env.revision_id]
        if party in revision_signers(state, revision.id):
            return
        if state.now > state.deadline or state.now > env.signing_deadline:
            _emit(
                state,
                "signature_ignored",
                "Signature arrived after the envelope's execution deadline.",
                revision_id=revision.id,
                party=party,
            )
            return
        policy = state.case.counterpart
        if (
            party == "seller"
            and policy.seller_available_at
            and state.now < policy.seller_available_at
        ):
            _message(
                state,
                "seller",
                f"Envelope {env.id} remains incomplete: seller is unavailable until {policy.seller_available_at.isoformat()}. Route the same revision again after that time to request the missing signature.",
            )
            _emit(
                state,
                "signature_missing",
                "Seller signature is still missing.",
                revision_id=revision.id,
                party=party,
            )
            return
        if party == "seller" and not _seller_accepts(state, revision.terms, revision.evidence_ids):
            _message(
                state,
                "seller",
                f"Seller declined to sign {revision.id}; terms/evidence do not meet the disclosed requirements.",
            )
            _emit(
                state,
                "signature_declined",
                "Seller declined the amendment.",
                revision_id=revision.id,
                party=party,
            )
            return
        sig = Signature(
            id=f"signature-{len(state.signatures) + 1:03}",
            revision_id=revision.id,
            envelope_id=env.id,
            party=party,
            signed_at=state.now,
        )
        state.signatures.append(sig)
        _emit(
            state,
            "signature_received",
            f"{party.title()} signed {revision.id}.",
            revision_id=revision.id,
            signature_id=sig.id,
            party=party,
        )
        if set(revision.required_signers).issubset(revision_signers(state, revision.id)):
            state.executed_at[revision.id] = state.now
            if isinstance(revision.terms, ExtensionTerms):
                state.deadline = max(state.deadline, revision.terms.deadline)
                for pending_envelope in state.envelopes.values():
                    if pending_envelope.revision_id not in state.executed_at:
                        pending_envelope.signing_deadline = state.deadline
            _emit(
                state,
                "amendment_executed",
                "Every required party signed this exact immutable revision in time.",
                revision_id=revision.id,
            )


def _expire(state: State, inclusive: bool = True) -> None:
    if state.business_status == "open" and (
        state.now >= state.deadline if inclusive else state.now > state.deadline
    ):
        state.business_status = "expired"
        _emit(
            state,
            "deadline_expired",
            "The response deadline passed without an authorized disposition.",
        )


def _advance(state: State, target: datetime, defer_exact_expiry: bool = False) -> None:
    while state.pending_events and state.pending_events[0].at <= target:
        event = state.pending_events[0]
        if state.business_status == "open" and event.at > state.deadline:
            state.now = state.deadline
            _expire(state)
        state.pending_events.pop(0)
        state.now = event.at
        _apply_event(state, event)
    if state.business_status == "open" and state.deadline < target:
        state.now = state.deadline
        _expire(state)
    state.now = target
    _expire(state, inclusive=not defer_exact_expiry)


def _validate(
    state: State, action: Action, *, completion: bool = False
) -> tuple[str | None, Authority | None]:
    if isinstance(action, ReadAction):
        try:
            observe(state, action.resource, action.resource_id)
        except ValueError as error:
            return str(error), None
        return None, None
    if isinstance(action, FinishAction):
        return ("The attempt has already been finished.", None) if state.finished else (None, None)
    if state.finished or state.business_status != "open":
        return (
            "Business changes are rejected after a terminal outcome; read and finish remain available.",
            None,
        )
    if state.budget_exhausted:
        return "The domain action limit is exhausted; read and finish remain available.", None
    if isinstance(action, WaitUntilAction):
        return (
            ("Wait target must be later than the current simulated time.", None)
            if action.at <= state.now
            else (None, None)
        )
    if isinstance(action, RequestAction):
        request = action.request
        if isinstance(request, SpecialistRequest):
            doc = state.documents.get(request.document_id)
            if not doc or doc.kind != "inspection":
                return (
                    "Specialist evidence requests must reference a public inspection document.",
                    None,
                )
            if not state.case.counterpart.specialist_document:
                return "The complete specialist quote is already in the public documents.", None
            return None, None
        if isinstance(request, BuyerAuthorityRequest):
            if not any(
                request.permission in a.permissions for a in state.case.counterpart.buyer_grants
            ):
                return (
                    "The buyer has not offered additional authority for that permission; review buyer instructions.",
                    None,
                )
            return None, None
        terms, ids = request.terms, request.evidence_ids
    elif isinstance(action, DraftAmendmentAction):
        terms, ids = action.terms, action.evidence_ids
    elif isinstance(action, SendForSignatureAction):
        if action.revision_id not in state.revisions:
            return "Unknown revision ID.", None
        revision = state.revisions[action.revision_id]
        terms, ids = revision.terms, revision.evidence_ids
    elif isinstance(action, SetDispositionAction):
        auth = authority_for(state, action.disposition)
        if auth is None:
            return f"Current buyer authority does not permit {action.disposition}.", None
        if action.disposition == "proceed" and completion:
            cost = eligible_cost(state)
            if cost is None:
                return "Proceed requires a specialist quote documenting eligible cost.", None
            residual = max(0, cost - effective_credit(state))
            if residual > auth.maximum_residual_cents:
                return (
                    f"Effective signed terms leave {residual} cents residual, above buyer's {auth.maximum_residual_cents}-cent limit.",
                    None,
                )
        return None, auth
    else:
        raise AssertionError("Unhandled action")
    if any(i not in state.documents for i in ids):
        return "Evidence IDs must reference available public documents.", None
    permission = "credit" if isinstance(terms, CreditTerms) else "extension"
    auth = authority_for(state, permission, terms)
    if auth is None:
        return "Current buyer authority does not permit these terms.", None
    if isinstance(terms, ExtensionTerms) and terms.deadline <= state.deadline:
        return "An extension must move the current deadline later.", None
    return None, auth


def _request_key(request: RequestSpec) -> str:
    return json.dumps(request.model_dump(mode="json"), sort_keys=True)


def _commit(state: State, action: Action, auth: Authority | None) -> str:
    if isinstance(action, RequestAction):
        request = action.request
        key = _request_key(request)
        if key in state.requests:
            return f"Identical request already recorded as {state.requests[key]}; no duplicate response scheduled."
        state.requests[key] = f"request-{len(state.requests) + 1:03}"
        policy = state.case.counterpart
        if isinstance(request, SpecialistRequest):
            _schedule(
                state,
                "document",
                policy.specialist_latency_minutes,
                {"document": policy.specialist_document.model_dump(mode="json")},
            )
        elif isinstance(request, BuyerAuthorityRequest):
            grant = next(a for a in policy.buyer_grants if request.permission in a.permissions)
            _schedule(
                state,
                "authority",
                policy.buyer_reply_minutes,
                {"authority": grant.model_dump(mode="json")},
            )
        else:
            _schedule(
                state,
                "negotiation",
                policy.seller_reply_minutes,
                {
                    "terms": request.terms.model_dump(mode="json"),
                    "evidence_ids": list(request.evidence_ids),
                },
            )
        _emit(
            state,
            "request_sent",
            "Counterpart request recorded; a response is pending.",
            intent=request.intent,
        )
        return "Request accepted; wait or continue coordination to receive its response."
    if isinstance(action, DraftAmendmentAction):
        assert auth
        # Identical drafts are harmless but retain a single immutable revision.
        existing = next(
            (
                r
                for r in state.revisions.values()
                if r.terms == action.terms and r.evidence_ids == action.evidence_ids
            ),
            None,
        )
        if existing:
            return f"Identical terms and evidence already exist as {existing.id}; no new revision created."
        n = len(state.revisions) + 1
        revision = Revision(
            id=f"revision-{n:03}",
            version=n,
            terms=action.terms,
            evidence_ids=action.evidence_ids,
            created_at=state.now,
            authority_id=auth.id,
            required_signers=state.case.required_signers,
        )
        state.revisions[revision.id] = revision
        _emit(
            state,
            "amendment_drafted",
            "Created immutable amendment revision; no signatures or execution implied.",
            revision_id=revision.id,
        )
        return f"Created {revision.id}."
    if isinstance(action, SendForSignatureAction):
        assert auth
        revision = state.revisions[action.revision_id]
        if revision.id in state.executed_at:
            return "This exact revision is already executed; no signatures duplicated."
        env = next((e for e in state.envelopes.values() if e.revision_id == revision.id), None)
        if env is None:
            env = Envelope(
                id=f"envelope-{len(state.envelopes) + 1:03}",
                revision_id=revision.id,
                created_at=state.now,
                authority_id=auth.id,
                signing_deadline=state.deadline,
            )
            state.envelopes[env.id] = env
        else:
            env.attempts += 1
            env.signing_deadline = state.deadline
        signed = revision_signers(state, revision.id)
        latency = (
            state.case.counterpart.extension_signature_minutes
            if isinstance(revision.terms, ExtensionTerms)
            else state.case.counterpart.signature_minutes
        )
        for party in revision.required_signers:
            if party not in signed and party not in env.pending_parties:
                env.pending_parties.append(party)
                _schedule(
                    state,
                    "signature",
                    latency.get(party, state.case.counterpart.signature_minutes[party]),
                    {"envelope_id": env.id, "revision_id": revision.id, "party": party},
                )
        _emit(
            state,
            "signature_requested",
            "Routed exact revision; only counterpart events can create signatures.",
            revision_id=revision.id,
            envelope_id=env.id,
        )
        return f"Signature request recorded in {env.id}; inspect the envelope to verify execution."
    if isinstance(action, SetDispositionAction):
        assert auth
        cost = eligible_cost(state)
        operative = operative_revision(state)
        state.disposition = Disposition(
            kind=action.disposition,
            at=state.now,
            authority_id=auth.id,
            revision_id=operative.id if operative else None,
            effective_credit_cents=effective_credit(state),
            residual_cents=max(0, cost - effective_credit(state)) if cost is not None else None,
            deadline=state.deadline,
            reason=action.reason,
        )
        state.business_status = action.disposition
        _emit(
            state,
            "disposition_set",
            f"Recorded authorized {action.disposition}.",
            authority_id=auth.id,
        )
        return f"Transaction disposition is {action.disposition}. Read-only inspection and finish remain available."
    if isinstance(action, FinishAction):
        state.finished = True
        state.final_claims = action.claims
        state.final_report = action.report
        _emit(
            state,
            "attempt_finished",
            "Final claims recorded; this action creates no business effects.",
        )
        return "Attempt finished. Claims will be checked against underlying evidence."
    if isinstance(action, WaitUntilAction):
        return "Wait completed; all intervening events processed in timestamp/sequence order."
    return "Read completed."


def step(state: State, action: Action | dict[str, Any]) -> TransitionResult:
    """Mutate only this sample. Invalid attempts count; accepted coordination consumes five minutes."""
    event_start = len(state.events)
    started = state.now
    raw = action.model_dump(mode="json") if isinstance(action, BaseModel) else action
    state.attempts += 1
    try:
        parsed = parse_action(action)
    except (ValueError, TypeError) as error:
        reason = f"Invalid typed action: {error}"
        state.history.append(
            {
                "action": raw,
                "started_at": started.isoformat(),
                "completed_at": state.now.isoformat(),
                "accepted": False,
                "reason": reason,
                "authority_id": None,
            }
        )
        if state.attempts >= state.case.action_limit:
            state.budget_exhausted = True
        return TransitionResult(
            accepted=False, reason=reason, observation=observe(state), events=[]
        )
    if state.attempts > state.case.action_limit:
        state.budget_exhausted = True
    error, auth = _validate(state, parsed)
    if error is None and isinstance(parsed, WaitUntilAction):
        _advance(state, parsed.at)
    elif error is None and not isinstance(parsed, (ReadAction, FinishAction)):
        _advance(state, state.now + timedelta(minutes=5), defer_exact_expiry=True)
        error, auth = _validate(state, parsed, completion=True)
    reason = error or _commit(state, parsed, auth)
    accepted = error is None
    if accepted and not isinstance(parsed, (ReadAction, FinishAction)):
        # A zero-latency event may have been scheduled by the committed request.
        _advance(state, state.now, defer_exact_expiry=True)
        _expire(state)
    elif state.now != started:
        # A completion-time rejection still reaches the deadline and must expire.
        _expire(state)
    state.history.append(
        {
            "action": parsed.model_dump(mode="json"),
            "started_at": started.isoformat(),
            "completed_at": state.now.isoformat(),
            "accepted": accepted,
            "reason": reason,
            "authority_id": auth.id if auth else None,
        }
    )
    if state.attempts >= state.case.action_limit:
        state.budget_exhausted = True
    observation = (
        observe(state, parsed.resource, parsed.resource_id)
        if accepted and isinstance(parsed, ReadAction)
        else observe(state)
    )
    return TransitionResult(
        accepted=accepted, reason=reason, observation=observation, events=state.events[event_start:]
    )


def evaluate(state: State):
    from dealroom.score import evaluate as score_evaluate

    return score_evaluate(state)
