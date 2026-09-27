"""Objective predicates over records, with concise reviewer-only diagnostics."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from dealroom.domain import (
    CreditTerms,
    ExtensionTerms,
    Model,
    State,
    current_authority,
    effective_credit,
    eligible_cost,
    operative_revision,
    parse_action,
    revision_signers,
)


class Diagnostic(Model):
    name: str
    passed: bool
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)
    timestamps: list[str] = Field(default_factory=list)


class Evaluation(Model):
    success: bool
    reward: int
    outcome: str
    diagnostics: list[Diagnostic]
    attempts: int
    invalid_attempts: int
    finished: bool
    budget_exhausted: bool


def evaluate(state: State) -> Evaluation:
    """No status label or self-reported completion can substitute for evidence."""
    diagnostics: list[Diagnostic] = []

    def add(
        name: str,
        passed: bool,
        success_reason: str,
        failure_reason: str,
        ids: list[str] | None = None,
        times: list[str] | None = None,
    ) -> None:
        diagnostics.append(
            Diagnostic(
                name=name,
                passed=bool(passed),
                reason=success_reason if passed else failure_reason,
                evidence_ids=ids or [],
                timestamps=times or [],
            )
        )

    disposition = state.disposition
    add(
        "disposition",
        disposition is not None and disposition.kind in state.case.acceptable_dispositions,
        "Recorded disposition is acceptable for the case objective.",
        "No acceptable authorized disposition was recorded.",
        times=[disposition.at.isoformat()] if disposition else [state.now.isoformat()],
    )

    bad_authority: list[str] = []
    authority_ids: list[str] = []
    for item in state.history:
        if not item["accepted"] or not item.get("authority_id"):
            continue
        at = datetime.fromisoformat(item["completed_at"])
        auth = current_authority(state, at)
        authority_ids.append(item["authority_id"])
        if not auth or auth.id != item["authority_id"]:
            bad_authority.append(item["completed_at"])
            continue
        action = parse_action(item["action"])
        action_type = item["action"]["type"]
        terms: Any = getattr(action, "terms", None)
        if action_type == "request":
            terms = getattr(action.request, "terms", None)
        elif action_type == "send_for_signature":
            terms = state.revisions[action.revision_id].terms
        permission = (
            "credit"
            if isinstance(terms, CreditTerms)
            else "extension"
            if isinstance(terms, ExtensionTerms)
            else getattr(action, "disposition", None)
        )
        allowed = permission in auth.permissions
        if isinstance(terms, CreditTerms):
            allowed = (
                allowed
                and auth.minimum_credit_cents <= terms.amount_cents <= auth.maximum_credit_cents
            )
        if isinstance(terms, ExtensionTerms):
            allowed = (
                allowed
                and auth.extension_limit is not None
                and terms.deadline <= auth.extension_limit
            )
        if not allowed:
            bad_authority.append(item["completed_at"])
    authority_ok = disposition is not None and not bad_authority
    if disposition:
        auth = current_authority(state, disposition.at)
        authority_ok = authority_ok and bool(
            auth and auth.id == disposition.authority_id and disposition.kind in auth.permissions
        )
    add(
        "authority",
        authority_ok,
        "Accepted coordination and disposition were authorized at their completion times.",
        "Disposition is missing or relevant historical authority is invalid.",
        sorted(set(authority_ids)),
        bad_authority,
    )

    invalid_revisions: list[str] = []
    signature_reasons: list[str] = []
    relevant_revisions = set(state.executed_at)
    if state.final_claims:
        relevant_revisions.update(state.final_claims.executed_revision_ids)
        if state.final_claims.disposition == "proceed" and not state.executed_at:
            relevant_revisions.update(
                r.id for r in state.revisions.values() if isinstance(r.terms, CreditTerms)
            )
    for rid in sorted(relevant_revisions):
        revision = state.revisions.get(rid)
        if revision is None:
            invalid_revisions.append(rid)
            signature_reasons.append(f"{rid} does not exist")
            continue
        executed_at = state.executed_at.get(rid)
        records = [s for s in state.signatures if s.revision_id == rid]
        missing = set(revision.required_signers) - revision_signers(state, rid)
        if missing or executed_at is None:
            invalid_revisions.append(rid)
            signature_reasons.append(
                f"{rid} is unexecuted; missing signatures: {', '.join(sorted(missing)) or 'valid timely execution record'}"
            )
        for sig in records:
            env = state.envelopes.get(sig.envelope_id)
            if (
                not env
                or env.revision_id != rid
                or (executed_at is not None and sig.signed_at > executed_at)
                or sig.signed_at > env.signing_deadline
            ):
                invalid_revisions.append(rid)
    operative = operative_revision(state)
    credit = effective_credit(state)
    execution_needed = bool(disposition and disposition.kind == "proceed" and credit > 0)
    valid_execution = not invalid_revisions and (not execution_needed or operative is not None)
    add(
        "revision_signatures",
        valid_execution,
        "Executed terms have all required signatures on their exact immutable revision."
        if state.executed_at
        else "No executed amendment is claimed by the underlying records.",
        "; ".join(signature_reasons)
        or "A required signature is missing, late, or attached to another revision.",
        sorted(set(invalid_revisions)) or list(state.executed_at),
        [s.signed_at.isoformat() for s in state.signatures if s.revision_id in relevant_revisions],
    )
    effective_ok = bool(
        disposition
        and disposition.effective_credit_cents == credit
        and disposition.revision_id == (operative.id if operative else None)
    )
    add(
        "effective_terms",
        effective_ok,
        f"The operative executed credit is {credit} cents; informal messages and unsigned drafts contribute zero.",
        f"No valid disposition confirms the effective credit of {credit} cents; informal agreement is insufficient.",
        [operative.id] if operative else [],
    )

    deadline_ok = bool(
        disposition
        and disposition.at <= disposition.deadline
        and disposition.deadline == state.deadline
    )
    add(
        "deadline",
        deadline_ok,
        "Disposition was committed by the effective contractual deadline.",
        "The disposition is absent or was not completed within the effective deadline.",
        [
            r.id
            for r in state.revisions.values()
            if isinstance(r.terms, ExtensionTerms) and r.id in state.executed_at
        ],
        [state.deadline.isoformat()] + ([disposition.at.isoformat()] if disposition else []),
    )

    auth = current_authority(state, disposition.at) if disposition else None
    cost = eligible_cost(state)
    residual = max(0, cost - credit) if cost is not None else None
    buyer_ok = bool(
        disposition
        and auth
        and (
            (disposition.kind == "cancel" and "cancel" in auth.permissions)
            or (
                disposition.kind == "proceed"
                and residual is not None
                and residual <= auth.maximum_residual_cents
            )
        )
    )
    add(
        "buyer_constraint",
        buyer_ok,
        "Cancellation was expressly authorized."
        if disposition and disposition.kind == "cancel"
        else f"Residual eligible expense is {residual} cents within the buyer's stated limit.",
        (
            f"Buyer constraint is not satisfied: residual eligible expense is {residual} cents."
            if residual is not None
            else "Buyer constraint cannot yet be verified: no specialist quote is public."
        ),
        [d.id for d in state.documents.values() if d.kind == "quote"]
        + ([auth.source_id] if auth else []),
    )

    claims = state.final_claims
    actual_disposition = disposition.kind if disposition else "unresolved"
    claimed_ids = set(claims.executed_revision_ids) if claims else set()
    claims_ok = bool(
        claims
        and claims.disposition == actual_disposition
        and claims.effective_credit_cents == credit
        and claimed_ids.issubset(state.executed_at)
        and (claims.deadline is None or claims.deadline == state.deadline)
    )
    add(
        "supported_final_claims",
        claims_ok,
        "Every structured final claim is supported by the engine's evidence.",
        "Final claims are absent or assert unsupported disposition, credit, revision execution, or deadline.",
        sorted(claimed_ids),
        [state.now.isoformat()],
    )
    add(
        "completed_attempt",
        state.finished,
        "The attempt has a structured final report.",
        "The attempt ended without finish; budget exhaustion and incomplete execution are not completed outcomes.",
    )
    success = all(d.passed for d in diagnostics)
    outcome = (
        "success"
        if success
        else "budget_exhausted"
        if state.budget_exhausted and not state.finished
        else "incomplete"
        if not state.finished
        else "failure"
    )
    return Evaluation(
        success=success,
        reward=int(success),
        outcome=outcome,
        diagnostics=diagnostics,
        attempts=state.attempts,
        invalid_attempts=sum(not h["accepted"] for h in state.history),
        finished=state.finished,
        budget_exhausted=state.budget_exhausted,
    )
