"""Known-fixture validation actions, not an agent or a general-purpose solver.

These witnesses may know fixture implementation. Never describe their scores as
model performance. The flagship failure is deliberately scripted.
"""

from __future__ import annotations

from copy import deepcopy

EVIDENCE = ["document-002", "document-004"]


def credit(amount: int) -> dict:
    return {"type": "credit", "amount_cents": amount}


def draft(amount: int) -> dict:
    return {"type": "draft_amendment", "terms": credit(amount), "evidence_ids": EVIDENCE.copy()}


def route(revision: int = 1) -> dict:
    return {"type": "send_for_signature", "revision_id": f"revision-{revision:03}"}


def wait(at: str) -> dict:
    return {"type": "wait_until", "at": at}


def proceed() -> dict:
    return {
        "type": "set_disposition",
        "disposition": "proceed",
        "reason": "Exact signed revision satisfies the buyer's documented residual limit.",
    }


def finish(
    amount: int, revisions: list[int], disposition: str = "proceed", deadline: str | None = None
) -> dict:
    return {
        "type": "finish",
        "claims": {
            "disposition": disposition,
            "effective_credit_cents": amount,
            "executed_revision_ids": [f"revision-{i:03}" for i in revisions],
            "deadline": deadline,
        },
        "report": "Fixture validation report; structured claims reference the engine's underlying records.",
    }


FLAGSHIP_PREFIX = [
    {"type": "read", "resource": "summary"},
    {
        "type": "request",
        "request": {"intent": "specialist_evidence", "document_id": "document-002"},
    },
    wait("2026-10-05T10:10:00-04:00"),
    {"type": "read", "resource": "document", "resource_id": "document-004"},
    {
        "type": "request",
        "request": {
            "intent": "seller_negotiation",
            "terms": credit(800_000),
            "evidence_ids": EVIDENCE.copy(),
        },
    },
    wait("2026-10-05T10:50:00-04:00"),
    draft(800_000),
]

FLAGSHIP_FLAWED_SUFFIX = [
    {
        "type": "set_disposition",
        "disposition": "proceed",
        "reason": "Scripted mistake: seller's informal acceptance means the credit is complete.",
    },
    {
        "type": "finish",
        "claims": {
            "disposition": "proceed",
            "effective_credit_cents": 800_000,
            "executed_revision_ids": ["revision-001"],
        },
        "report": "Deliberately flawed scripted demonstration: treats informal agreement and a draft as completed execution.",
    },
]

FLAGSHIP_REPAIRED_SUFFIX = [
    route(),
    wait("2026-10-05T12:05:00-04:00"),
    {"type": "read", "resource": "envelope", "resource_id": "envelope-001"},
    proceed(),
    finish(800_000, [1]),
]

WITNESSES = {
    "case-01": deepcopy(FLAGSHIP_PREFIX + FLAGSHIP_REPAIRED_SUFFIX),
    "case-02": [
        {"type": "read"},
        draft(350_000),
        route(),
        wait("2026-10-06T10:00:00-04:00"),
        proceed(),
        finish(350_000, [1]),
    ],
    "case-03": [
        {"type": "read"},
        {"type": "read", "resource": "envelope", "resource_id": "envelope-001"},
        wait("2026-10-06T13:00:00-04:00"),
        route(),
        wait("2026-10-06T13:30:00-04:00"),
        proceed(),
        finish(750_000, [1]),
    ],
    "case-04": [
        {"type": "read"},
        {"type": "read", "resource": "amendment", "resource_id": "revision-002"},
        route(2),
        wait("2026-10-07T11:00:00-04:00"),
        proceed(),
        finish(800_000, [1, 2]),
    ],
    "case-05": [
        {"type": "read"},
        {
            "type": "draft_amendment",
            "terms": {"type": "extension", "deadline": "2026-10-07T14:00:00-04:00"},
            "evidence_ids": ["document-001"],
        },
        route(),
        wait("2026-10-07T09:25:00-04:00"),
        draft(850_000),
        route(2),
        wait("2026-10-07T12:40:00-04:00"),
        proceed(),
        finish(850_000, [1, 2], deadline="2026-10-07T14:00:00-04:00"),
    ],
    "case-06": [
        {"type": "read"},
        {
            "type": "request",
            "request": {
                "intent": "seller_negotiation",
                "terms": credit(900_000),
                "evidence_ids": EVIDENCE.copy(),
            },
        },
        wait("2026-10-08T09:25:00-04:00"),
        {"type": "request", "request": {"intent": "buyer_authorization", "permission": "cancel"}},
        wait("2026-10-08T09:45:00-04:00"),
        {
            "type": "set_disposition",
            "disposition": "cancel",
            "reason": "The disclosed seller maximum leaves 600000 cents residual, exceeding the buyer's 300000-cent limit; express cancellation authority received.",
        },
        finish(0, [], "cancel"),
    ],
}
