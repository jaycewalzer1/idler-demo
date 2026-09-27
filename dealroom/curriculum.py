"""Versioned training curriculum and structurally held-out transaction fixtures.

This is fixture generation, not a model solver. Private witnesses prove that the
unchanged domain and binary verifier admit a solution. Only train witnesses may
be exported as demonstrations; validation and sealed-test witnesses remain local
fixture-validation artifacts. Public model inputs must come from observe(), not
from Case JSON, the manifest, or this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from dealroom.domain import Case, effective_credit, load_case, observe, reset, step
from dealroom.score import evaluate

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEARNING_DIR = ROOT / "cases" / "learning-v1"
DEFAULT_SEED = 20260928
GENERATOR_VERSION = "1.0.0"
LearningSplit = Literal["train", "validation", "sealed_test"]
SPLIT_COUNTS = {"train": 256, "validation": 32, "sealed_test": 64}
TRAIN_FAMILIES = (
    "credit_execution",
    "quote_credit",
    "authority_credit",
    "extension_credit",
    "retry_credit",
    "revised_credit",
    "cancellation",
    "boundary_credit",
)
SEALED_FAMILIES = (
    "extension_authority_credit",
    "quote_retry_credit",
    "revised_extension_credit",
    "quote_authority_credit",
)

DEPENDENCIES = {
    "credit_execution": [("credit", "signatures"), ("signatures", "disposition")],
    "quote_credit": [("quote", "credit"), ("credit", "signatures"), ("signatures", "disposition")],
    "authority_credit": [
        ("authority", "credit"),
        ("credit", "signatures"),
        ("signatures", "disposition"),
    ],
    "extension_credit": [
        ("extension_signatures", "credit_signatures"),
        ("credit", "credit_signatures"),
        ("credit_signatures", "disposition"),
    ],
    "retry_credit": [
        ("credit", "missing_signature"),
        ("missing_signature", "retry"),
        ("seller_availability", "retry"),
        ("retry", "disposition"),
    ],
    "revised_credit": [
        ("old_execution", "replacement_credit"),
        ("replacement_credit", "new_execution"),
        ("new_execution", "disposition"),
    ],
    "cancellation": [
        ("infeasible_credit", "cancel_authority"),
        ("cancel_authority", "disposition"),
    ],
    "boundary_credit": [
        ("credit", "signatures_at_deadline"),
        ("signatures_at_deadline", "disposition_completion"),
    ],
    "extension_authority_credit": [
        ("extension_signatures", "credit_signatures"),
        ("authority", "credit"),
        ("credit", "credit_signatures"),
        ("credit_signatures", "disposition"),
    ],
    "quote_retry_credit": [
        ("quote", "disposition"),
        ("credit", "missing_signature"),
        ("missing_signature", "retry"),
        ("seller_availability", "retry"),
        ("retry", "disposition"),
    ],
    "revised_extension_credit": [
        ("old_execution", "replacement_credit"),
        ("extension_signatures", "new_execution"),
        ("replacement_credit", "new_execution"),
        ("new_execution", "disposition"),
    ],
    "quote_authority_credit": [
        ("quote", "credit"),
        ("authority", "credit"),
        ("credit", "signatures"),
        ("signatures", "disposition"),
    ],
}


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _opaque_id(seed: int, split: str, index: int) -> str:
    return (
        "learn-"
        + hashlib.sha256(
            f"dealroom-learning:{GENERATOR_VERSION}:{seed}:{split}:{index}".encode()
        ).hexdigest()[:16]
    )


def _draft(amount: int, evidence: list[str]) -> dict:
    return {
        "type": "draft_amendment",
        "terms": {"type": "credit", "amount_cents": amount},
        "evidence_ids": evidence,
    }


def _route(revision_id: str) -> dict:
    return {"type": "send_for_signature", "revision_id": revision_id}


def _wait(at: datetime) -> dict:
    return {"type": "wait_until", "at": at.isoformat()}


def _specs():
    for split, per_family in (("train", 32), ("validation", 4), ("sealed_test", 16)):
        families = SEALED_FAMILIES if split == "sealed_test" else TRAIN_FAMILIES
        for family_index, family in enumerate(families):
            for variant in range(per_family):
                yield split, family_index * per_family + variant, family, variant


def _case_and_witness(
    split: str,
    index: int,
    family: str,
    variant: int,
    seed: int = DEFAULT_SEED,
) -> tuple[Case, list[dict], dict]:
    """Build one case; use actual transitions to construct every opening record."""
    case_id = _opaque_id(seed, split, index)
    rng = random.Random(f"{seed}:{case_id}:{GENERATOR_VERSION}")
    start = datetime.fromisoformat("2027-02-01T08:00:00-05:00") + timedelta(
        days=rng.randrange(250), hours=rng.randrange(4), minutes=rng.choice((0, 10, 20, 30))
    )
    at = lambda minutes: start + timedelta(minutes=minutes)  # noqa: E731
    cost = rng.randrange(450_000, 2_500_000, 100)
    residual = rng.randrange(50_000, min(350_000, cost), 100)
    credit = cost - residual
    seller_cap = credit + rng.randrange(0, 100_000, 100)
    buyer_delay, seller_delay = rng.randrange(3, 9), rng.randrange(10, 21)
    quote_delay, reply_delay = rng.randrange(15, 31), rng.randrange(8, 17)
    has_quote = family in {"quote_credit", "quote_retry_credit", "quote_authority_credit"}
    has_authority = family in {
        "authority_credit",
        "extension_authority_credit",
        "quote_authority_credit",
    }
    has_extension = family in {
        "extension_credit",
        "extension_authority_credit",
        "revised_extension_credit",
    }
    has_retry = family in {"retry_credit", "quote_retry_credit"}
    has_revised = family in {"revised_credit", "revised_extension_credit"}
    is_cancel = family == "cancellation"
    deadline = at(270 + rng.randrange(0, 120))
    extended = at(360 + rng.randrange(30, 100)) if has_extension else None
    seller_available = at(100 + rng.randrange(0, 25)) if has_retry else None
    extension_delays = {"buyer": 2, "seller": 4} if has_extension else {}
    if has_extension:
        if has_revised:
            # Prior execution completes at 10+seller_delay. Only 22 minutes
            # remain: another credit needs 10+seller_delay+5, exceeding 22.
            deadline = at(10 + seller_delay + 22)
        else:
            deadline = at(45 + rng.randrange(0, 10))
            seller_delay = rng.randrange(70, 100)
    if family == "boundary_credit":
        deadline = at(10 + seller_delay)
    if is_cancel:
        seller_cap = max(0, credit - rng.randrange(50_000, 150_000, 100))
    expiry = extended or deadline
    evidence = ["document-002", "document-004"]
    if family == "quote_retry_credit":
        # The opening proposal is backed by the inspection, with one signature
        # already missing. A quote is independently necessary to establish the
        # eligible residual before proceeding; it need not alter this proposal.
        evidence = ["document-002"]
    if variant % 3 == 1:
        evidence.append("document-005")
    permissions = ["proceed"] + ([] if has_authority else ["credit"])
    if has_extension:
        permissions.append("extension")
    authority = {
        "id": "authority-001",
        "effective_at": start.isoformat(),
        "valid_through": expiry.isoformat(),
        "permissions": permissions,
        "minimum_credit_cents": 0,
        "maximum_credit_cents": credit + 200_000,
        "maximum_residual_cents": residual,
        "extension_limit": extended.isoformat() if extended else None,
        "source_id": "message-001",
    }
    grants = []
    if has_authority or is_cancel:
        grants = [
            {
                **authority,
                "id": "authority-002",
                "permissions": ["credit", "proceed"]
                + (["extension"] if has_extension else [])
                + (["cancel"] if is_cancel else []),
            }
        ]
    names = ("Morgan", "Kai", "Ari", "Sasha", "Robin", "Casey", "Drew", "Quinn")
    buyer, seller = f"{rng.choice(names)} Arden", f"{rng.choice(names)} Vale"
    condition = rng.choice(
        (
            "roof membrane",
            "main sewer",
            "heating equipment",
            "foundation drainage",
            "supply plumbing",
            "electrical service",
        )
    )
    address = (
        f"{rng.randrange(100, 9999)} {rng.choice(('Alder', 'Willow', 'Orchard', 'Tamarack'))} Lane"
    )
    quote = {
        "id": "document-004",
        "title": f"Fixed repair quote: {condition}",
        "kind": "quote",
        "issued_at": start.isoformat(),
        "eligible_cost_cents": cost,
        "body": f"The eligible {condition} repair costs {cost} cents (${cost / 100:,.2f}). "
        "Decorative upgrades are excluded. This quote does not create a credit.",
    }
    documents = [
        {
            "id": "document-001",
            "title": "Inspection contingency agreement",
            "kind": "contract",
            "issued_at": start.isoformat(),
            "body": f"Response deadline: {deadline.isoformat()}. Each exact amendment revision "
            "requires buyer and seller signatures. Only an executed extension changes "
            "the deadline. An authorized disposition must be committed in time.",
        },
        {
            "id": "document-002",
            "title": f"Inspection referral: {condition}",
            "kind": "inspection",
            "issued_at": start.isoformat(),
            "body": f"A material defect affects the {condition}. Specialist referral is authorized "
            "through this inspection. Use the fixed eligible-cost quote; cosmetic work "
            "is outside the inspection contingency.",
        },
        {
            "id": "document-003",
            "title": "Coordination checklist",
            "kind": "other",
            "issued_at": start.isoformat(),
            "body": "An administrator marked this transaction ready. This label is neither a "
            "signature nor a disposition. Decorative upgrades listed here are ineligible.",
        },
    ]
    if not has_quote:
        documents.append(quote)
    if "document-005" in evidence:
        documents.append(
            {
                "id": "document-005",
                "title": "Scope eligibility confirmation",
                "kind": "other",
                "issued_at": start.isoformat(),
                "body": f"The {condition} work is eligible. Attach this confirmation along with "
                "the inspection and fixed quote to credit terms.",
            }
        )
    buyer_text = (
        f"Current permissions: {', '.join(permissions)}. Credit range is 0 through "
        f"{credit + 200_000} cents. My maximum eligible residual is {residual} cents. "
        f"This authority is valid through {expiry.isoformat()}. "
    )
    if has_authority:
        buyer_text += (
            "I have not authorized credit yet. On a credit-authorization request I will "
            f"grant credit and proceed, keeping the same monetary bounds, {reply_delay} "
            "minutes after request completion. "
        )
    if is_cancel:
        buyer_text += (
            "If the seller's stated cap cannot meet my residual limit, request cancellation "
            f"authority. My cancellation grant arrives {reply_delay} minutes after the "
            "request completes. Existing authority does not permit cancellation. "
        )
    if extended:
        buyer_text += f"I permit an extension no later than {extended.isoformat()}. "
    seller_text = (
        f"Maximum seller credit: {seller_cap} cents. Every credit revision must include "
        f"{', '.join(evidence)}. Credit signatures arrive buyer {buyer_delay} minutes and "
        f"seller {seller_delay} minutes after routing completes. Negotiation replies take "
        f"{reply_delay} minutes and do not execute terms. "
    )
    if has_quote:
        seller_text += (
            f"The fixed specialist quote is not yet available. Request it with document-002; "
            f"it arrives {quote_delay} minutes after request completion. "
        )
    if seller_available:
        seller_text += (
            f"Seller is unavailable until {seller_available.isoformat()}. An earlier "
            "signature response will be missing, not pending; reroute that same revision "
            "after availability to request the missing signature. "
        )
    if extended:
        seller_text += (
            f"Seller accepts an extension through {extended.isoformat()}. Extension "
            "signatures take buyer 2 minutes and seller 4 minutes after routing completes. "
        )

    stage = "coordination"
    opening = "The transaction is open; inspect its public records and coordinate the response."
    prelude: list[dict] = []
    if family == "credit_execution" and variant % 4 == 0:
        stage = "commit_ready"
        prelude = [
            _draft(credit, evidence),
            _route("revision-001"),
            _wait(at(10 + max(buyer_delay, seller_delay))),
        ]
        opening = "A prior coordinator obtained an executed credit; the disposition is still open."
    elif family == "credit_execution" and variant % 4 == 1:
        stage = "route_ready"
        prelude = [_draft(credit, evidence)]
        opening = "A prior coordinator drafted credit revision-001; it has not been routed."
    elif family in {"retry_credit", "quote_retry_credit"}:
        prelude = [
            _draft(credit, evidence),
            _route("revision-001"),
            _wait(at(10 + max(buyer_delay, seller_delay))),
        ]
        opening = "The opening envelope has a buyer signature and a missing seller response."
        if has_quote:
            opening += (
                " The proposed credit does not establish repair cost; obtain the fixed quote."
            )
    elif has_revised:
        prelude = [
            _draft(max(0, credit - 80_000), evidence),
            _route("revision-001"),
            _wait(at(10 + max(buyer_delay, seller_delay))),
        ]
        opening = (
            "An older credit revision is executed. Recheck whether it meets the buyer's "
            "residual limit. A replacement credit must obtain its own signatures; credits "
            "replace one another rather than add."
        )
    case = Case.model_validate(
        {
            "id": case_id,
            "title": f"{address} · {condition}",
            "split": "evaluation" if split == "sealed_test" else "development",
            "brief": f"Coordinate {buyer}'s fictional inspection response for {address}. "
            "Resolve the eligible defect within buyer authority, commit an authorized "
            "disposition before the deadline, and finish with evidence-supported claims. "
            + opening,
            "property": f"{address}, Ashford, New York (synthetic)",
            "buyer": buyer,
            "seller": seller,
            "start_at": start.isoformat(),
            "deadline": deadline.isoformat(),
            "public_policy": load_case("case-01").public_policy
            + " Opening records, if present, were produced through the same engine. "
            "Prior actions do not consume this attempt's action budget.",
            "documents": documents,
            "messages": [
                {
                    "id": "message-001",
                    "sender": "buyer",
                    "body": buyer_text,
                    "at": start.isoformat(),
                },
                {
                    "id": "message-002",
                    "sender": "seller",
                    "body": seller_text,
                    "at": start.isoformat(),
                },
            ],
            "authorities": [authority],
            "counterpart": {
                "specialist_document": quote if has_quote else None,
                "specialist_latency_minutes": quote_delay,
                "required_evidence_ids": evidence,
                "seller_max_credit_cents": seller_cap,
                "seller_reply_minutes": reply_delay,
                "buyer_reply_minutes": reply_delay,
                "buyer_grants": grants,
                "signature_minutes": {"buyer": buyer_delay, "seller": seller_delay},
                "extension_signature_minutes": extension_delays,
                "seller_available_at": seller_available.isoformat() if seller_available else None,
                "maximum_extension": extended.isoformat() if extended else None,
            },
            "acceptable_dispositions": ["cancel" if is_cancel else "proceed"],
            "action_limit": 256,
            "prelude": prelude,
        }
    )
    state = reset(case)
    witness: list[dict] = []

    def act(action: dict) -> None:
        result = step(state, action)
        if not result.accepted:
            raise ValueError(f"{case_id} {family} witness rejected {action}: {result.reason}")
        witness.append(action)

    def receive_pending() -> None:
        if state.pending_events:
            act(_wait(max(event.at for event in state.pending_events)))

    def route_new_credit() -> None:
        act(_draft(credit, evidence))
        act(_route(next(reversed(state.revisions))))

    # Every demonstration obtains the public facts it uses. Private targets and
    # future events never appear in tool arguments or in model observations.
    for resource, ids in (
        ("document", [d.id for d in case.documents]),
        ("message", [m.id for m in case.messages]),
    ):
        for resource_id in ids:
            act({"type": "read", "resource": resource, "resource_id": resource_id})
    if has_extension:
        act(
            {
                "type": "draft_amendment",
                "terms": {"type": "extension", "deadline": extended.isoformat()},
                "evidence_ids": ["document-001"],
            }
        )
        act(_route(next(reversed(state.revisions))))
        receive_pending()
    if has_quote:
        act(
            {
                "type": "request",
                "request": {"intent": "specialist_evidence", "document_id": "document-002"},
            }
        )
        receive_pending()
        act({"type": "read", "resource": "document", "resource_id": "document-004"})
    if has_authority or is_cancel:
        act(
            {
                "type": "request",
                "request": {
                    "intent": "buyer_authorization",
                    "permission": "cancel" if is_cancel else "credit",
                },
            }
        )
        receive_pending()
    if not is_cancel:
        if stage == "route_ready":
            act(_route("revision-001"))
        elif stage != "commit_ready" and family not in {"retry_credit", "quote_retry_credit"}:
            route_new_credit()
        if family == "boundary_credit":
            act(_wait(state.deadline - timedelta(minutes=5)))
        else:
            receive_pending()
        if has_retry:
            if state.now < seller_available:
                act(_wait(seller_available))
            act(_route("revision-001"))
            receive_pending()
    disposition = "cancel" if is_cancel else "proceed"
    act(
        {
            "type": "set_disposition",
            "disposition": disposition,
            "reason": "Public cost evidence, current authority, and execution records support this disposition.",
        }
    )
    act(
        {
            "type": "finish",
            "claims": {
                "disposition": disposition,
                "effective_credit_cents": effective_credit(state),
                "executed_revision_ids": list(state.executed_at),
                "deadline": state.deadline.isoformat(),
            },
            "report": "Recorded the authorized disposition and verified the effective terms against transaction records.",
        }
    )
    result = evaluate(state)
    if not result.success or not all(item.passed for item in result.diagnostics):
        raise ValueError(f"{case_id} witness failed: {result.model_dump()}")
    metadata = {
        "id": case_id,
        "split": split,
        "family": family,
        "stage": stage,
        "dependency_edges": [list(edge) for edge in DEPENDENCIES[family]],
        "fingerprint": _fingerprint(case.model_dump(mode="json")),
        "public_initial_fingerprint": _fingerprint(observe(reset(case))),
        "validation": {
            "kind": "engine_executed_private_witness",
            "reward": result.reward,
            "actions": len(witness),
            "invalid_attempts": result.invalid_attempts,
            "all_predicates_passed": True,
            "witness_fingerprint": _fingerprint(witness),
        },
    }
    return case, witness, metadata


def generate_curriculum(
    directory: str | Path = DEFAULT_LEARNING_DIR,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Create 352 deterministic fixtures, validating all before writing any files."""
    prepared = [_case_and_witness(*spec, seed=seed) for spec in _specs()]
    old = json.loads((ROOT / "cases" / "generated" / "manifest.json").read_text())
    if {m["id"] for _, _, m in prepared} & {task["id"] for task in old["tasks"]}:
        raise ValueError("Learning and development case IDs overlap")
    if {m["public_initial_fingerprint"] for _, _, m in prepared} & {
        task["public_initial_fingerprint"] for task in old["tasks"]
    }:
        raise ValueError("Learning cases overlap the inspected development suite")
    manifest = {
        "schema_version": 1,
        "suite_id": "dealroom-learning-v1",
        "generator": "dealroom.curriculum",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "task_count": len(prepared),
        "split_counts": dict(SPLIT_COUNTS),
        "split_field": "manifest.tasks[].split",
        "training_families": list(TRAIN_FAMILIES),
        "sealed_families": list(SEALED_FAMILIES),
        "heldout_policy": "sealed_test cases and witnesses must not be used for training or checkpoint selection; evaluate only after the workflow and checkpoints are frozen",
        "selection_split": "validation",
        "training_split": "train",
        "provenance": "Deterministic synthetic fixtures. Engine-executed private witnesses establish satisfiability, not model performance. Sealed cases combine dependencies absent from both training and validation. Model inputs must use public observations only.",
        "previous_development_suite_fingerprint": old["fingerprint"],
        "fingerprint": _fingerprint([metadata for _, _, metadata in prepared]),
        "tasks": [metadata for _, _, metadata in prepared],
    }
    output = Path(directory)
    (output / "private").mkdir(parents=True, exist_ok=True)
    for case, witness, metadata in prepared:
        (output / f"{case.id}.json").write_text(case.model_dump_json(indent=2) + "\n")
        (output / "private" / f"{case.id}.json").write_text(
            json.dumps(
                {
                    "case_id": case.id,
                    "split": metadata["split"],
                    "use": "training_demonstration"
                    if metadata["split"] == "train"
                    else "fixture_validation_only",
                    "witness": witness,
                },
                indent=2,
            )
            + "\n"
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(directory: str | Path = DEFAULT_LEARNING_DIR) -> dict:
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("suite_id") != "dealroom-learning-v1":
        raise ValueError("Unsupported learning manifest")
    tasks = manifest.get("tasks", [])
    if Counter(task["split"] for task in tasks) != Counter(SPLIT_COUNTS):
        raise ValueError("Learning split counts changed")
    if len({task["id"] for task in tasks}) != sum(SPLIT_COUNTS.values()):
        raise ValueError("Duplicate or missing learning case IDs")
    if _fingerprint(tasks) != manifest.get("fingerprint"):
        raise ValueError("Learning manifest fingerprint mismatch")
    return manifest


def case_entry(case_id: str, directory: str | Path = DEFAULT_LEARNING_DIR) -> dict:
    entry = next(
        (task for task in load_manifest(directory)["tasks"] if task["id"] == case_id), None
    )
    if entry is None:
        raise ValueError(f"Unknown learning case: {case_id}")
    return entry


def learning_case_ids(
    split: LearningSplit = "train",
    directory: str | Path = DEFAULT_LEARNING_DIR,
) -> list[str]:
    if split not in SPLIT_COUNTS:
        raise ValueError(f"Unknown learning split: {split}")
    return [task["id"] for task in load_manifest(directory)["tasks"] if task["split"] == split]


training_case_ids = learning_case_ids


def load_learning_case(
    case_id: str,
    directory: str | Path = DEFAULT_LEARNING_DIR,
    *,
    case_dir: str | Path | None = None,
) -> Case:
    directory = case_dir if case_dir is not None else directory
    entry = case_entry(case_id, directory)
    case = Case.model_validate_json((Path(directory) / f"{case_id}.json").read_text())
    if case.id != case_id or _fingerprint(case.model_dump(mode="json")) != entry["fingerprint"]:
        raise ValueError(f"Learning fixture fingerprint mismatch: {case_id}")
    return case


def training_witness(case_id: str, directory: str | Path = DEFAULT_LEARNING_DIR) -> list[dict]:
    """Only training demonstrations are exportable; selection/test answers are refused."""
    entry = case_entry(case_id, directory)
    if entry["split"] != "train":
        raise PermissionError(f"{entry['split']} witnesses are for fixture validation only")
    load_learning_case(case_id, directory)
    artifact = json.loads((Path(directory) / "private" / f"{case_id}.json").read_text())
    witness = artifact["witness"]
    if (
        artifact["case_id"] != case_id
        or artifact["split"] != "train"
        or (_fingerprint(witness) != entry["validation"]["witness_fingerprint"])
    ):
        raise ValueError(f"Training witness fingerprint mismatch: {case_id}")
    return witness


def verify_curriculum(directory: str | Path = DEFAULT_LEARNING_DIR) -> None:
    """Offline fixture validation; never returns selection or sealed witness actions."""
    manifest = load_manifest(directory)
    expected = [_case_and_witness(*spec, seed=manifest["seed"]) for spec in _specs()]
    if [metadata for _, _, metadata in expected] != manifest["tasks"]:
        raise ValueError("Learning manifest disagrees with deterministic generator")
    for expected_case, witness, metadata in expected:
        case = load_learning_case(expected_case.id, directory)
        if case != expected_case:
            raise ValueError(f"Learning case changed: {case.id}")
        artifact = json.loads((Path(directory) / "private" / f"{case.id}.json").read_text())
        if artifact["witness"] != witness or artifact["split"] != metadata["split"]:
            raise ValueError(f"Private validation witness changed: {case.id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_LEARNING_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_curriculum(args.directory)
        print("Learning curriculum verified")
    else:
        manifest = generate_curriculum(args.directory, args.seed)
        print(
            json.dumps(
                {
                    key: manifest[key]
                    for key in ("suite_id", "task_count", "split_counts", "fingerprint")
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
