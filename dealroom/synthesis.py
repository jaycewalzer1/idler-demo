"""Reproducible task generation and private fixture validation, never a model solver.

The eight scenario families change the required coordination behavior, not just names.
Witness actions below validate satisfiability and are never part of an observation or
model prompt. The manifest contains only their validation result and fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dealroom.domain import Case, load_case, observe, reset, step
from dealroom.score import evaluate

DEFAULT_SEED = 20260927
DEFAULT_CASE_DIR = Path(__file__).resolve().parent.parent / "cases" / "generated"
GENERATOR_VERSION = "1.0.0"
FAMILIES = (
    "quote_discovery",
    "straightforward_execution",
    "partial_signature_retry",
    "revised_credit_execution",
    "deadline_extension",
    "cancellation_authority",
    "credit_authority_refresh",
    "boundary_timing",
)
VARIANTS_PER_FAMILY = 6

_BUYERS = ("Rowan Hale", "Mira Voss", "Ellis Reed", "Tessa Quinn", "Jules Avery", "Nico Lane")
_SELLERS = ("Sage Mercer", "Ada Wynn", "Leon Vale", "Iris Finch", "Remy Cole", "Avery Wells")
_STREETS = ("Juniper Walk", "Birch Court", "Cedar Place", "Harbor Lane", "Maple Rise", "Brook Way")
_CONDITIONS = (
    ("roof flashing", "replace failed flashing and water-damaged sheathing", "exterior repainting"),
    ("electrical panel", "replace an unsafe panel and correct grounding", "decorative fixtures"),
    ("foundation drainage", "install drainage and seal active foundation leaks", "garden redesign"),
    ("heating system", "replace a cracked heat exchanger and vent connector", "smart thermostats"),
    (
        "plumbing supply",
        "replace corroded supply pipes and repair the active leak",
        "new vanity tops",
    ),
    (
        "crawlspace support",
        "replace damaged support posts and stabilize the beam",
        "finish flooring",
    ),
)


def suite_case_ids() -> list[str]:
    return [f"synth-{number:03}" for number in range(1, len(FAMILIES) * VARIANTS_PER_FAMILY + 1)]


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _credit(amount: int, evidence: list[str]) -> dict:
    return {
        "type": "draft_amendment",
        "terms": {"type": "credit", "amount_cents": amount},
        "evidence_ids": evidence.copy(),
    }


def _route(number: int) -> dict:
    return {"type": "send_for_signature", "revision_id": f"revision-{number:03}"}


def _wait(at: datetime) -> dict:
    return {"type": "wait_until", "at": at.isoformat()}


def _case_and_witness(number: int, seed: int) -> tuple[Case, list[dict]]:
    family_index, variant = divmod(number - 1, VARIANTS_PER_FAMILY)
    family = FAMILIES[family_index]
    rng = random.Random(f"{seed}:{number}:{GENERATOR_VERSION}")
    start = datetime.fromisoformat("2026-10-12T09:00:00-04:00") + timedelta(
        days=family_index + variant, hours=variant % 3, minutes=15 * (variant % 2)
    )
    stamp = start.isoformat()
    at = lambda minutes: start + timedelta(minutes=minutes)  # noqa: E731
    deadline = at(240 + variant * 25)
    cost = (70 + rng.randrange(95)) * 10_000 + 100 * number
    residual_limit = (15 + variant * 4) * 10_000
    target_credit = cost - residual_limit
    minimum_credit = max(0, target_credit - 300_000)
    maximum_credit = target_credit + 250_000
    seller_cap = target_credit + (variant % 3) * 25_000
    buyer_delay = 5 + variant * 2
    seller_delay = 15 + variant * 5
    quote_delay = 20 + variant * 7
    reply_delay = 10 + variant * 3
    evidence = ["document-002", "document-004"]
    if variant % 2:
        evidence.append("document-005")
    buyer, seller = _BUYERS[variant], _SELLERS[(variant + family_index) % 6]
    address = f"{110 + number * 3} {_STREETS[variant]}"
    condition, scope, excluded = _CONDITIONS[(variant + family_index) % 6]
    permissions = ["credit", "proceed"]
    extension_deadline = None
    extension_minutes = {}
    seller_available = None
    grants = []
    prelude = []
    witness: list[dict] = [{"type": "read", "resource": "summary"}]
    note = ""
    buyer_note = ""
    disposition = "proceed"
    executed_ids = ["revision-001"]

    if family == "deadline_extension":
        deadline = at(45 + variant * 5)
        extension_deadline = at(240 + variant * 20)
        permissions.append("extension")
        extension_minutes = {"buyer": 5 + variant, "seller": 10 + variant * 2}
        seller_delay = 90 + variant * 10
        note = (
            f"The current response deadline is {deadline.isoformat()}. Standard credit signatures "
            f"take longer than the remaining window. The buyer and seller permit an extension "
            f"through {extension_deadline.isoformat()}, which must itself be executed in time."
        )
        extension_action = {
            "type": "draft_amendment",
            "terms": {"type": "extension", "deadline": extension_deadline.isoformat()},
            "evidence_ids": ["document-001"],
        }
        extension_ready = 10 + max(extension_minutes.values())
        witness += [extension_action, _route(1), _wait(at(extension_ready))]
        witness += [
            _credit(target_credit, evidence),
            _route(2),
            _wait(at(extension_ready + 10 + seller_delay)),
        ]
        executed_ids = ["revision-001", "revision-002"]
    elif family == "boundary_timing":
        # The final seller signature lands at the deadline. Waiting to that instant
        # expires the transaction; a disposition begun five minutes before can commit.
        deadline = at(10 + seller_delay)
        note = (
            "This is an urgent handoff with no extension authority. Read-only inspection costs "
            "no time; each coordination action costs five minutes. Account for the seller's "
            "signature latency and the time needed to complete a disposition."
        )
        witness += [
            _credit(target_credit, evidence),
            _route(1),
            _wait(deadline - timedelta(minutes=5)),
        ]
    elif family == "partial_signature_retry":
        unavailable_until = 90 + variant * 10
        seller_available = at(unavailable_until)
        prelude = [
            _credit(target_credit, evidence),
            _route(1),
            _wait(at(10 + seller_delay)),
        ]
        note = (
            "Opening handoff: revision-001 has a buyer signature but the seller's first "
            "response reported unavailability. Its missing signature does not arrive "
            "automatically; inspect the envelope and retry after seller availability."
        )
        witness += [
            {"type": "read", "resource": "envelope", "resource_id": "envelope-001"},
            _wait(seller_available),
            _route(1),
            _wait(at(unavailable_until + 5 + seller_delay)),
        ]
    elif family == "revised_credit_execution":
        previous_credit = target_credit - (8 + variant) * 10_000
        prelude = [
            _credit(previous_credit, evidence),
            _route(1),
            _wait(at(10 + seller_delay)),
            _credit(target_credit, evidence),
        ]
        note = (
            f"Opening handoff: revision-001 is executed at {previous_credit} cents. "
            f"Revision-002 changes the credit to {target_credit} cents but is still unsigned. "
            "Determine which terms currently operate and obtain any additional execution "
            "needed to meet the buyer's residual limit. Credits replace rather than add."
        )
        witness += [
            {"type": "read", "resource": "amendment", "resource_id": "revision-002"},
            _route(2),
            _wait(at(20 + 2 * seller_delay)),
        ]
        executed_ids = ["revision-001", "revision-002"]
    elif family == "quote_discovery":
        note = (
            "The initial inspection is provisional; request the specialist's fixed quote "
            "using document-002 before proposing evidence-backed terms."
        )
        witness += [
            {
                "type": "request",
                "request": {"intent": "specialist_evidence", "document_id": "document-002"},
            },
            _wait(at(5 + quote_delay)),
            _credit(target_credit, evidence),
            _route(1),
            _wait(at(15 + quote_delay + seller_delay)),
        ]
    elif family == "cancellation_authority":
        seller_cap = target_credit - (8 + variant * 2) * 10_000
        minimum_credit = max(0, seller_cap - 100_000)
        disposition = "cancel"
        executed_ids = []
        note = (
            "Compare the seller's disclosed maximum concession with the buyer's residual "
            "limit. If no authorized credit can resolve the gap, request the buyer's "
            "explicit cancellation permission before recording a cancellation."
        )
        buyer_note = (
            "Cancellation is not authorized yet. If the documented concession cannot meet "
            "my residual limit, request cancellation authorization; a scoped response will "
            f"arrive {reply_delay} minutes after your request completes."
        )
        witness += [
            {
                "type": "request",
                "request": {"intent": "buyer_authorization", "permission": "cancel"},
            },
            _wait(at(5 + reply_delay)),
        ]
    elif family == "credit_authority_refresh":
        if variant % 2:
            maximum_credit = target_credit - 50_000
            buyer_note = (
                f"My current credit authorization ends at {maximum_credit} cents and cannot "
                "satisfy my residual limit. Request refreshed credit authorization before "
                "drafting stronger terms."
            )
        else:
            permissions = ["proceed"]
            buyer_note = (
                "I have authorized proceeding only after the constraints are met, but have "
                "not yet authorized credit coordination. Request credit authorization first."
            )
        buyer_note += (
            f" My refreshed scope permits credit from {minimum_credit} through "
            f"{target_credit + 250_000} cents and proceed; it arrives {reply_delay} minutes "
            "after the request completes."
        )
        note = "Resolve the authority gap before taking actions outside the current buyer scope."
        witness += [
            {
                "type": "request",
                "request": {"intent": "buyer_authorization", "permission": "credit"},
            },
            _wait(at(5 + reply_delay)),
            _credit(target_credit, evidence),
            _route(1),
            _wait(at(15 + reply_delay + seller_delay)),
        ]
    else:
        note = (
            "The fixed specialist quote and required supporting records are already public. "
            "Complete authorized terms with the required evidence and exact signatures."
        )
        witness += [
            _credit(target_credit, evidence),
            _route(1),
            _wait(at(10 + seller_delay)),
        ]

    authority_expiry = extension_deadline or deadline
    authority = {
        "id": "authority-001",
        "effective_at": stamp,
        "valid_through": authority_expiry.isoformat(),
        "permissions": permissions,
        "minimum_credit_cents": minimum_credit,
        "maximum_credit_cents": maximum_credit,
        "maximum_residual_cents": residual_limit,
        "extension_limit": extension_deadline.isoformat() if extension_deadline else None,
        "source_id": "message-001",
    }
    if family in {"cancellation_authority", "credit_authority_refresh"}:
        grants = [
            {
                **authority,
                "id": "authority-002",
                "permissions": ["credit", "proceed"]
                + (["cancel"] if disposition == "cancel" else []),
                "maximum_credit_cents": target_credit + 250_000,
                # The standing buyer instruction is the evidence source for the grant;
                # the runtime response itself provides its actual effective timestamp.
                "source_id": "message-001",
            }
        ]

    documents = [
        {
            "id": "document-001",
            "title": "Purchase agreement and inspection contingency",
            "body": (
                f"Fictional transaction: {buyer} buys {address} from {seller}. Response "
                f"deadline: {deadline.isoformat()}. Both buyer and seller must sign an exact "
                "amendment revision for any concession or extension to become effective. "
                "Only a fully executed extension moves the deadline."
            ),
            "kind": "contract",
            "issued_at": stamp,
        },
        {
            "id": "document-002",
            "title": f"Inspection: {condition}",
            "body": (
                f"The inspection identified a material defect in the {condition}. The "
                f"eligible work is to {scope}. Obtain or use the fixed specialist quote; "
                f"{excluded} is excluded. This inspection is the referral for a specialist request."
            ),
            "kind": "inspection",
            "issued_at": stamp,
        },
        {
            "id": "document-003",
            "title": "Administrative coordination board",
            "body": (
                f"The coordination board says '{('agreed in principle', 'ready to close', 'seller approved')[variant % 3]}'. "
                f"A separate wishlist prices {excluded} at {80_000 + 15_000 * variant} cents. "
                "This board is not an executed amendment, a signature record, or eligible cost evidence."
            ),
            "kind": "other",
            "issued_at": stamp,
        },
    ]
    quote = {
        "id": "document-004",
        "title": f"Fixed specialist quote: {condition}",
        "body": (
            f"Synthetic specialist scope: {scope}. Fixed eligible repair cost is {cost} "
            f"cents (${cost / 100:,.2f}). The price excludes {excluded}. This cost evidence "
            "does not itself create a seller credit."
        ),
        "kind": "quote",
        "issued_at": stamp,
        "eligible_cost_cents": cost,
    }
    if family != "quote_discovery":
        documents.append(quote)
    if variant % 2:
        documents.append(
            {
                "id": "document-005",
                "title": "Contractor scope eligibility confirmation",
                "body": (
                    f"The contractor confirms the {condition} work is within the contingency "
                    f"scope and excludes {excluded}. The seller requires this confirmation "
                    "alongside the inspection and quote on credit proposals."
                ),
                "kind": "other",
                "issued_at": stamp,
            }
        )
    seller_text = (
        f"Seller will sign credit up to {seller_cap} cents if the revision includes "
        f"all evidence IDs: {', '.join(evidence)}. Negotiation replies take {reply_delay} "
        f"minutes after request completion. Credit signature latencies after routing "
        f"completion are buyer {buyer_delay} minutes and seller {seller_delay} minutes. "
        "An informal acceptance does not execute terms. "
    )
    if family == "quote_discovery":
        seller_text += (
            f"The specialist response arrives {quote_delay} minutes after request completion. "
        )
    if seller_available:
        seller_text += (
            f"Seller is unavailable to sign until {seller_available.isoformat()}; earlier "
            "requests return a missing-signature response. Reroute the same revision once available. "
        )
    if extension_deadline:
        seller_text += (
            f"Seller accepts an extension no later than {extension_deadline.isoformat()}; "
            f"extension signatures take buyer {extension_minutes['buyer']} minutes and seller "
            f"{extension_minutes['seller']} minutes after routing. "
        )
    buyer_text = (
        f"Current permissions: {', '.join(permissions)}. Credit scope: {minimum_credit} "
        f"through {maximum_credit} cents. Proceed only with eligible residual cost at most "
        f"{residual_limit} cents. This scope lasts through {authority_expiry.isoformat()}. "
        + buyer_note
    )
    if extension_deadline:
        buyer_text += f" Extension authority is capped at {extension_deadline.isoformat()}."

    policy = load_case("case-01").public_policy
    if prelude:
        policy += (
            " Opening transaction records were created by prior coordination through the same "
            "engine; those records and elapsed time remain valid, but their actions do not "
            "consume this attempt's action budget."
        )
    case = Case.model_validate(
        {
            "id": f"synth-{number:03}",
            "title": f"{address} · {condition}",
            "split": "development" if variant < 2 else "evaluation",
            "brief": (
                f"Coordinate buyer {buyer}'s fictional inspection response for {address}. "
                "Resolve the documented defect under current authority, secure exact-revision "
                "execution when needed, and record an authorized disposition before the "
                "deadline. Finish with claims supported by the transaction records. " + note
            ),
            "property": f"{address}, Bellmere, New York (synthetic)",
            "buyer": buyer,
            "seller": seller,
            "start_at": stamp,
            "deadline": deadline.isoformat(),
            "public_policy": policy,
            "documents": documents,
            "messages": [
                {"id": "message-001", "sender": "buyer", "body": buyer_text, "at": stamp},
                {"id": "message-002", "sender": "seller", "body": seller_text, "at": stamp},
            ],
            "authorities": [authority],
            "counterpart": {
                "specialist_document": quote if family == "quote_discovery" else None,
                "specialist_latency_minutes": quote_delay,
                "required_evidence_ids": evidence,
                "seller_max_credit_cents": seller_cap,
                "seller_reply_minutes": reply_delay,
                "buyer_reply_minutes": reply_delay,
                "buyer_grants": grants,
                "signature_minutes": {"buyer": buyer_delay, "seller": seller_delay},
                "extension_signature_minutes": extension_minutes,
                "seller_available_at": seller_available.isoformat() if seller_available else None,
                "maximum_extension": extension_deadline.isoformat() if extension_deadline else None,
            },
            "acceptable_dispositions": [disposition],
            "action_limit": 64,
            "prelude": prelude,
        }
    )
    witness += [
        {
            "type": "set_disposition",
            "disposition": disposition,
            "reason": "Fixture validation: the underlying records satisfy current buyer authority.",
        },
        {
            "type": "finish",
            "claims": {
                "disposition": disposition,
                "effective_credit_cents": 0 if disposition == "cancel" else target_credit,
                "executed_revision_ids": executed_ids,
                "deadline": authority_expiry.isoformat(),
            },
            "report": "Private deterministic fixture witness; not model performance.",
        },
    ]
    return case, witness


def validation_witness(case_id: str, seed: int = DEFAULT_SEED) -> list[dict]:
    """Private validation helper; never include this script in a model dataset."""
    if case_id not in suite_case_ids():
        raise ValueError(f"Unknown synthetic case: {case_id}")
    return _case_and_witness(int(case_id.removeprefix("synth-")), seed)[1]


def generate_suite(
    output_dir: str | Path = DEFAULT_CASE_DIR, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    """Validate every task with the unchanged engine, then write a deterministic suite."""
    output = Path(output_dir)
    prepared: list[tuple[Case, dict]] = []
    for number, case_id in enumerate(suite_case_ids(), start=1):
        case, witness = _case_and_witness(number, seed)
        state = reset(case)
        initial = observe(state)
        for action in witness:
            result = step(state, action)
            if not result.accepted:
                raise ValueError(f"{case_id} witness rejected {action}: {result.reason}")
        score = evaluate(state)
        if not score.success:
            raise ValueError(f"{case_id} witness failed: {score.model_dump()}")
        prepared.append(
            (
                case,
                {
                    "id": case_id,
                    "title": case.title,
                    "split": case.split,
                    "family": FAMILIES[(number - 1) // VARIANTS_PER_FAMILY],
                    "variant": (number - 1) % VARIANTS_PER_FAMILY + 1,
                    "fingerprint": _fingerprint(case.model_dump(mode="json")),
                    "public_initial_fingerprint": _fingerprint(initial),
                    "validation": {
                        "kind": "engine_executed_private_witness",
                        "reward": score.reward,
                        "attempts": score.attempts,
                        "invalid_attempts": score.invalid_attempts,
                        "all_predicates_passed": all(d.passed for d in score.diagnostics),
                        "witness_fingerprint": _fingerprint(witness),
                    },
                },
            )
        )
    manifest = {
        "schema_version": 1,
        "suite_id": "dealroom-synthetic-48-v1",
        "generator": "dealroom.synthesis",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "task_count": len(prepared),
        "families": list(FAMILIES),
        "variants_per_family": VARIANTS_PER_FAMILY,
        "provenance": (
            "Deterministic parameterized synthetic generation; no scraped transactions or "
            "model-generated facts. Eight behavioral families use the unchanged DealRoom "
            "transition engine and verifier. Private witnesses validate satisfiability and "
            "are never included in model prompts. These validation scores are not model results."
        ),
        "fingerprint": _fingerprint([entry[1]["fingerprint"] for entry in prepared]),
        "tasks": [entry[1] for entry in prepared],
    }
    output.mkdir(parents=True, exist_ok=True)
    for case, _ in prepared:
        (output / f"{case.id}.json").write_text(case.model_dump_json(indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def verify_manifest(manifest: dict, case_dir: str | Path = DEFAULT_CASE_DIR) -> None:
    """Reject changed fixtures or mislabeled provenance before starting/resuming a run.

    Recompute from the declared generator seed as well as the on-disk files, so a
    stale or edited manifest cannot silently mix task versions in one benchmark.
    """
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("suite_id") != "dealroom-synthetic-48-v1"
        or manifest.get("generator") != "dealroom.synthesis"
        or manifest.get("generator_version") != GENERATOR_VERSION
        or type(manifest.get("seed")) is not int
        or manifest.get("task_count") != len(suite_case_ids())
        or manifest.get("families") != list(FAMILIES)
        or manifest.get("variants_per_family") != VARIANTS_PER_FAMILY
    ):
        raise ValueError("Synthetic manifest has unsupported or inconsistent provenance.")
    tasks = manifest.get("tasks", [])
    if (
        not isinstance(tasks, list)
        or not all(isinstance(task, dict) for task in tasks)
        or [task.get("id") for task in tasks] != suite_case_ids()
    ):
        raise ValueError("Synthetic manifest must contain the 48 unique ordered task IDs.")
    fingerprints = []
    for number, entry in enumerate(tasks, start=1):
        expected, witness = _case_and_witness(number, manifest["seed"])
        actual = load_case(Path(case_dir) / f"{expected.id}.json")
        fingerprint = _fingerprint(actual.model_dump(mode="json"))
        if fingerprint != _fingerprint(expected.model_dump(mode="json")):
            raise ValueError(f"Synthetic fixture {expected.id} differs from its declared seed.")
        expected_fields = {
            "id": expected.id,
            "title": expected.title,
            "split": expected.split,
            "family": FAMILIES[(number - 1) // VARIANTS_PER_FAMILY],
            "variant": (number - 1) % VARIANTS_PER_FAMILY + 1,
            "fingerprint": fingerprint,
            "public_initial_fingerprint": _fingerprint(observe(reset(actual))),
            "validation": {
                "kind": "engine_executed_private_witness",
                "reward": 1,
                "attempts": len(witness),
                "invalid_attempts": 0,
                "all_predicates_passed": True,
                "witness_fingerprint": _fingerprint(witness),
            },
        }
        if any(entry.get(key) != value for key, value in expected_fields.items()):
            raise ValueError(f"Synthetic manifest metadata does not match {expected.id}.")
        fingerprints.append(fingerprint)
    if manifest.get("fingerprint") != _fingerprint(fingerprints):
        raise ValueError("Synthetic suite fingerprint does not match its task fingerprints.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = generate_suite(args.output_dir, args.seed)
    print(f"Validated {manifest['task_count']} tasks in {args.output_dir}")
    print(f"Suite fingerprint: {manifest['fingerprint']}")


if __name__ == "__main__":
    main()
