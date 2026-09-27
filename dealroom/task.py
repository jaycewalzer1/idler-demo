"""Inspect adapter. All business effects go through domain.step.

An engine and a lock are created inside each sample's solver invocation. Only
public observations cross the tool boundary; compact actions/events are saved
in Inspect metadata for deterministic scorer and reviewer replay.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Annotated, Literal

from inspect_ai import Task, task
from inspect_ai.agent import AgentPrompt, AgentSubmit, as_solver, react
from inspect_ai.dataset import Sample
from inspect_ai.event import SampleLimitEvent
from inspect_ai.log import transcript
from inspect_ai.model import CompactionTrim, GenerateConfig
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import Tool, ToolDef, ToolError, ToolParam, tool
from inspect_ai.viewer import TaskSamplesColumn, TaskSamplesView, ViewerConfig
from pydantic import Field, TypeAdapter

from dealroom.domain import (
    BuyerAuthorityRequest,
    CreditTerms,
    ExtensionTerms,
    FinalClaims,
    SellerNegotiationRequest,
    SpecialistRequest,
    load_case,
    observe,
    reset,
    step,
)
from dealroom.score import evaluate

Request = Annotated[
    SpecialistRequest | BuyerAuthorityRequest | SellerNegotiationRequest,
    Field(discriminator="intent"),
]
Terms = Annotated[CreditTerms | ExtensionTerms, Field(discriminator="type")]

PROMPT = """You are the buyer's transaction coordinator in a synthetic transaction.
Use only the public case materials and the seven supplied tools. Read the public
policy and discoverable evidence. Actions and signatures have simulated timing;
plan within the contractual deadline and buyer's current authority. Execute one
tool at a time. Use finish to submit structured claims backed by public evidence.
This fictional benchmark does not provide real legal advice or real services.
"""


def typed_parameter(annotation) -> ToolParam:
    """Preserve nested discriminated unions in Inspect 0.3.271's tool schema.

    That release's automatic class conversion drops nested Annotated terms.
    Use Pydantic's schema, flatten references and express const/oneOf using the
    equivalent subset supported by Inspect's public ToolParam surface.
    """
    schema = TypeAdapter(annotation).json_schema()
    definitions = schema.get("$defs", {})

    def normalize(value):
        if isinstance(value, list):
            return [normalize(v) for v in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            return normalize(definitions[value["$ref"].split("/")[-1]])
        result = {
            ("anyOf" if k == "oneOf" else k): normalize(v)
            for k, v in value.items()
            if k not in {"$defs", "title", "discriminator", "const"}
        }
        if "const" in value:
            result["enum"] = [value["const"]]
        if "properties" in value:
            tags = [key for key, child in value["properties"].items() if "const" in child]
            result["required"] = list(dict.fromkeys(result.get("required", []) + tags))
        return result

    return ToolParam.model_validate(normalize(schema))


def case_for_record(record: dict):
    """Reconstruct the recorded episode's effective fixture without changing source cases."""
    case = load_case(record["case_id"])
    base_fingerprint = record.get("base_fixture_sha256")
    if base_fingerprint is not None and base_fingerprint != fixture_fingerprint(case):
        raise ValueError("The base fixture changed after this episode was recorded.")
    action_limit = record.get("action_limit")
    if action_limit is not None:
        if type(action_limit) is not int or action_limit < 1:
            raise ValueError("Episode action_limit must be a positive integer.")
        case = case.model_copy(update={"action_limit": action_limit})
    return case


def replay(case_id: str, actions: list[dict], action_limit: int | None = None):
    engine = reset(case_for_record({"case_id": case_id, "action_limit": action_limit}))
    for action in actions:
        step(engine, action)
    return engine


def fixture_fingerprint(case) -> str:
    return hashlib.sha256(
        json.dumps(case.model_dump(mode="json"), sort_keys=True).encode()
    ).hexdigest()


def public_index(public: dict, *, include_policy: bool = False) -> dict:
    """Keep tool receipts compact; full evidence remains available via read IDs."""
    keys = [
        "case_id",
        "now",
        "deadline",
        "business_status",
        "finished",
        "attempts",
        "action_limit",
        "budget_exhausted",
        "authority",
        "resources",
        "amendments",
        "operative_revision_id",
        "effective_credit_cents",
        "eligible_cost_cents",
        "disposition",
    ]
    if include_policy:
        keys += ["title", "brief", "property", "buyer", "seller", "policy", "required_signers"]
    return {key: public[key] for key in keys}


def sample_for(case_id: str, action_limit: int | None = None) -> Sample:
    public = observe(reset(case_for_record({"case_id": case_id, "action_limit": action_limit})))
    return Sample(
        id=case_id,
        input=json.dumps(public_index(public, include_policy=True), ensure_ascii=False),
        # No target answers, witness actions, policy internals or evaluator hints.
        metadata={"case_id": case_id},
    )


def tools_for(engine, record: dict) -> tuple[list[Tool], Tool]:
    lock = asyncio.Lock()

    def payload(value):
        # Inspect supplies nested arguments as JSON objects; domain.step owns
        # authoritative Pydantic validation, including typed invalid attempts.
        return value.model_dump(mode="json") if hasattr(value, "model_dump") else value

    async def apply(action: dict) -> str:
        # Serialize even if a provider disregards parallel_tool_calls=False.
        async with lock:
            result = step(engine, action)
            data = result.model_dump(mode="json")
            record["actions"].append(action)
            record["events"].extend(data.get("events", []))
            if action["type"] != "read" or action.get("resource", "summary") == "summary":
                data["observation"] = public_index(
                    data["observation"],
                    include_policy=action["type"] == "read",
                )
            return json.dumps(data, ensure_ascii=False)

    @tool
    def read() -> Tool:
        async def execute(
            resource: Literal[
                "summary", "document", "message", "envelope", "amendment"
            ] = "summary",
            resource_id: str | None = None,
        ) -> str:
            """Read public evidence or the resource index; consumes no simulated time.

            Args:
                resource: Public resource kind. Summary lists discoverable IDs.
                resource_id: ID of the resource, omitted for summary.
            """
            return await apply({"type": "read", "resource": resource, "resource_id": resource_id})

        return execute

    @tool
    def request() -> Tool:
        async def execute(request: Request) -> str:
            """Request specialist evidence, buyer authority, or a seller response.

            Args:
                request: Typed intent and its required evidence or proposed terms.
            """
            return await apply({"type": "request", "request": payload(request)})

        return execute

    @tool
    def draft_amendment() -> Tool:
        async def execute(terms: Terms, evidence_ids: list[str]) -> str:
            """Create an immutable credit or extension revision. Drafting does not execute it.

            Args:
                terms: Credit in integer cents, or an explicit extension deadline.
                evidence_ids: Public evidence IDs supporting these terms.
            """
            return await apply(
                {
                    "type": "draft_amendment",
                    "terms": payload(terms),
                    "evidence_ids": evidence_ids,
                }
            )

        return execute

    @tool
    def send_for_signature() -> Tool:
        async def execute(revision_id: str) -> str:
            """Route an exact revision for its stipulated signatures.

            Args:
                revision_id: Immutable amendment revision to route.
            """
            return await apply({"type": "send_for_signature", "revision_id": revision_id})

        return execute

    @tool
    def wait_until() -> Tool:
        async def execute(at: str) -> str:
            """Advance simulated time and process intervening counterpart events.

            Args:
                at: Timezone-aware ISO 8601 timestamp, e.g. 2026-10-05T12:00:00-04:00.
            """
            return await apply({"type": "wait_until", "at": at})

        return execute

    @tool
    def set_disposition() -> Tool:
        async def execute(disposition: Literal["proceed", "cancel"], reason: str) -> str:
            """Attempt a buyer-authorized disposition subject to case constraints.

            Args:
                disposition: Proceed with the transaction or cancel it.
                reason: Evidence-based reason for this action.
            """
            return await apply(
                {"type": "set_disposition", "disposition": disposition, "reason": reason}
            )

        return execute

    @tool
    def finish() -> Tool:
        async def execute(claims: dict, report: str) -> str:
            """End the attempt with structured factual claims. Has no business effects.

            Args:
                claims: Disposition, effective credit, executed revision IDs, and deadline.
                report: Concise final report grounded in public records.
            """
            receipt = await apply({"type": "finish", "claims": payload(claims), "report": report})
            data = json.loads(receipt)
            if not data["accepted"]:
                # Built-in ReAct only considers error-free finish calls submitted.
                raise ToolError(data["reason"])
            return receipt

        return execute

    request_tool = ToolDef(request(), parallel=False, max_output=0)
    request_tool.parameters.properties["request"] = typed_parameter(Request)
    request_tool.parameters.properties[
        "request"
    ].description = (
        "Typed intent with its required evidence, authority permission, or proposed terms."
    )
    draft_tool = ToolDef(draft_amendment(), parallel=False, max_output=0)
    draft_tool.parameters.properties["terms"] = typed_parameter(Terms)
    draft_tool.parameters.properties[
        "terms"
    ].description = "Immutable credit in integer cents or an explicit extension deadline."
    tools = [
        read(),
        request_tool.as_tool(),
        draft_tool.as_tool(),
        send_for_signature(),
        wait_until(),
        set_disposition(),
    ]
    finish_tool = ToolDef(finish(), parallel=False, max_output=0)
    finish_tool.parameters.properties["claims"] = typed_parameter(FinalClaims)
    finish_tool.parameters.properties[
        "claims"
    ].description = "Structured final disposition, effective credit, executed revision IDs, and optional deadline."
    return [
        ToolDef(t, parallel=False, max_output=0).as_tool() for t in tools
    ], finish_tool.as_tool()


@solver
def coordinate(
    action_limit: int | None = None,
    end_on_expiry: bool = False,
    compaction_threshold: int | None = None,
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        case_id = state.metadata["case_id"]
        base_case = load_case(case_id)
        case = case_for_record({"case_id": case_id, "action_limit": action_limit})
        engine = reset(case)
        record = {
            "case_id": case_id,
            "fixture_sha256": fixture_fingerprint(case),
            "base_fixture_sha256": fixture_fingerprint(base_case),
            "action_limit": case.action_limit,
            "end_on_expiry": end_on_expiry,
            "compaction_threshold": compaction_threshold,
            "actions": [],
            "events": [],
        }
        state.metadata["dealroom"] = record
        tools, finish = tools_for(engine, record)

        async def continue_run(_state):
            if end_on_expiry and engine.business_status == "expired":
                record["termination_reason"] = "contract_deadline_expired"
                return False
            return not getattr(engine, "budget_exhausted", False)

        agent = react(
            name="transaction_coordinator",
            prompt=AgentPrompt(
                instructions=PROMPT
                + (
                    "\nThis evaluation ends the episode immediately if the contractual deadline "
                    "expires. Complete the required coordination before expiry; no further "
                    "actions will be requested after the transaction is irreversibly expired.\n"
                    if end_on_expiry
                    else ""
                ),
                assistant_prompt=None,
                handoff_prompt=None,
                submit_prompt=None,
            ),
            tools=tools,
            submit=AgentSubmit(name="finish", tool=finish, keep_in_messages=True),
            attempts=1,
            on_continue=continue_run,
            compaction=CompactionTrim(threshold=compaction_threshold, preserve=0.5, memory=False)
            if compaction_threshold is not None
            else None,
        )
        return await as_solver(agent)(state, generate)

    return solve


@scorer(metrics=[mean()])
def transaction_success():
    async def score(state: TaskState, target: Target) -> Score:
        record = state.metadata.get("dealroom", {"actions": []})
        evaluation = evaluate(
            replay(
                state.metadata["case_id"],
                record["actions"],
                action_limit=record.get("action_limit"),
            ),
            end_on_expiry=record.get("end_on_expiry") is True
            and record.get("termination_reason") == "contract_deadline_expired",
        )
        data = evaluation.model_dump(mode="json")
        limit_event = next(
            (event for event in transcript().events if isinstance(event, SampleLimitEvent)),
            None,
        )
        if limit_event is not None:
            return Score.unscored(
                reason="sample_limit",
                answer="unscored",
                explanation=(
                    f"Unscored: {limit_event.message}\n"
                    "The following evaluator snapshot is diagnostic context, not a terminal reward.\n"
                    + json.dumps(data, indent=2)
                ),
                metadata=data,
            )
        if evaluation.outcome in {"budget_exhausted", "incomplete"}:
            return Score.unscored(
                reason=evaluation.outcome,
                answer="unscored",
                explanation=(
                    "Unscored: the attempt did not reach a completed domain outcome.\n"
                    "The following evaluator snapshot is diagnostic context, not a terminal reward.\n"
                    + json.dumps(data, indent=2)
                ),
                metadata=data,
            )
        return Score(
            value=int(data["success"]),
            answer="success" if data["success"] else "failure",
            explanation=json.dumps(data, indent=2),
            metadata=data,
        )

    return score


@task
def dealroom(
    split: Literal["all", "development", "evaluation"] = "all",
    case_id: str | None = None,
    suite: Literal["curated", "synthetic"] = "curated",
    action_limit: int | None = None,
    end_on_expiry: bool = False,
    compaction_threshold: int | None = None,
):
    """Curated or generated tasks using the same tools, engine, and scorer."""
    for name, value in (
        ("action_limit", action_limit),
        ("compaction_threshold", compaction_threshold),
    ):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer when supplied.")
    if suite == "synthetic":
        from dealroom.synthesis import suite_case_ids

        ids = suite_case_ids()
    else:
        ids = [f"case-{i:02d}" for i in range(1, 7)]
    if split != "all":
        ids = [item for item in ids if load_case(item).split == split]
    if case_id is not None:
        if case_id not in ids:
            raise ValueError(f"{case_id!r} is not in split {split!r}")
        ids = [case_id]
    return Task(
        dataset=[sample_for(i, action_limit=action_limit) for i in ids],
        solver=coordinate(
            action_limit=action_limit,
            end_on_expiry=end_on_expiry,
            compaction_threshold=compaction_threshold,
        ),
        scorer=transaction_success(),
        config=GenerateConfig(parallel_tool_calls=False, max_tokens=2048, temperature=0),
        token_limit=100_000,
        message_limit=120,
        time_limit=180,
        fail_on_error=False,
        score_on_error=False,
        name="dealroom",
        display_name="DealRoom · evidence-based transaction coordination",
        version="0.1.0",
        metadata={"benchmark": suite, "engine_version": "0.1.0"},
        viewer=ViewerConfig(
            task_samples_view=TaskSamplesView(
                name="Transaction outcomes",
                multiline=False,
                columns=[
                    TaskSamplesColumn(id="sampleId"),
                    TaskSamplesColumn.score("transaction_success"),
                    TaskSamplesColumn(id="tokens"),
                    TaskSamplesColumn(id="duration"),
                    TaskSamplesColumn(id="error"),
                    TaskSamplesColumn(id="limit"),
                ],
                score_labels={"transaction_success": "Valid terminal outcome"},
                score_color_scales={"transaction_success": "good-high"},
            )
        ),
    )
