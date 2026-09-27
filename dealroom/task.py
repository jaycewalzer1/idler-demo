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
from inspect_ai.model import GenerateConfig
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


def replay(case_id: str, actions: list[dict]):
    engine = reset(load_case(case_id))
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


def sample_for(case_id: str) -> Sample:
    public = observe(reset(load_case(case_id)))
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
def coordinate():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        case_id = state.metadata["case_id"]
        case = load_case(case_id)
        engine = reset(case)
        record = {
            "case_id": case_id,
            "fixture_sha256": fixture_fingerprint(case),
            "actions": [],
            "events": [],
        }
        state.metadata["dealroom"] = record
        tools, finish = tools_for(engine, record)

        async def continue_run(_state):
            return not getattr(engine, "budget_exhausted", False)

        agent = react(
            name="transaction_coordinator",
            prompt=AgentPrompt(
                instructions=PROMPT,
                assistant_prompt=None,
                handoff_prompt=None,
                submit_prompt=None,
            ),
            tools=tools,
            submit=AgentSubmit(name="finish", tool=finish, keep_in_messages=True),
            attempts=1,
            on_continue=continue_run,
        )
        return await as_solver(agent)(state, generate)

    return solve


@scorer(metrics=[mean()])
def transaction_success():
    async def score(state: TaskState, target: Target) -> Score:
        record = state.metadata.get("dealroom", {"actions": []})
        evaluation = evaluate(replay(state.metadata["case_id"], record["actions"]))
        data = evaluation.model_dump(mode="json")
        return Score(
            value=int(data["success"]),
            answer="success" if data["success"] else "failure",
            explanation=json.dumps(data, indent=2),
            metadata=data,
        )

    return score


@task
def dealroom(
    split: Literal["all", "development", "evaluation"] = "all", case_id: str | None = None
):
    """Six curated cases. Pass the provider/model using Inspect's --model option."""
    ids = [f"case-{i:02d}" for i in range(1, 7)]
    if split == "development":
        ids = ids[:2]
    elif split == "evaluation":
        ids = ids[2:]
    if case_id is not None:
        if case_id not in ids:
            raise ValueError(f"{case_id!r} is not in split {split!r}")
        ids = [case_id]
    return Task(
        dataset=[sample_for(i) for i in ids],
        solver=coordinate(),
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
        metadata={"benchmark": "six curated synthetic cases", "engine_version": "0.1.0"},
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
