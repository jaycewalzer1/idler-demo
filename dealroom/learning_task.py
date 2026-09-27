"""Versioned learning workflow on the unchanged DealRoom engine and evaluator.

The original adapter and its recorded benchmark remain frozen. This adapter can
load isolated curriculum fixtures and embeds the exact fixture in reviewer-only
sample metadata so a training or evaluation log remains independently replayable.
Neither fixture internals nor training witnesses enter the model's messages.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.agent import AgentPrompt, AgentSubmit, as_solver, react
from inspect_ai.dataset import Sample
from inspect_ai.event import SampleLimitEvent
from inspect_ai.log import transcript
from inspect_ai.model import CompactionTrim, GenerateConfig
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import Tool, ToolDef

from dealroom.curriculum import DEFAULT_LEARNING_DIR, load_learning_case
from dealroom.domain import Case, observe, reset, step
from dealroom.score import evaluate
from dealroom.task import PROMPT, fixture_fingerprint, public_index, tools_for

Workflow = Literal["original", "clarified"]
LEARNING_WORKFLOW_VERSION = "dealroom-learning-workflow-v1"

EXPIRY_INSTRUCTIONS = (
    "\nThis evaluation ends the episode immediately if the contractual deadline "
    "expires. Complete the required coordination before expiry; no further "
    "actions will be requested after the transaction is irreversibly expired.\n"
)

CLARIFIED_INSTRUCTIONS = """
Resource access:
- The resources field is an index. Read the listed public documents and messages
  to obtain their contents before making decisions. Use read with the resource
  kind and resource_id copied from the corresponding index entry. Reading summary
  returns the index again. Reads consume no simulated time but do count as actions.
- After a response arrives, new resources appear in the index. Read their contents.

Tool effects and completion:
- Monetary amounts use integer cents. Derive terms from current public evidence
  and current buyer authority. Informal agreement is not an executed amendment.
- request sends a request. draft_amendment creates an unsigned, immutable revision.
  send_for_signature routes that exact revision. Check its required signatures and
  execution status before relying on its credit or extension.
- wait_until advances all the way to the timestamp supplied. It does not stop at
  an earlier response. Use publicly disclosed response timings, check receipts,
  and leave time for subsequent actions before the contractual deadline.
- set_disposition is the tool that commits proceed or cancel. A rejected receipt
  means no disposition was committed. Verify the resulting recorded disposition.
- finish only submits factual claims and ends the attempt. It does not commit a
  disposition, execute amendments, or make claims true. Finish after completing
  the required authorized coordination; report only the recorded outcome.
- Every receipt says whether the action was accepted and why. Correct rejected
  actions using public evidence; repeating an identical request does not schedule
  an additional response. Make one tool call at a time.
"""

CLARIFIED_TOOL_DESCRIPTIONS = {
    "read": (
        "Read a public resource by its kind and exact resource_id from the resources index. "
        "Use summary without an ID to refresh the index. Reads consume no simulated time; "
        "an invalid resource ID returns a rejected receipt and the index."
    ),
    "wait_until": (
        "Advance simulated time all the way to the supplied timestamp, processing earlier events. "
        "This does not stop at the next response. Leave time for required coordination before expiry."
    ),
    "set_disposition": (
        "Commit proceed or cancel if authorized and feasible. This tool changes the transaction's "
        "recorded disposition; verify accepted and the resulting observation."
    ),
    "finish": (
        "Submit factual structured claims and end the attempt. This tool has no business effects: "
        "it does not commit a disposition or execute an amendment. Report the actual recorded state."
    ),
}


def _validate_workflow(workflow: str) -> None:
    if workflow not in {"original", "clarified"}:
        raise ValueError("workflow must be 'original' or 'clarified'.")


def _positive_integer(name: str, value: int | None) -> None:
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError(f"{name} must be a positive integer when supplied.")


def effective_learning_case(case: Case, action_limit: int | None = 256) -> Case:
    _positive_integer("action_limit", action_limit)
    return case.model_copy(update={"action_limit": action_limit}) if action_limit else case


def public_learning_input(case: Case, workflow: Workflow = "clarified") -> str:
    """Return only initial public observations; use the effective-budget case.

    Both workflows get identical case information. The clarified condition changes
    procedural instructions and tool descriptions, never the answer or evidence.
    """
    _validate_workflow(workflow)
    return json.dumps(public_index(observe(reset(case)), include_policy=True), ensure_ascii=False)


def learning_prompt(workflow: Workflow = "clarified", *, end_on_expiry: bool = True) -> str:
    """Exact AgentPrompt instructions shared by data construction and evaluation."""
    _validate_workflow(workflow)
    return (
        PROMPT
        + (CLARIFIED_INSTRUCTIONS if workflow == "clarified" else "")
        + (EXPIRY_INSTRUCTIONS if end_on_expiry else "")
    )


def learning_tools_for(engine, record: dict, workflow: Workflow = "clarified") -> tuple[list[Tool], Tool]:
    """Reuse all seven typed tools and authoritative engine transitions.

    A failed resource read in the frozen original adapter returns the full public
    observation. The clarified adapter consistently returns the resource index on
    such failures. Valid reads, actions, events, and terminal effects are identical.
    """
    _validate_workflow(workflow)
    tools, finish = tools_for(engine, record)
    if workflow == "original":
        return tools, finish

    original_read = tools[0]

    async def read(
        resource: Literal["summary", "document", "message", "envelope", "amendment"] = "summary",
        resource_id: str | None = None,
    ) -> str:
        data = json.loads(await original_read(resource=resource, resource_id=resource_id))
        if not data["accepted"]:
            data["observation"] = public_index(data["observation"], include_policy=True)
        return json.dumps(data, ensure_ascii=False)

    read_definition = ToolDef(original_read)
    tools[0] = ToolDef(
        read,
        name=read_definition.name,
        description=CLARIFIED_TOOL_DESCRIPTIONS["read"],
        parameters=read_definition.parameters,
        parallel=False,
        max_output=0,
    ).as_tool()

    def describe(tool: Tool) -> Tool:
        definition = ToolDef(tool)
        if definition.name in CLARIFIED_TOOL_DESCRIPTIONS:
            definition.description = CLARIFIED_TOOL_DESCRIPTIONS[definition.name]
        return definition.as_tool()

    return [describe(tool) for tool in tools], describe(finish)


def learning_source_fingerprints() -> dict[str, str]:
    """Source provenance of the shared environment and versioned adapter."""
    directory = Path(__file__).parent
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("domain.py", "score.py", "task.py", "learning_task.py", "curriculum.py")
    }


def learning_case_for_record(record: dict) -> Case:
    """Reconstruct an exact recorded case without depending on a mutable case store."""
    case = Case.model_validate(record["case_snapshot"])
    if case.id != record["case_id"]:
        raise ValueError("Recorded case ID does not match the embedded fixture.")
    if fixture_fingerprint(case) != record["base_fixture_sha256"]:
        raise ValueError("Recorded base fixture fingerprint does not match its snapshot.")
    case = effective_learning_case(case, record.get("action_limit"))
    if fixture_fingerprint(case) != record["fixture_sha256"]:
        raise ValueError("Recorded effective fixture fingerprint does not match its snapshot.")
    return case


def replay_learning_record(record: dict):
    engine = reset(learning_case_for_record(record))
    for action in record["actions"]:
        step(engine, action)
    return engine


def learning_sample_for(
    case: Case, workflow: Workflow = "clarified", action_limit: int | None = 256
) -> Sample:
    return Sample(
        id=case.id,
        input=public_learning_input(effective_learning_case(case, action_limit), workflow),
        # Inspect metadata is reviewer-only; never append this snapshot to messages.
        metadata={"case_id": case.id, "learning_case_snapshot": case.model_dump(mode="json")},
    )


@solver
def learning_coordinate(
    workflow: Workflow = "clarified",
    action_limit: int | None = 256,
    end_on_expiry: bool = True,
    compaction_threshold: int | None = None,
):
    _validate_workflow(workflow)
    _positive_integer("action_limit", action_limit)
    _positive_integer("compaction_threshold", compaction_threshold)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        base_case = Case.model_validate(state.metadata["learning_case_snapshot"])
        case = effective_learning_case(base_case, action_limit)
        engine = reset(case)
        record = {
            "case_id": case.id,
            "case_snapshot": base_case.model_dump(mode="json"),
            "fixture_sha256": fixture_fingerprint(case),
            "base_fixture_sha256": fixture_fingerprint(base_case),
            "workflow": workflow,
            "workflow_version": LEARNING_WORKFLOW_VERSION,
            "source_fingerprints": learning_source_fingerprints(),
            "action_limit": case.action_limit,
            "end_on_expiry": end_on_expiry,
            "compaction_threshold": compaction_threshold,
            "actions": [],
            "events": [],
        }
        state.metadata["dealroom"] = record
        tools, finish = learning_tools_for(engine, record, workflow)

        async def continue_run(_state):
            if end_on_expiry and engine.business_status == "expired":
                record["termination_reason"] = "contract_deadline_expired"
                return False
            return not engine.budget_exhausted

        agent = react(
            name="transaction_coordinator",
            prompt=AgentPrompt(
                instructions=learning_prompt(workflow, end_on_expiry=end_on_expiry),
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
def learning_success():
    """Unchanged terminal predicates; unfinished computation is explicitly unscored."""
    async def score(state: TaskState, target: Target) -> Score:
        record = state.metadata.get("dealroom")
        if record is None:
            evaluation = evaluate(reset(Case.model_validate(state.metadata["learning_case_snapshot"])))
        else:
            evaluation = evaluate(
                replay_learning_record(record),
                end_on_expiry=record.get("end_on_expiry") is True
                and record.get("termination_reason") == "contract_deadline_expired",
            )
        data = evaluation.model_dump(mode="json")
        limit = next((e for e in transcript().events if isinstance(e, SampleLimitEvent)), None)
        if limit is not None or evaluation.outcome in {"budget_exhausted", "incomplete"}:
            return Score.unscored(
                reason="sample_limit" if limit is not None else evaluation.outcome,
                answer="unscored",
                explanation=(
                    f"Unscored: {limit.message if limit else 'no completed domain outcome'}.\n"
                    "The following evaluator snapshot is diagnostic context, not a terminal reward.\n"
                    + json.dumps(data, indent=2)
                ),
                metadata=data,
            )
        return Score(
            value=int(evaluation.success),
            answer="success" if evaluation.success else "failure",
            explanation=json.dumps(data, indent=2),
            metadata=data,
        )

    return score


@task
def learning_dealroom(
    case_id: str,
    case_dir: str | Path = DEFAULT_LEARNING_DIR,
    workflow: Workflow = "clarified",
    action_limit: int | None = 256,
    end_on_expiry: bool = True,
    compaction_threshold: int | None = None,
):
    """One curriculum fixture with a declared original or clarified interface."""
    _validate_workflow(workflow)
    _positive_integer("action_limit", action_limit)
    _positive_integer("compaction_threshold", compaction_threshold)
    case = load_learning_case(case_id, case_dir=Path(case_dir))
    return Task(
        dataset=[learning_sample_for(case, workflow, action_limit)],
        solver=learning_coordinate(workflow, action_limit, end_on_expiry, compaction_threshold),
        scorer=learning_success(),
        config=GenerateConfig(parallel_tool_calls=False, max_tokens=2048, temperature=0),
        token_limit=2_000_000,
        message_limit=512,
        turn_limit=128,
        time_limit=900,
        fail_on_error=False,
        score_on_error=False,
        name="learning_dealroom",
        display_name=f"DealRoom learning · {workflow} workflow",
        version=LEARNING_WORKFLOW_VERSION,
        metadata={
            "workflow": workflow,
            "workflow_version": LEARNING_WORKFLOW_VERSION,
            "source_fingerprints": learning_source_fingerprints(),
            "base_fixture_sha256": fixture_fingerprint(case),
        },
    )
