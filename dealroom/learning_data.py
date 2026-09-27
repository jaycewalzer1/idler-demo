"""Export verified supervised demonstrations using Inspect's exact tool wire format.

These are scripted demonstrations, never model-performance measurements. Training
and validation rows stay separate, and sealed-test cases are rejected. The JSONL
``messages`` field contains the actual OpenAI-compatible text-tool prompt followed
by one assistant target; a trainer must mask every token except that last response.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.event import ModelEvent, ToolEvent
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput, get_model
from inspect_ai.model._openai import messages_to_openai
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleHandler
from inspect_ai.model._providers.util.chatapi import chat_api_messages_for_handler
from inspect_ai.tool import ToolCall

from dealroom.curriculum import (
    DEFAULT_LEARNING_DIR,
    case_entry,
    load_learning_case,
    load_manifest,
    training_witness,
)
from dealroom.domain import parse_action
from dealroom.learning_task import (
    LEARNING_WORKFLOW_VERSION,
    learning_dealroom,
    learning_sample_for,
    learning_source_fingerprints,
    replay_learning_record,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "training" / "data"
INSPECT_VERSION = "0.3.271"
DATA_VERSION = "dealroom-sft-data-v1"
DemonstrationSplit = Literal["train", "validation"]


def _fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def demonstration_actions(
    case_id: str, split: DemonstrationSplit, directory: str | Path = DEFAULT_LEARNING_DIR
) -> list[dict]:
    """Load only the requested permitted split; validation targets are loss-only.

    Validation targets may not enter the training optimizer. Sealed witness files
    are never read, even if a caller supplies their IDs under another split name.
    """
    if split not in {"train", "validation"}:
        raise PermissionError("Only train and validation demonstrations may be exported.")
    entry = case_entry(case_id, directory)
    if entry["split"] != split:
        raise PermissionError(f"Case {case_id} is {entry['split']}, not permitted split {split}.")
    if split == "train":
        return training_witness(case_id, directory)
    load_learning_case(case_id, directory)
    artifact = json.loads((Path(directory) / "private" / f"{case_id}.json").read_text())
    actions = artifact["witness"]
    if (
        artifact["case_id"] != case_id
        or artifact["split"] != "validation"
        or _fingerprint(actions) != entry["validation"]["witness_fingerprint"]
    ):
        raise ValueError("Validation demonstration fingerprint mismatch.")
    return actions


def demonstration_output(action: dict, index: int) -> ModelOutput:
    """A clean supervised target, with no fabricated model reasoning or results."""
    arguments = deepcopy(action)
    function = arguments.pop("type")
    return ModelOutput(
        model="mockllm/scripted-sft-demonstration",
        choices=[ChatCompletionChoice(
            message=ChatMessageAssistant(content="", tool_calls=[ToolCall(
                id=f"demonstration-{index}", function=function, arguments=arguments,
            )]),
            stop_reason="tool_calls",
        )],
    )


def _check_inspect_version() -> None:
    if version("inspect-ai") != INSPECT_VERSION:
        raise RuntimeError(f"SFT wire conversion is pinned to inspect-ai=={INSPECT_VERSION}.")


async def emulated_messages(event: ModelEvent) -> list[dict]:
    """Use exactly the conversion used by Inspect's OpenAICompatibleAPI.generate."""
    _check_inspect_version()
    handler = OpenAICompatibleHandler("scripted-sft-demonstration")
    prompt = chat_api_messages_for_handler(event.input, event.tools, handler)
    wire = await messages_to_openai(prompt)
    completion = handler.assistant_message(event.output.choices[0].message)
    messages = [dict(message) for message in wire] + [completion]
    if any(set(message) != {"role", "content"} or not isinstance(message["content"], str)
           for message in messages):
        raise ValueError("The text-only SFT exporter cannot silently flatten multimodal messages.")
    if any("attachment://" in message["content"] for message in messages):
        raise ValueError("Resolve native log attachments before exporting model prompts.")
    return messages


def run_demonstrations(
    case_ids: list[str], split: DemonstrationSplit, directory: str | Path,
    log_dir: str | Path,
) -> EvalLog:
    """Execute scripted targets through the native ReAct/tools/scorer machinery."""
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("Demonstration cases must be non-empty and unique.")
    actions = {case_id: demonstration_actions(case_id, split, directory) for case_id in case_ids}
    task = learning_dealroom(case_ids[0], case_dir=directory, workflow="clarified")
    task.dataset = MemoryDataset([
        learning_sample_for(load_learning_case(case_id, directory), "clarified")
        for case_id in case_ids
    ])
    # Per-case fingerprints remain in each native sample; this task is a batch.
    task.metadata.pop("base_fixture_sha256", None)
    task.metadata.update({
        "run_kind": "scripted_sft_demonstration",
        "is_model_performance": False,
        "split": split,
        "target_use": "supervised_training" if split == "train" else "validation_loss_only",
        "data_version": DATA_VERSION,
        "curriculum_fingerprint": load_manifest(directory)["fingerprint"],
    })

    def output(messages, tools, tool_choice, config):
        initial = next(message.text for message in messages if message.role == "user")
        case_id = json.loads(initial)["case_id"]
        index = sum(message.role == "assistant" for message in messages)
        return demonstration_output(actions[case_id][index], index)

    model = get_model("mockllm/model", custom_outputs=output, memoize=False)
    logs = inspect_eval(
        task, model=model, log_dir=str(log_dir), display="none", log_model_api=True,
        max_samples=8, max_connections=8,
    )
    if len(logs) != 1:
        raise RuntimeError("Expected one native demonstration log.")
    log = read_eval_log(logs[0].location, resolve_attachments=True)
    if log.status != "success" or not log.samples or len(log.samples) != len(case_ids):
        raise RuntimeError("Native demonstration batch did not complete.")
    for sample in log.samples:
        if sample.error or sample.limit or sample.scores["learning_success"].value != 1:
            raise RuntimeError(f"Demonstration failed the unchanged evaluator: {sample.id}")
        record = sample.metadata["dealroom"]
        if [parse_action(action) for action in record["actions"]] != [
            parse_action(action) for action in actions[sample.id]
        ]:
            raise RuntimeError("The executed trajectory differs from its intended targets.")
        engine = replay_learning_record(record)
        if any(not item["accepted"] for item in engine.history):
            raise RuntimeError("Rejected actions cannot enter the successful demonstration export.")
        if any(event.error for event in sample.events if isinstance(event, ToolEvent)):
            raise RuntimeError("Tool errors cannot enter the successful demonstration export.")
    return log


async def _rows_from_log(log: EvalLog, split: DemonstrationSplit, log_reference: str) -> list[dict]:
    rows = []
    for sample in log.samples or []:
        record = sample.metadata["dealroom"]
        events = [event for event in sample.events if isinstance(event, ModelEvent)]
        if len(events) != len(record["actions"]):
            raise RuntimeError("Expected one model decision for each demonstration action.")
        for index, event in enumerate(events):
            if event.error or len(event.output.choices) != 1:
                raise RuntimeError("Invalid native model event in demonstration.")
            messages = await emulated_messages(event)
            rows.append({
                "messages": messages,
                "metadata": {
                    "case_id": sample.id,
                    "split": split,
                    "action_index": index,
                    "action_type": record["actions"][index]["type"],
                    "native_log": log_reference,
                    "workflow_version": LEARNING_WORKFLOW_VERSION,
                    "fixture_sha256": record["fixture_sha256"],
                    "source": "scripted_sft_demonstration",
                    "target_use": "supervised_training" if split == "train" else "validation_loss_only",
                },
            })
    return rows


def export_sft_data(
    directory: str | Path = DEFAULT_LEARNING_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit_per_split: int | None = None,
) -> dict:
    """Export train.jsonl and validation.jsonl, never sealed-test data.

    A limit is an explicit development convenience, recorded in the manifest;
    cases are taken in frozen manifest order without outcome-based selection.
    """
    _check_inspect_version()
    if limit_per_split is not None and (type(limit_per_split) is not int or limit_per_split < 1):
        raise ValueError("limit_per_split must be a positive integer.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("SFT output already has a manifest; use a fresh directory.")
    curriculum = load_manifest(directory)
    manifest = {
        "schema_version": 1,
        "data_version": DATA_VERSION,
        "workflow_version": LEARNING_WORKFLOW_VERSION,
        "inspect_version": INSPECT_VERSION,
        "curriculum_fingerprint": curriculum["fingerprint"],
        "source_fingerprints": learning_source_fingerprints(),
        "wire_format": "Inspect OpenAICompatibleAPI emulate_tools=true",
        "native_log_attachments": "Fully resolved; unresolved attachment references are rejected.",
        "loss_mask": "Only the final assistant response is a target; mask all prompt tokens.",
        "validation_policy": "validation.jsonl is exclusively for loss measurement/checkpoint selection, never optimizer updates.",
        "sealed_test_policy": "Sealed case inputs, trajectories, and answers are never exported.",
        "run_kind": "scripted_sft_demonstration",
        "is_model_performance": False,
        "limit_per_split": limit_per_split,
        "splits": {},
    }
    for split in ("train", "validation"):
        case_ids = [task["id"] for task in curriculum["tasks"] if task["split"] == split]
        if limit_per_split is not None:
            case_ids = case_ids[:limit_per_split]
        log = run_demonstrations(case_ids, split, directory, output / "inspect" / split)
        log_reference = str(Path(log.location).relative_to(output))
        rows = asyncio.run(_rows_from_log(log, split, log_reference))
        file_path = output / f"{split}.jsonl"
        temporary = file_path.with_suffix(".jsonl.tmp")
        with temporary.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(file_path)
        manifest["splits"][split] = {
            "cases": len(case_ids), "rows": len(rows), "case_ids": case_ids,
            "file": file_path.name,
            "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
            "native_log": log_reference,
            "all_demonstrations_accepted_and_successful": True,
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_LEARNING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()
    manifest = export_sft_data(args.directory, args.output_dir, args.limit_per_split)
    print(json.dumps({split: {key: item[key] for key in ("cases", "rows", "file")}
                      for split, item in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
