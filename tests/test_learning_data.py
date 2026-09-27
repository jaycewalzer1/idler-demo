"""Supervised data preserves native tool-wire semantics and split boundaries."""

import json

import httpx2
import pytest
from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log
from inspect_ai.model._providers.openai_compatible import (
    OpenAICompatibleAPI,
    OpenAICompatibleHandler,
)

from dealroom.curriculum import DEFAULT_LEARNING_DIR, load_manifest
from dealroom.learning_data import (
    demonstration_actions,
    emulated_messages,
    export_sft_data,
    run_demonstrations,
)


@pytest.fixture(scope="module")
def exported_data(tmp_path_factory):
    output = tmp_path_factory.mktemp("sft-data")
    manifest = export_sft_data(output_dir=output, limit_per_split=1)
    return output, manifest


def test_export_separates_training_validation_and_excludes_sealed(exported_data):
    output, manifest = exported_data
    expected = load_manifest()
    sealed_ids = {task["id"] for task in expected["tasks"] if task["split"] == "sealed_test"}
    assert manifest["is_model_performance"] is False
    assert manifest["run_kind"] == "scripted_sft_demonstration"
    assert "final assistant" in manifest["loss_mask"]
    case_ids = set()
    for split in ("train", "validation"):
        entry = manifest["splits"][split]
        assert entry["cases"] == 1 and entry["rows"] > 1
        assert entry["all_demonstrations_accepted_and_successful"]
        rows = [json.loads(line) for line in (output / entry["file"]).read_text().splitlines()]
        assert len(rows) == entry["rows"]
        assert {row["metadata"]["case_id"] for row in rows} == set(entry["case_ids"])
        assert not case_ids.intersection(entry["case_ids"])
        case_ids.update(entry["case_ids"])
        for index, row in enumerate(rows):
            assert row["metadata"]["split"] == split
            assert row["metadata"]["action_index"] == index
            if split == "validation":
                assert row["metadata"]["target_use"] == "validation_loss_only"
            messages = row["messages"]
            assert messages[-1]["role"] == "assistant"
            assert messages[-1]["content"].startswith("<tool_call>")
            assert messages[-1]["content"].endswith("</tool_call>")
            assert "<tools>" in messages[0]["content"]
            assert "You are the buyer's transaction coordinator" in messages[1]["content"]
            assert json.loads(messages[2]["content"])["case_id"] == row["metadata"]["case_id"]
            prompt = json.dumps(messages)
            for forbidden in ("case_snapshot", "fixture_sha256", "acceptable_dispositions",
                              "buyer_grants", "signature_minutes", "pending_events", "attachment://"):
                assert forbidden not in prompt
        log = read_eval_log(output / entry["native_log"])
        assert log.eval.task_args["workflow"] == "clarified"
        assert log.eval.metadata["is_model_performance"] is False
        assert all(sample.scores["learning_success"].value == 1 for sample in log.samples)
    assert not case_ids.intersection(sealed_ids)


@pytest.mark.parametrize("requested_split", ["train", "validation", "sealed_test"])
def test_sealed_ids_are_refused_before_loading_witness(requested_split):
    case_id = next(task["id"] for task in load_manifest()["tasks"] if task["split"] == "sealed_test")
    with pytest.raises(PermissionError):
        demonstration_actions(case_id, requested_split)


def test_validation_cases_cannot_be_exported_as_train_rows():
    case_id = next(task["id"] for task in load_manifest()["tasks"] if task["split"] == "validation")
    with pytest.raises(PermissionError):
        demonstration_actions(case_id, "train")


def test_export_does_not_overwrite_existing_manifest(exported_data):
    output, _ = exported_data
    with pytest.raises(FileExistsError):
        export_sft_data(output_dir=output, limit_per_split=1)


async def test_exported_messages_equal_actual_openai_compatible_http_wire(exported_data):
    output, manifest = exported_data
    log = read_eval_log(output / manifest["splits"]["train"]["native_log"], resolve_attachments=True)
    events = [event for event in log.samples[0].events if isinstance(event, ModelEvent)]
    # The final event exercises prior assistant calls and user-wrapped tool results.
    event = events[-1]
    expected = await emulated_messages(event)
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "id": "wire-equivalence", "object": "chat.completion", "created": 0,
            "model": "fixture", "choices": [{
                "index": 0, "message": expected[-1], "finish_reason": "stop",
            }], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    provider = OpenAICompatibleAPI(
        "fixture/model", base_url="http://test.invalid/v1", api_key="fixture-only",
        emulate_tools=True, http_client=client,
    )
    try:
        result = await provider.generate(event.input, event.tools, event.tool_choice, event.config)
    finally:
        await provider.aclose()
    assert len(captured) == 1
    assert captured[0]["messages"] == expected[:-1]
    assert "tools" not in captured[0] and "tool_choice" not in captured[0]
    result = result[0] if isinstance(result, tuple) else result
    parsed = result.choices[0].message
    target = event.output.choices[0].message
    assert [(call.function, call.arguments) for call in parsed.tool_calls] == [
        (call.function, call.arguments) for call in target.tool_calls
    ]
    assert OpenAICompatibleHandler("fixture").assistant_message(parsed) == expected[-1]


async def test_unresolved_log_attachments_cannot_enter_export(exported_data):
    output, manifest = exported_data
    log = read_eval_log(output / manifest["splits"]["train"]["native_log"])
    event = next(event for event in log.samples[0].events if isinstance(event, ModelEvent))
    with pytest.raises(ValueError, match="Resolve native log attachments"):
        await emulated_messages(event)


def test_log_uses_only_requested_split(tmp_path):
    manifest = load_manifest()
    train = next(item["id"] for item in manifest["tasks"] if item["split"] == "train")
    validation = next(item["id"] for item in manifest["tasks"] if item["split"] == "validation")
    with pytest.raises(PermissionError):
        run_demonstrations([train, validation], "train", DEFAULT_LEARNING_DIR, tmp_path)
