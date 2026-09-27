"""Math and data-boundary checks for local training; tiny tensors on CPU."""

import json
from types import SimpleNamespace

import pytest

from dealroom.learning_train import balanced_order, tokenize_demonstration


class Tokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        tokens = [1]
        for message in messages:
            tokens.extend([2 if message["role"] == "user" else 3, len(message["content"]), 4])
        return tokens + ([3] if add_generation_prompt else [])


def row(split="train"):
    return {"metadata": {"split": split, "target_use": "supervised_training" if split == "train" else "validation_loss_only",
                         "case_id": "learn-1", "action_type": "finish"},
            "messages": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": "answer"}]}


def test_only_last_assistant_is_supervised_and_validation_cannot_train():
    result = tokenize_demonstration(row(), Tokenizer(), "train")
    assert result["prompt_ids"] == [1, 2, 6, 4, 3]
    assert result["completion_ids"] == [6, 4]
    with pytest.raises(PermissionError):
        tokenize_demonstration(row("validation"), Tokenizer(), "train")
    with pytest.raises(PermissionError):
        tokenize_demonstration(row("sealed_test"), Tokenizer(), "sealed_test")
    value = row()
    value["messages"][0]["content"] = "attachment://missing"
    with pytest.raises(ValueError):
        tokenize_demonstration(value, Tokenizer(), "train")


def test_balanced_order_preserves_all_rows_without_mutating_and_is_seeded():
    rows = [{"action_type": "read", "id": n} for n in range(10)] + [{"action_type": "finish", "id": 11}]
    original = json.dumps(rows)
    result = balanced_order(rows, 3)
    assert any(item["action_type"] == "finish" for item in result[:2])
    assert sorted(item["id"] for item in result) == sorted(item["id"] for item in rows)
    assert balanced_order(rows, 3) == result
    assert json.dumps(rows) == original


def tiny_model():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    mx.set_default_device(mx.cpu)

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(11, 6)

        def __call__(self, inputs):
            return mx.cumsum(self.embed_tokens(inputs), axis=1)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.lm_head = nn.Linear(6, 11, bias=False)
            self.args = SimpleNamespace(tie_word_embeddings=False)

        def __call__(self, inputs):
            return self.lm_head(self.model(inputs))

    mx.random.seed(42)
    return mx, nn, Model()


def test_completion_loss_matches_full_logits_mask_and_remains_prompt_conditioned():
    from dealroom.learning_train import completion_loss

    mx, nn, model = tiny_model()
    prompt, completion = mx.array([[1, 2, 3]]), mx.array([[4, 5]])
    actual, count = completion_loss(model, prompt, completion, mx.array(1.0))
    all_tokens = mx.concatenate([prompt, completion], axis=1)
    expected = nn.losses.cross_entropy(model(all_tokens[:, :-1])[:, 2:], completion).mean()
    assert float(actual) == pytest.approx(float(expected), rel=1e-6)
    assert int(count) == 2
    changed, _ = completion_loss(model, mx.array([[8, 7, 3]]), completion, mx.array(1.0))
    assert float(changed) != pytest.approx(float(actual))


@pytest.mark.parametrize("advantage", [1.0, -1.0])
def test_policy_gradient_moves_sample_likelihood_in_reward_direction(advantage):
    from dealroom.learning_train import policy_loss

    mx, nn, model = tiny_model()
    from mlx.utils import tree_map
    prompt, completion = mx.array([[1, 2, 3]]), mx.array([[4, 5]])

    def actor_ce(model):
        return policy_loss(model, prompt, completion, mx.array(1.0))[0]

    before = float(actor_ce(model))
    _, gradients = nn.value_and_grad(model, policy_loss)(model, prompt, completion, mx.array(advantage))
    model.update(tree_map(lambda value, grad: value - 1e-3 * grad, model.trainable_parameters(), gradients))
    after = float(actor_ce(model))
    assert (before - after) * advantage > 0


def test_batches_refuse_truncation():
    pytest.importorskip("mlx.core")
    from dealroom.learning_train import batches

    with pytest.raises(ValueError, match="truncation"):
        next(batches([{"prompt_ids": [1, 2], "completion_ids": [3, 4]}], 1, 3))


@pytest.mark.parametrize("violation", ["empty", "validation", "wrong_policy", "overwrite", "zero_steps"])
def test_training_boundary_checks_run_before_model_loading(tmp_path, monkeypatch, violation):
    pytest.importorskip("mlx.core")
    mlx_lm = pytest.importorskip("mlx_lm")
    from dealroom.learning_train import train_adapter

    def forbidden_load(*args, **kwargs):
        pytest.fail("Invalid training input must be rejected before any model is loaded.")

    monkeypatch.setattr(mlx_lm, "load", forbidden_load)
    data = tmp_path / "data.jsonl"
    value = {"prompt_ids": [1, 2], "completion_ids": [3], "split": "train", "action_type": "finish"}
    options = {}
    output = tmp_path / "checkpoint"
    if violation == "validation":
        value["split"] = "validation"
    elif violation == "wrong_policy":
        parent = tmp_path / "parent"
        parent.mkdir()
        (parent / "adapters.safetensors").write_bytes(b"checkpoint bytes")
        options.update(mode="policy", parent=parent)
        value["policy_fingerprint"] = "different-checkpoint"
    elif violation == "overwrite":
        output.mkdir()
        (output / "adapters.safetensors").write_bytes(b"preserve")
    elif violation == "zero_steps":
        options["steps"] = 0
    data.write_text("" if violation == "empty" else json.dumps(value) + "\n")
    with pytest.raises((ValueError, PermissionError, FileExistsError)):
        train_adapter(tmp_path / "model", output, data, **options)
