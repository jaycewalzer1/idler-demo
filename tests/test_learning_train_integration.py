"""Actual upstream MLX LoRA optimization on tiny CPU-only Llama fixtures.

Run with training/.venv/bin/python -m pytest. The ordinary environment skips this
optional integration test if MLX is absent; no downloaded model is loaded.
"""

import json
import math

import pytest


@pytest.mark.parametrize("quantized", [False, True])
def test_sft_then_one_policy_update_preserves_base_and_parent_lineage(
    tmp_path, monkeypatch, quantized
):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    mlx_lm = pytest.importorskip("mlx_lm")
    from mlx.utils import tree_flatten
    from mlx_lm.models.llama import Model, ModelArgs, TransformerBlock
    from mlx_lm.tuner import utils as tuner_utils

    from dealroom.learning_policy import make_policy_segments
    from dealroom.learning_train import file_digest, train_adapter

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    # Upstream checks hardware availability instead of the selected device and
    # otherwise requests a Metal-only memory-limit field from CPU device_info.
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    # Upstream grad_checkpoint changes the class globally. Restore it after the
    # test so repeated train calls do not alter unrelated tests' model classes.
    monkeypatch.setattr(TransformerBlock, "__call__", TransformerBlock.__call__)
    try:
        args = ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            tie_word_embeddings=False,
        )

        def new_base():
            model = Model(args)
            if quantized:
                # Matches the real model's frozen FP16 scales/4-bit base and
                # checks that new trainable LoRA matrices remain float32.
                model.apply(lambda parameter: parameter.astype(mx.float16))
                nn.quantize(model, group_size=32, bits=4)
            mx.eval(model.parameters())
            return model

        mx.random.seed(17)
        base_model = new_base()
        model_dir = tmp_path / "base"
        model_dir.mkdir()
        base_path = model_dir / "model.safetensors"
        mx.save_safetensors(str(base_path), dict(tree_flatten(base_model.parameters())))
        base_digest = file_digest(base_path)
        original_base = mx.load(str(base_path))
        loaded_models = []

        def fake_load(path, **kwargs):
            assert path == str(model_dir)
            model = new_base()
            model.load_weights(str(base_path))
            loaded_models.append(model)
            return model, None

        monkeypatch.setattr(mlx_lm, "load", fake_load)
        rows = [
            {
                "prompt_ids": [1, 2, 3],
                "completion_ids": [4, 5, 6],
                "case_id": "train-a",
                "action_type": "read",
                "split": "train",
                "weight": 1.0,
            },
            {
                "prompt_ids": [1, 7],
                "completion_ids": [8, 9],
                "case_id": "train-b",
                "action_type": "finish",
                "split": "train",
                "weight": 1.0,
            },
        ]
        train_path = tmp_path / "train.jsonl"
        train_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        sft_dir = tmp_path / "sft"
        sft = train_adapter(
            model_dir,
            sft_dir,
            train_path,
            steps=2,
            learning_rate=1e-3,
            max_length=32,
            seed=20,
        )
        assert sft["optimizer_updates"] == 2
        assert sft["parent"] is None and sft["parent_digest"] is None
        assert sft["training_examples_seen"] == 2
        assert sft["training_cases_seen"] == 2
        assert sft["processed_prompt_tokens"] == 5
        assert sft["optimized_completion_tokens"] == 5
        sft_weights = mx.load(str(sft_dir / "adapters.safetensors"))
        assert sft_weights and all(name.endswith(("lora_a", "lora_b")) for name in sft_weights)
        assert all(value.dtype == mx.float32 for value in sft_weights.values())
        assert sft["trainable_parameters"] == sum(value.size for value in sft_weights.values())
        assert all(bool(mx.all(mx.isfinite(value))) for value in sft_weights.values())
        assert any(
            bool(mx.any(value != 0))
            for name, value in sft_weights.items()
            if name.endswith("lora_b")
        )
        sft_optimizer = mx.load(str(sft_dir / "optimizer.safetensors"))
        assert int(sft_optimizer["step"]) == 2

        # Real policy-segment weighting: uneven trajectory lengths, three
        # segments, one mean-accumulated update across those three segments.
        fingerprint = sft["digest"]

        def segment(prompt, completion):
            return {
                "prompt_ids": prompt,
                "completion_ids": completion,
                "policy_fingerprint": fingerprint,
                "behavior_logprobs": [-1.0] * len(completion),
            }

        group = make_policy_segments(
            [
                {
                    "trajectory_id": "good",
                    "case_id": "train-a",
                    "status": "success",
                    "reward": 1,
                    "segments": [segment([1, 2, 3], [4, 5]), segment([1, 2, 3, 4, 5], [6])],
                },
                {
                    "trajectory_id": "bad",
                    "case_id": "train-a",
                    "status": "failure",
                    "reward": 0,
                    "segments": [segment([1, 2, 3], [9, 10, 11, 12])],
                },
            ]
        )
        policy_path = tmp_path / "policy.jsonl"
        policy_path.write_text("".join(json.dumps({**row, "split": "train"}) + "\n" for row in group["segments"]))
        observed_parent = {}
        actual_load_adapters = tuner_utils.load_adapters

        def observe_parent(model, adapter_path):
            result = actual_load_adapters(model, adapter_path)
            observed_parent.update(dict(tree_flatten(model.trainable_parameters())))
            mx.eval(observed_parent)
            return result

        monkeypatch.setattr(tuner_utils, "load_adapters", observe_parent)
        policy_dir = tmp_path / "policy"
        policy = train_adapter(
            model_dir,
            policy_dir,
            policy_path,
            parent=sft_dir,
            mode="policy",
            steps=999,
            learning_rate=5e-4,
            max_length=32,
            seed=21,
            optimizer_state=sft_dir / "optimizer.safetensors",
        )
        assert policy["steps"] == 3
        assert policy["optimizer_updates"] == 1
        assert policy["training_examples_seen"] == 3
        assert policy["training_cases_seen"] == 1
        assert policy["processed_prompt_tokens"] == 11
        assert policy["optimized_completion_tokens"] == 7
        assert policy["parent"] == str(sft_dir)
        assert policy["parent_digest"] == sft["digest"]
        assert policy["digest"] != sft["digest"]
        assert policy["digest"] == file_digest(policy_dir / "adapters.safetensors")
        assert observed_parent.keys() == sft_weights.keys()
        assert all(
            bool(mx.array_equal(observed_parent[name], value))
            for name, value in sft_weights.items()
        )
        policy_weights = mx.load(str(policy_dir / "adapters.safetensors"))
        assert all(value.dtype == mx.float32 for value in policy_weights.values())
        assert all(bool(mx.all(mx.isfinite(value))) for value in policy_weights.values())
        assert any(
            not bool(mx.array_equal(policy_weights[name], value))
            for name, value in sft_weights.items()
        )
        policy_optimizer = mx.load(str(policy_dir / "optimizer.safetensors"))
        assert int(policy_optimizer["step"]) == 3  # two SFT + exactly one policy update
        assert float(policy_optimizer["learning_rate"]) == pytest.approx(5e-4)
        assert all(
            math.isfinite(row["train_loss"]) for row in policy["history"] if row["kind"] == "train"
        )
        config = json.loads((policy_dir / "adapter_config.json").read_text())
        assert config["parent"] == str(sft_dir)

        # Both loaded base models remain bit-identical, including embedding,
        # output projection, layer norms, and all quantized weight/scale arrays.
        for model in loaded_models:
            for name, value in tree_flatten(model.parameters()):
                if name.endswith(("lora_a", "lora_b")):
                    continue
                base_name = name.replace(".q_proj.linear.", ".q_proj.").replace(
                    ".v_proj.linear.", ".v_proj."
                )
                assert base_name in original_base, base_name
                assert bool(mx.array_equal(value, original_base[base_name])), base_name
            assert all(
                name.endswith(("lora_a", "lora_b"))
                for name, _ in tree_flatten(model.trainable_parameters())
            )
        assert file_digest(base_path) == base_digest
        assert file_digest(sft_dir / "adapters.safetensors") == sft["digest"]
    finally:
        mx.set_default_device(previous_device)
