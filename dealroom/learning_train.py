"""Local MLX LoRA training, with completion-only SFT and on-policy policy gradients.

Inspect remains the agent/evaluation harness. This module only optimizes recorded
assistant tokens; it never generates actions, reads sealed targets, or scores tasks.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_ARGS = {"date_string": "27 Sep 2026"}
LORA_CONFIG = {"rank": 8, "scale": 16.0, "dropout": 0.0, "keys": ["self_attn.q_proj", "self_attn.v_proj"]}


def detached_array(value):
    """Make an eager constant for a custom VJP's independently recomputed forward.

    stop_gradient still carries MLX tracer state, which makes SDPA retain dense
    attention intermediates. A host-backed constant removes that trace entirely;
    the custom VJP explicitly supplies the correct first-order derivative.
    """
    import mlx.core as mx
    import numpy as np

    mx.eval(value)
    return mx.array(np.asarray(value))


@contextlib.contextmanager
def memory_bounded_attention(chunk_size: int = 256):
    """Exact causal query blocks with recomputation, avoiding full NxN gradients.

    Only training's Llama attention is wrapped. Every query retains all preceding
    keys; context, targets, LoRA layout, and inference behavior remain unchanged.
    """
    import mlx.core as mx
    from mlx_lm.models import llama

    original = llama.scaled_dot_product_attention
    original_compile = mx.compile

    def attention(queries, keys, values, *, cache, scale, mask, **kwargs):
        length = queries.shape[-2]
        if cache is not None or length <= chunk_size or not isinstance(mask, str) or mask != "causal":
            return original(queries, keys, values, cache=cache, scale=scale, mask=mask, **kwargs)
        def block(q, k, v):
            return original(q, k, v, cache=None, scale=scale, mask="causal", **kwargs)

        @mx.custom_function
        def bounded(q, k, v):
            q, k, v = map(detached_array, (q, k, v))
            pieces = []
            for start in range(0, length, chunk_size):
                stop = min(start + chunk_size, length)
                piece = block(q[..., start:stop, :], k[..., :stop, :], v[..., :stop, :])
                mx.eval(piece)
                if os.getenv("DEALROOM_MEMORY_TRACE") and start % 4096 == 0:
                    print("attention", q.shape, start, mx.get_active_memory(), mx.get_cache_memory(), flush=True)
                pieces.append(piece)
                mx.clear_cache()
            return mx.concatenate(pieces, axis=-2)

        @bounded.vjp
        def backward(primals, cotangent, output):
            q, k, v = map(detached_array, primals)
            cotangent = detached_array(cotangent)
            dq = mx.zeros_like(q)
            dk, dv = mx.zeros(k.shape, mx.float32), mx.zeros(v.shape, mx.float32)
            for start in range(0, length, chunk_size):
                stop = min(start + chunk_size, length)
                # Explicit grouped-query attention VJP. MLX's generic SDPA
                # backward can allocate a full square matrix even for slices.
                qb = q[..., start:stop, :]
                kb, vb = k[..., :stop, :], v[..., :stop, :]
                batch, heads, count, width = qb.shape
                kv_heads = kb.shape[1]
                groups = heads // kv_heads
                qg = qb.reshape(batch, kv_heads, groups, count, width)
                kg, vg = kb[:, :, None], vb[:, :, None]
                scores = (qg * scale) @ mx.swapaxes(kg, -1, -2)
                causal = mx.arange(stop)[None, :] <= mx.arange(start, stop)[:, None]
                scores = mx.where(causal, scores, -float("inf"))
                probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
                go = cotangent[..., start:stop, :].reshape(batch, kv_heads, groups, count, width)
                gp = go @ mx.swapaxes(vg, -1, -2)
                gs = probabilities * (gp - (gp * probabilities).sum(axis=-1, keepdims=True))
                gq = ((gs @ kg) * scale).reshape(qb.shape)
                gk = ((mx.swapaxes(gs, -1, -2) @ qg) * scale).sum(axis=2)
                gv = (mx.swapaxes(probabilities, -1, -2) @ go).sum(axis=2)
                gq, gk, gv = map(detached_array, (gq, gk, gv))
                dq = dq.at[..., start:stop, :].add(gq)
                dk = dk.at[..., :stop, :].add(gk.astype(mx.float32))
                dv = dv.at[..., :stop, :].add(gv.astype(mx.float32))
                # Materialize one block before constructing the next. Otherwise
                # the lazy graph can retain every block's K/V gradient buffers.
                dq, dk, dv = map(detached_array, (dq, dk, dv))
                mx.eval(dq, dk, dv)
                if os.getenv("DEALROOM_MEMORY_TRACE") and start % 4096 == 0:
                    print("backward", start, mx.get_active_memory(), mx.get_cache_memory(), flush=True)
                mx.clear_cache()
            return dq, dk.astype(k.dtype), dv.astype(v.dtype)

        return bounded(queries, keys, values)

    llama.scaled_dot_product_attention = attention
    # The upstream trainer compiles the whole optimizer step. Eager stepping
    # preserves the explicit per-block materialization/release boundary above.
    mx.compile = lambda fun=None, **options: fun if fun is not None else lambda function: function
    try:
        yield
    finally:
        llama.scaled_dot_product_attention = original
        mx.compile = original_compile


@contextlib.contextmanager
def memory_bounded_mlp(chunk_size=256):
    """Recompute frozen feed-forward input gradients in independent token blocks."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models.llama import MLP

    original = MLP.__call__

    def call(layer, hidden):
        if tree_flatten(layer.trainable_parameters()):
            raise ValueError("Blocked MLP gradients require frozen feed-forward weights.")
        if hidden.shape[1] <= chunk_size:
            return original(layer, hidden)

        @mx.custom_function
        def bounded(x):
            x = detached_array(x)
            pieces = []
            for start in range(0, x.shape[1], chunk_size):
                value = original(layer, x[:, start:start + chunk_size])
                mx.eval(value)
                pieces.append(detached_array(value))
                mx.clear_cache()
            return mx.concatenate(pieces, axis=1)

        @bounded.vjp
        def backward(primals, cotangent, output):
            x = detached_array(primals)
            cotangent = detached_array(cotangent)
            pieces = []
            for start in range(0, x.shape[1], chunk_size):
                _, gradients = mx.vjp(lambda value: original(layer, value),
                                      [x[:, start:start + chunk_size]],
                                      [cotangent[:, start:start + chunk_size]])
                mx.eval(gradients)
                pieces.append(detached_array(gradients[0]))
                mx.clear_cache()
            return mx.concatenate(pieces, axis=1)

        return bounded(hidden)

    MLP.__call__ = call
    try:
        yield
    finally:
        MLP.__call__ = original


@contextlib.contextmanager
def materialized_layer_gradients():
    """Use reverse-mode layer VJPs without an outer full-model gradient trace."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.models.base import create_attention_mask

    original = nn.value_and_grad

    def dispatch(model, loss):
        if getattr(loss, "func", loss) is not prefix_completion_loss:
            return original(model, loss)
        policy = getattr(loss, "keywords", {}).get("policy", False)

        def value_and_grad(model, hidden, completion, weight):
            mask = create_attention_mask(hidden)
            inputs = []
            layers = model.layers[-8:]
            for layer in layers:
                inputs.append(hidden)
                hidden = layer(hidden, mask=mask)
                mx.eval(hidden)

            def head(value):
                return head_completion_loss(model, value, completion, weight, policy=policy)

            result, upstream = mx.value_and_grad(head)(hidden)
            mx.eval(result, upstream)
            flat_gradients = {}
            offset = max(0, len(model.layers) - 8)
            for index in reversed(range(len(layers))):
                layer, x = layers[index], inputs.pop()
                params = layer.trainable_parameters()
                flattened = tree_flatten(params)
                names = [name for name, _ in flattened]

                def compute(*values):
                    layer.update(tree_unflatten(list(zip(names, values[:-1], strict=True))))
                    return layer(values[-1], mask=mask)

                _, gradients = mx.vjp(compute, [value for _, value in flattened] + [x], [upstream])
                mx.eval(gradients)
                layer.update(params)
                for name, gradient in zip(names, gradients[:-1], strict=True):
                    flat_gradients[f"model.layers.{offset + index}.{name}"] = gradient
                upstream = gradients[-1]
                mx.clear_cache()
            return result, tree_unflatten(list(flat_gradients.items()))

        return value_and_grad

    nn.value_and_grad = dispatch
    try:
        yield
    finally:
        nn.value_and_grad = original


def file_digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prompt_tokens(tokenizer, messages: list[dict]) -> list[int]:
    return list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False, **TEMPLATE_ARGS,
    ))


def tokenize_demonstration(row: dict, tokenizer, split: str) -> dict:
    metadata = row["metadata"]
    if split not in {"train", "validation"} or metadata["split"] != split:
        raise PermissionError("Training and validation rows must stay in their own split.")
    expected_use = "supervised_training" if split == "train" else "validation_loss_only"
    if metadata["target_use"] != expected_use:
        raise PermissionError("Unexpected demonstration target use.")
    messages = row["messages"]
    if messages[-1]["role"] != "assistant" or "attachment://" in json.dumps(messages):
        raise ValueError("Expected a resolved assistant completion target.")
    prompt = prompt_tokens(tokenizer, messages[:-1])
    full = list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=False, **TEMPLATE_ARGS,
    ))
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError("Chat template is not an exact prefix of the supervised sequence.")
    return {"prompt_ids": prompt, "completion_ids": full[len(prompt):], "split": split,
            "case_id": metadata["case_id"], "action_type": metadata["action_type"], "weight": 1.0}


def prepare_data(data_dir: Path, model_path: Path, output_dir: Path, max_length: int = 24576) -> dict:
    """CPU tokenization; rejects oversized rows rather than truncating supervision."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"max_length": max_length, "template_args": TEMPLATE_ARGS, "splits": {}}
    for split in ("train", "validation"):
        rows, excluded, lengths = [], [], []
        for line in (data_dir / f"{split}.jsonl").read_text().splitlines():
            row = tokenize_demonstration(json.loads(line), tokenizer, split)
            length = len(row["prompt_ids"]) + len(row["completion_ids"])
            lengths.append(length)
            if length > max_length:
                excluded.append({"case_id": row["case_id"], "action_type": row["action_type"], "tokens": length})
            else:
                rows.append(row)
        if not rows:
            raise ValueError(f"No usable {split} rows.")
        (output_dir / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        ordered = sorted(lengths)
        manifest["splits"][split] = {
            "accepted": len(rows), "excluded": excluded, "max_tokens": max(lengths),
            "median_tokens": ordered[len(ordered) // 2], "source_sha256": file_digest(data_dir / f"{split}.jsonl"),
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def completion_loss(model, prompt, completion, weight, *, policy: bool = False):
    """Project only assistant positions into the 128k vocabulary to fit local RAM.

    Prefix hidden states condition predictions normally. Prompt/tool tokens never
    become prediction targets. Shapes (not array values) determine the slice.
    """
    import mlx.core as mx
    import mlx.nn as nn

    tokens = mx.concatenate([prompt, completion], axis=1)
    hidden = model.model(tokens[:, :-1])[:, prompt.shape[1] - 1:, :]
    logits = (model.model.embed_tokens.as_linear(hidden) if model.args.tie_word_embeddings
              else model.lm_head(hidden))
    ce = nn.losses.cross_entropy(logits, completion).astype(mx.float32)
    value = ce.sum() * weight if policy else ce.mean()
    return value, mx.array(completion.size)


def policy_loss(model, prompt, completion, weight):
    return completion_loss(model, prompt, completion, weight, policy=True)


def prefix_batches(model, dataset, **kwargs):
    """Materialize the frozen backbone before entering the gradient transform."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models.base import create_attention_mask

    frozen_layers = model.layers[:-8]
    if any(tree_flatten(layer.trainable_parameters()) for layer in frozen_layers):
        raise ValueError("The precomputed prefix must contain no trainable parameters.")
    for prompt, completion, weight in batches(dataset, **kwargs):
        tokens = mx.concatenate([prompt, completion], axis=1)[:, :-1]
        hidden = model.model.embed_tokens(tokens)
        mask = create_attention_mask(hidden)
        for layer in frozen_layers:
            hidden = layer(hidden, mask=mask)
            mx.eval(hidden)
        mx.eval(hidden)
        mx.clear_cache()
        yield hidden, completion, weight


def prefix_completion_loss(model, hidden, completion, weight, *, policy=False):
    from mlx_lm.models.base import create_attention_mask

    mask = create_attention_mask(hidden)
    for layer in model.layers[-8:]:
        hidden = layer(hidden, mask=mask)
    return head_completion_loss(model, hidden, completion, weight, policy=policy)


def head_completion_loss(model, hidden, completion, weight, *, policy=False):
    import mlx.core as mx
    import mlx.nn as nn

    hidden = model.model.norm(hidden)[:, -completion.shape[1]:, :]
    logits = (model.model.embed_tokens.as_linear(hidden) if model.args.tie_word_embeddings
              else model.lm_head(hidden))
    ce = nn.losses.cross_entropy(logits, completion).astype(mx.float32)
    return (ce.sum() * weight if policy else ce.mean()), mx.array(completion.size)


def batches(dataset, batch_size, max_seq_length, loop=False, comm_group=None, **kwargs):
    import mlx.core as mx

    if batch_size != 1 or (comm_group is not None and comm_group.size() != 1):
        raise ValueError("This memory-bounded local trainer supports batch size one on one device.")
    if not dataset:
        raise ValueError("Empty optimization dataset.")
    while True:
        for row in dataset:
            prompt, completion = row["prompt_ids"], row["completion_ids"]
            if not prompt or not completion or len(prompt) + len(completion) > max_seq_length:
                raise ValueError("Empty or oversized sequence; truncation is prohibited.")
            yield mx.array([prompt]), mx.array([completion]), mx.array(row.get("weight", 1.0))
        if not loop:
            break


def balanced_order(rows: list[dict], seed: int) -> list[dict]:
    """Round-robin action types so a short pilot sees coordination and completion."""
    rng = random.Random(seed)
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row["action_type"], []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    result = []
    while any(groups.values()):
        keys = sorted(groups)
        rng.shuffle(keys)
        result.extend(groups[key].pop() for key in keys if groups[key])
    return result


def train_adapter(
    model_path: Path, output_dir: Path, data_path: Path, *, parent: Path | None = None,
    validation_path: Path | None = None, mode: str = "sft", steps: int = 128,
    learning_rate: float = 2e-5, max_length: int = 24576, seed: int = 20260928,
    optimizer_state: Path | None = None,
) -> dict:
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm import load
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters

    if mode not in {"sft", "policy"}:
        raise ValueError("Unknown training mode.")
    if type(steps) is not int or steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Training steps and learning rate must be positive.")
    if parent and output_dir.resolve() == parent.resolve():
        raise ValueError("A new checkpoint must not overwrite its parent.")
    if (output_dir / "adapters.safetensors").exists():
        raise FileExistsError("Refusing to overwrite an existing checkpoint; resume through the experiment ledger.")
    if mode == "policy" and validation_path is not None:
        raise ValueError("Policy updates do not use supervised validation loss.")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    if not rows:
        raise ValueError("No training rows were supplied.")
    if mode == "sft":
        if any(row.get("split") != "train" for row in rows):
            raise PermissionError("Only explicitly marked training rows may receive SFT updates.")
        rows = balanced_order(rows, seed)
    else:
        if parent is None:
            raise ValueError("Policy optimization requires an explicit parent checkpoint.")
        parent_digest = file_digest(parent / "adapters.safetensors")
        if any(row.get("split") != "train" or row.get("policy_fingerprint") != parent_digest for row in rows):
            raise ValueError("Policy rows must be from the training split and this exact parent checkpoint.")
        if any(not math.isfinite(row.get("weight", float("nan"))) for row in rows):
            raise ValueError("Policy weights must be finite.")
        # One fresh rollout group, one accumulated optimizer update. No stale reuse.
        steps = len(rows)
    validation = ([json.loads(line) for line in validation_path.read_text().splitlines()]
                  if validation_path else None)
    if validation:
        if any(row.get("split") != "validation" for row in validation):
            raise PermissionError("Validation loss rows must be marked validation.")
        validation = balanced_order(validation, seed)[:8]
    mx.random.seed(seed)
    np.random.seed(seed)
    started = time.monotonic()
    model, _ = load(str(model_path))
    model.freeze()
    if parent is None:
        linear_to_lora_layers(model, 8, LORA_CONFIG)
    else:
        # Freeze the base before creating LoRA modules. Adapter loading itself
        # does not freeze base weights.
        load_adapters(model, str(parent))
    trainable_names = [name for name, _ in tree_flatten(model.trainable_parameters())]
    if not trainable_names or any(not name.endswith(("lora_a", "lora_b")) for name in trainable_names):
        raise RuntimeError("Only LoRA parameters may enter the optimizer.")
    configuration = {
        "fine_tune_type": "lora", "num_layers": 8, "lora_parameters": LORA_CONFIG,
        "base_model": str(model_path), "parent": str(parent) if parent else None,
    }
    (output_dir / "adapter_config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    optimizer = optim.Adam(learning_rate=learning_rate)
    if optimizer_state and optimizer_state.exists():
        optimizer.state = tree_unflatten(list(mx.load(str(optimizer_state)).items()))
        optimizer.learning_rate = learning_rate
    history = []

    class Callback(TrainingCallback):
        def on_train_loss_report(self, info):
            if not math.isfinite(info["train_loss"]):
                raise RuntimeError("Non-finite optimization loss; refusing this checkpoint.")
            history.append({"kind": "train", **info})
            (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")

        def on_val_loss_report(self, info):
            if not math.isfinite(info["val_loss"]):
                raise RuntimeError("Non-finite validation loss; refusing this checkpoint.")
            history.append({"kind": "validation", **info})
            (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")

    args = TrainingArgs(
        batch_size=1, iters=steps, val_batches=8, steps_per_report=min(8, steps),
        steps_per_eval=64, steps_per_save=64, max_seq_length=max_length,
        adapter_file=str(output_dir / "adapters.safetensors"), grad_checkpoint=False,
        grad_accumulation_steps=len(rows) if mode == "policy" else 1,
    )
    with memory_bounded_attention(), memory_bounded_mlp(), materialized_layer_gradients():
        from functools import partial

        train(model, optimizer, rows, validation, args=args,
              loss=partial(prefix_completion_loss, policy=mode == "policy"),
              iterate_batches=partial(prefix_batches, model), training_callback=Callback())
    if any(not bool(mx.all(mx.isfinite(value))) for _, value in tree_flatten(model.trainable_parameters())):
        raise RuntimeError("Non-finite adapter parameters; refusing this checkpoint.")
    mx.save_safetensors(str(output_dir / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    result = {
        "mode": mode, "steps": steps, "optimizer_updates": 1 if mode == "policy" else steps,
        "learning_rate": learning_rate, "seed": seed, "parent": str(parent) if parent else None,
        "parent_digest": file_digest(parent / "adapters.safetensors") if parent else None,
        "digest": file_digest(output_dir / "adapters.safetensors"),
        "data_digest": file_digest(data_path), "seconds": time.monotonic() - started,
        "peak_memory_bytes": mx.get_peak_memory(), "max_length": max_length,
        "attention_training": "exact causal query blocks of 256; serial block/layer VJP; eager optimizer",
        "frozen_prefix_layers": max(0, len(model.layers) - 8),
        "training_examples_seen": steps,
        "training_cases_seen": len({rows[index % len(rows)].get("case_id") for index in range(steps)}),
        "processed_prompt_tokens": sum(len(rows[index % len(rows)]["prompt_ids"]) for index in range(steps)),
        "optimized_completion_tokens": sum(len(rows[index % len(rows)]["completion_ids"]) for index in range(steps)),
        "trainable_parameters": sum(value.size for _, value in tree_flatten(model.trainable_parameters())),
        "validation_selection": "fixed final checkpoint; validation loss is diagnostic only",
        "history": history,
    }
    if mode == "policy" and result["digest"] == result["parent_digest"]:
        raise RuntimeError("Policy optimizer produced no checkpoint change.")
    (output_dir / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "sft", "policy"])
    parser.add_argument("--model", type=Path, default=ROOT / "training/models/llama3.1-8b-4bit")
    parser.add_argument("--data", type=Path, default=ROOT / "training/data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--optimizer-state", type=Path)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=24576)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_data(args.data, args.model, args.output, args.max_length)
    else:
        result = train_adapter(args.model, args.output, args.data, parent=args.parent,
                               validation_path=args.validation, mode="policy" if args.command == "policy" else "sft",
                               steps=args.steps, learning_rate=args.learning_rate,
                               max_length=args.max_length, seed=args.seed, optimizer_state=args.optimizer_state)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
