"""The bounded training path must preserve all causal context and gradients."""
import pytest


@pytest.mark.parametrize("dtype_name,tolerance", [("float32", 3e-5), ("float16", 5e-3)])
def test_query_blocks_match_full_attention_values_and_all_input_gradients(dtype_name, tolerance):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models import llama

    from dealroom.learning_train import memory_bounded_attention

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(32)
        q = mx.random.normal((1, 4, 19, 8)).astype(getattr(mx, dtype_name))
        k = mx.random.normal((1, 2, 19, 8)).astype(getattr(mx, dtype_name))
        v = mx.random.normal((1, 2, 19, 8)).astype(getattr(mx, dtype_name))
        original = llama.scaled_dot_product_attention

        def output(q, k, v):
            return llama.scaled_dot_product_attention(q, k, v, cache=None, scale=8**-0.5, mask="causal")

        def objective(q, k, v):
            return mx.square(output(q, k, v)).sum()

        expected = output(q, k, v)
        gradients = mx.grad(objective, argnums=(0, 1, 2))(q, k, v)
        mx.eval(expected, gradients)
        with memory_bounded_attention(4):
            actual = output(q, k, v)
            actual_gradients = mx.grad(objective, argnums=(0, 1, 2))(q, k, v)
            mx.eval(actual, actual_gradients)
        assert llama.scaled_dot_product_attention is original
        assert bool(mx.allclose(actual, expected, atol=tolerance, rtol=tolerance))
        for actual_gradient, gradient in zip(actual_gradients, gradients, strict=True):
            assert bool(mx.allclose(actual_gradient, gradient, atol=tolerance, rtol=tolerance))
    finally:
        mx.set_default_device(previous)


def test_frozen_prefix_preserves_adapter_loss_and_gradients():
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.llama import Model, ModelArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from dealroom.learning_train import (
        LORA_CONFIG,
        completion_loss,
        materialized_layer_gradients,
        memory_bounded_attention,
        memory_bounded_mlp,
        prefix_batches,
        prefix_completion_loss,
    )

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(11)
        model = Model(ModelArgs(model_type="llama", hidden_size=16, intermediate_size=32,
                                num_hidden_layers=9, num_attention_heads=4, num_key_value_heads=2,
                                rms_norm_eps=1e-5, vocab_size=32, tie_word_embeddings=False))
        model.freeze()
        linear_to_lora_layers(model, 8, LORA_CONFIG)
        prompt, completion, weight = mx.array([[1, 2, 3, 4]]), mx.array([[5, 6]]), mx.array(1.0)
        expected, gradients = nn.value_and_grad(model, completion_loss)(model, prompt, completion, weight)
        mx.eval(expected, gradients)
        prepared = next(prefix_batches(model, [{"prompt_ids": [1, 2, 3, 4], "completion_ids": [5, 6]}],
                                       batch_size=1, max_seq_length=32))
        with memory_bounded_attention(4), memory_bounded_mlp(2), materialized_layer_gradients():
            actual, actual_gradients = nn.value_and_grad(model, prefix_completion_loss)(model, *prepared)
            mx.eval(actual, actual_gradients)
        assert float(actual[0]) == pytest.approx(float(expected[0]), rel=1e-6)
        flat = dict(tree_flatten(gradients))
        for name, gradient in tree_flatten(actual_gradients):
            assert bool(mx.allclose(gradient, flat[name], atol=1e-5, rtol=1e-5)), name
        import mlx.optimizers as optim

        optim.Adam(learning_rate=1e-5).update(model, actual_gradients)
        mx.eval(model.parameters())
    finally:
        mx.set_default_device(previous)
