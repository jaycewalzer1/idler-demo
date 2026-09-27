# Local learning experiment

This experiment compares four conditions of the same Llama 3.1 8B Instruct MLX
4-bit checkpoint: original workflow, clarified workflow, clarified workflow with
supervised LoRA training, and the supervised checkpoint with environment-reward
policy updates. Its baseline is separate from the existing Ollama comparison:
MLX 4-bit and Ollama Q4_K_M are different quantizations.

The curriculum contains 256 training cases, 32 validation cases, and 64 sealed
test cases. The sealed cases combine dependencies absent from the training and
validation families. Every fixture has an engine-verified successful witness.
Witnesses are teacher data or private satisfiability checks, never measured model
performance or evaluation prompt material. The original 48 inspected scenarios
are development data and are excluded from the new splits.

## Reproduction

From the project root:

```sh
uv sync --project training --locked
training/.venv/bin/hf download mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --revision 241a666dad6cb93c8ff213d39a7f34a36bf26db4 \
  --local-dir training/models/llama3.1-8b-4bit
training/.venv/bin/python -m dealroom.learning_data
training/.venv/bin/python -m dealroom.learning_train prepare --output training/tokenized
training/.venv/bin/python -m dealroom.learning_experiment --prepare
training/.venv/bin/python -u -m dealroom.learning_experiment --run
```

The base model is downloaded locally under `training/models/llama3.1-8b-4bit`.
`model_source.json` records the repository and immutable revision; a new machine
must fetch that revision before running. Do not substitute a moving model tag.
The runner checks source, dataset, and checkpoint fingerprints and resumes its
recorded phases. It waits for the existing Ollama benchmark to complete before
using the GPU. Keep the machine awake and the process running.

Open the console's **Training experiment** tab for live progress. Inspect is the
only agent harness: all sampled trajectories use its ReAct loop, typed tools,
canonical scorer, and native logs. The deterministic DealRoom engine and binary
evaluator are unchanged. Native logs and checkpoint metadata provide the audit
trail; progress-only or incomplete runs are not evidence of improvement.

## Supervised stage

The exporter executes successful scripted training trajectories through Inspect
and constructs the exact OpenAI-compatible text-tool wire format used for model
inference. It fully resolves transcript attachments. The export contains 3,064
training decision rows and 380 separate validation decision rows. Only the final
assistant completion is supervised; tool receipts and earlier messages remain
conditioning context. Rows over the declared length are excluded explicitly,
never silently truncated. All current rows fit.

The local pilot uses 128 optimizer steps, batch size one, Adam at 2e-5, rank-8
LoRA on query/value projections in the final eight layers, and gradient
checkpointing. Only adapter parameters can change. Completion-position logits
avoid projecting the entire long prompt into the large vocabulary. The checkpoint
is the fixed final step; validation loss is diagnostic. This is a short pilot,
not a full epoch over every decision. The saved metadata records actual examples,
cases, prompt tokens, completion tokens, wall time, memory, and parameter counts.

## Environment-reward stage

The RL pilot uses on-policy REINFORCE with a group-relative baseline. Each group
contains fresh stochastic Inspect episodes for one training case. Population-
standardized rewards weight the mean log probability of the actor tokens in each
scored trajectory. One gradient accumulation over a group produces one update;
old rollouts are not reused. This implementation does not claim PPO or GRPO.

Rollouts use temperature one without probability truncation or penalties. The
MLX server shim records exact prompt and sampled token IDs, their log probabilities,
and the loaded checkpoint identity. Output text is never re-tokenized to invent
the sampled actions. The extractor checks the chat template, API usage, and policy
fingerprint. A small compatibility fix adds token strings to MLX's logprob records
for Inspect; generation remains in the official MLX server. Explicit adapter
selection and actual loaded-model provenance prevent silent base-model fallback.

Training rewards combine terminal success with an explicitly bounded semantic
progress term of at most 0.2 in magnitude. It uses actual valid transaction state,
not prose, read counts, or repeated tool calls. This shaping changes the training
objective; no policy-invariance claim is made. Evaluation remains strictly binary.
Limits, errors, and incomplete episodes are excluded from gradients. Equal-reward
groups produce no update and are recorded as such. No improvement is assumed.

## Sealed evaluation and interpretation

Checkpoint selection and the workflow freeze before any sealed model evaluation.
All four conditions then receive the same cases, runtime, sampling settings, and
resource budgets. Resource limits remain unscored and visible; they are never
reported as evidence of low capability. Compare both success rate and coverage.
The report also preserves token use, elapsed time, native trajectories, and
checkpoint lineage. A training claim requires measured performance beyond the
supervised checkpoint on held-out cases; lower training loss alone is insufficient.

One synthetic suite, one local model size, and a short pilot cannot establish a
general model ranking. Composition holdouts test transfer within this environment;
they do not establish performance on real transactions or other environments.

## Checks

```sh
training/.venv/bin/python -m pytest -q
training/.venv/bin/ruff check dealroom tests
npm --prefix web run build
```

Small CPU integration fixtures verify the real training and serving paths without
loading the 8B weights or competing with an active benchmark. They are software
checks and never enter experiment performance totals.
