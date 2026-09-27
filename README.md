# DealRoom

A synthetic transaction environment for evaluating tool-using agents, with a research console for comparing models and inspecting tasks, trajectories, state, and verifier evidence. Inspect AI provides the native model integration, ReAct loop, tools, evaluation logs, limits, scorer, and full transcript viewer.

The current comparison, `dealroom-local-48-text-tools-v4`, runs **48 generated tasks through two downloaded local models: Qwen3 8B and Llama 3.1 8B, for 96 fresh planned attempts**. This is actual local inference through Inspect's native Ollama provider, with Inspect's built-in text tool emulation enabled identically for both models. Task prompts, tools, and emulation are unchanged from v3; the expanded episode and compute budgets are unchanged from v2. Version 4 corrects canonical scoring of operational cutoffs. The original six curated cases and eight scripted demonstrations remain available in the replay library and are excluded from model results.

**Evaluation status:** v4 runs a matched 96-attempt matrix under the corrected scoring policy. Check the experiment ID and recorded progress in `web/public/benchmark.json` and the console; no final totals are asserted here. The v1 partial run, v2 native-tool diagnostic, and interrupted v3 run are archived separately, and none of their outcomes are merged into v4. Within each experiment, tasks, model digests, prompts, tools, tool-calling format, scoring policy, and limits are frozen; individual outcomes are not retried or selected away.

## Llama failure investigation

The first 16 completed Llama v4 attempts were audited against raw model requests and responses, parsed calls, public facts, and deterministic replay. Thirteen explicitly waited to contractual expiry; three finished without a valid disposition. None hit an Inspect limit or infrastructure error. All 119 tagged calls matched parsed calls. The public facts were sufficient to solve every audited case. See `reports/llama-v4-forensics.md` for the case table and evidence.

Zero successful document or message reads did **not** mean zero evidence exposure: eight attempts received all current public resource bodies in failed-read responses. This inconsistent presentation is an interface weakness, but no evidence was dropped by transport. Seven expiry failures occurred after the seller facts had reached the model.

Six separate diagnostic trials tested two selected cases with clearer resource-read instructions, initially preloaded public evidence, or an explicit explanation of tool effects. All six remained objective failures, without compute limits. Clearer read instructions enabled Llama to execute the correct credit amendment in `synth-007`, but it reported proceed without committing a disposition. Even with explicit tool-effect guidance, it later waited to expiry after routing the correct amendment; on `synth-025`, it read every initial document and message but omitted the required extension. These probes show sensitivity to the interface and persistent planning/tool-contract mistakes. They do not establish a general capability ranking or prove a broken evaluator.

The six logs and two reports live under `logs/diagnostics/`; they are excluded from the model matrix. `uv run --extra openai python -m dealroom.diagnostics` reproduces the first two conditions on the selected cases; add `--conditions explicit_tool_semantics` for the follow-up condition. Run these only with exclusive access to Ollama. Each probe starts fresh and uses the unchanged task tools, engine, scorer, model digest, and numeric budgets. Only the declared public input supplement changes; no private witness is supplied. The first report inherits baseline configuration prose; its condition labels and recorded input supplements identify the actual ablations.

The batch was temporarily interrupted for these diagnostics. Qwen `synth-044` is retained as an **incomplete, unscored operator interruption**, with its original log and measured serving context; it is not retried or counted as model failure. Canonical cancelled logs are now distinguished from genuine execution errors. The matched run resumes with its original frozen conditions.

## Open the console

From the project directory:

```sh
npm --prefix web ci
npm --prefix web run build
npm --prefix web run preview
```

Open the printed preview URL, normally `http://127.0.0.1:4173/`. The supplied JSON exports are sufficient to view recorded results without starting models. To develop the frontend, use `npm --prefix web run dev` instead; its default port is 5173.

The console opens **Model results** when `benchmark.json` is present. Its main metric is **successes ÷ scored tasks**, where scored tasks are successes plus objective failures. With no scored tasks, it displays **—**. **Scored coverage** separately shows how many of the 48 planned tasks have a terminal score. Budget limits, execution errors, incomplete attempts, running attempts, and pending attempts are excluded from the pass-rate denominator and remain visible as distinct outcomes. Small or partial scored coverage cannot support a model comparison. Search the task matrix, filter by model or outcome, and select an available result to open its trajectory. **Details** shows recorded timing, tokens, reasons, the native log link, and counts of domain actions (`actions`), Inspect tool calls (`tool_calls`), tool errors (`tool_errors`), and model responses (`generations`). A model response need not invoke a tool, and a tool call rejected during schema validation need not reach the domain engine. Missing measurements remain missing; operational failures do not become reward-zero scored completions.

**Replay library** also retains the scripted failure and repaired continuation. Select `case-01`, inspect the unsigned amendment at step 7, then use **Compare** to inspect the repaired execution. Task details, grading predicates, public evidence, exact-revision signatures, and step controls remain available for each completed replay. Terminal reward describes the whole attempt; state and snapshot diagnostics describe the selected step.

Open canonical evaluation records in another terminal:

```sh
uv run inspect view --log-dir logs --host 127.0.0.1 --port 7575
```

Inspect links preserve paths beneath `logs/`, including `demo/`, `pilot/`, `pilot-emulation/`, and `benchmark/`. Full model conversations stay in the native Inspect viewer. The React frontend reads local JSON; it has no model-call endpoint or job-launch backend. **Import export** loads a replay JSON file in browser memory without uploading it. A running benchmark refreshes every 15 seconds. The runner updates both `web/public/` and an existing `web/dist/` so the static preview receives progress without repeated builds.

## Reproduce the local evaluation

Python 3.12 is selected by `.python-version`; the frontend supports Node `^20.19.0 || >=22.12.0`. Inspect AI is pinned to **0.3.271**. `uv.lock` and `web/package-lock.json` pin project dependencies. The local inference run uses **Ollama 0.31.1** on an **Apple M4 Max with 36 GB memory**.

| Model | Inspect identifier | Quantization | Downloaded size |
| --- | --- | --- | --- |
| Qwen3 8B | `ollama/qwen3:8b` | Q4_K_M | Approximately 5.225 GB |
| Llama 3.1 8B | `ollama/llama3.1:8b` | Q4_K_M | Approximately 4.921 GB |

Model digests and exact sizes are recorded in `benchmark.json`. The downloaded weights persist locally in `models/`; weights, dependency installations, and caches are excluded from the project archive. A fresh machine needs Ollama on its PATH and must download the models. Tags can change; resuming an existing evaluation requires its recorded digests and runtime version.

Install dependencies and generate the deterministic task suite:

```sh
uv sync --locked --extra openai
uv run python -m dealroom.synthesis --seed 20260927
```

The `openai` extra supplies the SDK used by Inspect's native Ollama integration; it does not send this evaluation to an OpenAI service or require an API key.

Start the loopback-only model server in a dedicated terminal from the project directory:

```sh
OLLAMA_HOST=127.0.0.1:11434 \
OLLAMA_MODELS="$PWD/models" \
OLLAMA_CONTEXT_LENGTH=40960 \
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_NO_CLOUD=1 \
ollama serve
```

This command runs a foreground server and installs no automatic startup service. In another terminal, download the models if they are not already present, then run the evaluation:

```sh
OLLAMA_HOST=127.0.0.1:11434 ollama pull qwen3:8b
OLLAMA_HOST=127.0.0.1:11434 ollama pull llama3.1:8b
uv run --extra openai python -m dealroom.benchmark \
  --emulate-tools --id dealroom-local-48-text-tools-v4
```

Run the same benchmark command, including `--emulate-tools` and the v4 ID, to resume. The runner reconstructs outcomes from canonical Inspect logs, preserves completed attempts including errors and limits, and continues remaining work. It rejects changes to model digests, Ollama/Inspect versions, task/engine/scorer hashes, suite fingerprints, settings, or the planned attempt matrix. Each canonical v4 log records `model_args={"emulate_tools": true}`; resume checks this against the selected format, and the report records `config.tool_calling="inspect_text_emulation"` and `config.score_policy="canonical_unscored_limits"`. An interrupted attempt with a canonical record is retained; it is not silently replaced by another outcome. A different experiment requires a new `--id`, `--output`, and `--runs-output`. The runner's bare default ID, `dealroom-local-48-native-v4`, selects a separate native-tool condition; reproduce this text-tool comparison with the explicit command above.

Serving-context measurements are persisted in the report separately from canonical outcome logs. Resume requires a matching historical measurement for each Ollama attempt with recorded tokens. An interruption after log creation but before that measurement is published can stop resume; restore a genuine report backup containing the observation if available, or keep the attempt unvalidated and start a separate experiment. The runner retains the original log and does not repeat that attempt.

Active experiment settings are identical for both models:

| Setting | Value |
| --- | --- |
| Planned attempts | One per task/model; 48 per model |
| Tool-calling format | Inspect built-in text emulation; `model_args={"emulate_tools": true}` |
| Task order | Same order, interleaved scenario families; models served in batches of eight |
| Temperature / generation seed | 0 / 20260927 |
| Thinking | Disabled |
| Context window | 40,960 tokens, checked against the serving runtime |
| Conversation compaction | Native deterministic `CompactionTrim`; threshold 28,672 tokens, `preserve=0.5`, `memory=False` |
| Output budget per generation | 2,048 tokens |
| Cumulative model-token budget | 2,000,000 tokens, including repeated input context |
| Turn / message budget | 128 turns / 512 messages |
| Wall-clock budget per attempt | 900 seconds |
| Domain-action budget | 256 actions per episode, including invalid attempts |
| Contractual expiry | End on actual irreversible expiry; objective failure with reward 0 |
| Scoring policy | `canonical_unscored_limits`; operational cutoffs are unscored in Inspect and the report |
| Model/tool concurrency | One sample, one connection, sequential tool calls |
| Generation cache / outcome retries | Disabled / none |

Task instructions, tool schemas, and Inspect's emulation instructions are the same for both models on each task. The [built-in emulation](https://inspect.aisi.org.uk/providers.html) presents tool schemas in a system prompt, requests calls in `<tool_call>` tags with `name` and `arguments`, and converts those model-produced responses into ordinary Inspect tool calls. It sends no native API tool declarations and returns tool results as user messages. The existing ReAct loop, typed argument validation, domain engine, and scorer remain in use; no custom provider or agent loop is added. Emulation does not guarantee a valid call or a solved task.

This is an explicit change of evaluation condition from the native-tool diagnostics. Native responses sometimes contained repeated prose describing a function call while the server returned no actual tool calls. The shared text format tests coordination through a supported alternative interface; its results cannot isolate base-model ability from runtime, template, parser, and interface effects. Model-specific chat templates remain part of their serving implementations. Timing includes local serving overhead and model loading; it is not a controlled hardware-speed benchmark. Temperature zero and a fixed seed do not guarantee bitwise reproducible inference across runtime or hardware changes.

The runner writes canonical logs to `logs/benchmark/dealroom-local-48-text-tools-v4/`, per-model reporting to `web/public/benchmark.json`, and completed replays plus separately counted issues to `web/public/runs.json`. Reports are derived from logs, not hand-authored performance data. Missing, incomplete, limited, or errored attempts have no numeric terminal reward and do not enter the pass-rate denominator; they remain in the planned matrix and reduce scored coverage.

The v4 scorer uses Inspect's built-in `Score.unscored()` for operational limits and unresolved episodes, consulting public `SampleLimitEvent` records and the environment's evaluated outcome. Both the canonical Inspect viewer and DealRoom report therefore retain these outcomes as unscored. A real completed objective failure, including irreversible contractual expiry without an Inspect cutoff, still receives reward 0.

The larger domain-action budget is an explicit episode override, not a mutation of the generated fixtures. Logs record both the base fixture fingerprint and effective episode fingerprint so the exporter reconstructs the exact 256-action configuration. Native `CompactionTrim` trims history at the declared threshold without a model-generated memory summary. Both models receive the same compaction settings.

## Archived experiments and infrastructure pilots

The original `dealroom-local-48-v1` run stopped after **52 resolved attempts: 6 successes, 18 scored failures, and 28 budget limits**. One additional attempt was cancelled by the operator, and 43 remained pending. This was a partial run, not a completed 96-attempt baseline. Its 53 native Inspect logs, `report.json`, `replays.json`, and snapshots of the three environment source files are retained beneath `logs/benchmark/dealroom-local-48-v1/`.

The expanded native-tool diagnostic, `dealroom-local-48-expanded-v2`, retained **four Llama records: one objective contractual-expiry failure, two budget limits, and one operator-cancelled incomplete attempt**, with 92 attempts still pending. No Qwen attempt ran, so v2 is not a model comparison. Its four canonical logs, report, replays, and source snapshots remain under `logs/benchmark/dealroom-local-48-expanded-v2/`.

The early shared text-tool run, `dealroom-local-48-text-tools-v3`, was stopped when a canonical scoring defect was found: Inspect could display a numeric zero for an operational limit even though DealRoom's report correctly kept that attempt unscored. The run retained **six Llama records: five objective failures and one operator-cancelled incomplete attempt**, with 90 pending. No Qwen attempt ran. Its exact logs, report, replays, and source snapshots remain under `logs/benchmark/dealroom-local-48-text-tools-v3/`.

Archived native Inspect logs predate the v4 scorer correction; numeric zeros shown on limited attempts in those old logs must not be interpreted as objective task failures. The old records are preserved, with operational outcomes distinguished in their reports. All three archives are excluded from v4, which starts all 96 task/model pairs afresh with canonical unscored cutoffs. It does not replace archived records or rerun only selected failures.

Two original infrastructure pilots remain in `logs/pilot/`. Two additional **unscored** emulation smoke attempts are retained in `logs/pilot-emulation/20260927T180401Z/`. On the same curated development case, both Llama and Qwen produced actual model-backed tool events and three recorded domain actions. These smoke attempts verified transport and provenance with scoring disabled; they are not capability scores and all pilots are excluded from per-model results.

## Task suite and provenance

The **48 tasks are programmatically synthesized**, using eight behavioral families with six deterministic variants each. They are parameterized transaction scenarios, not model-generated facts or scraped transactions. Seed 20260927 controls variations in costs, credit thresholds, authority, evidence, signature latency, deadlines, parties, and properties.

| Family | Tasks | Coordination requirement |
| --- | --- | --- |
| Quote discovery | `synth-001`–`synth-006` | Obtain specialist evidence before resolving a repair credit. |
| Straightforward execution | `synth-007`–`synth-012` | Execute an authorized credit with complete evidence. |
| Partial signature retry | `synth-013`–`synth-018` | Recover the missing signature on the existing exact revision. |
| Revised credit execution | `synth-019`–`synth-024` | Preserve the old executed terms until the changed revision executes. |
| Deadline extension | `synth-025`–`synth-030` | Execute an extension before completing slower credit signatures. |
| Cancellation authority | `synth-031`–`synth-036` | Obtain express authority and cancel when available credit is insufficient. |
| Credit authority refresh | `synth-037`–`synth-042` | Refresh the buyer's authority before taking actions outside its initial scope. |
| Boundary timing | `synth-043`–`synth-048` | Coordinate signatures and disposition at a tight timing boundary. |

Each family has two development and four evaluation variants: 16 development and 32 evaluation tasks. The matched evaluation uses all 48. `cases/generated/manifest.json` records generator version, seed, task fingerprints, public initial-state fingerprints, and private validation results. Every task is validated by executing a private witness through the unchanged transition engine and objective scorer. Witness actions, expected outcomes, and private counterpart implementation are never model inputs. Witness success establishes satisfiability, not model performance.

The original curated collection remains separate:

| ID | Split | Scenario |
| --- | --- | --- |
| `case-01` | Development | Flagship: $12,000 eligible cost, maximum $5,000 residual, $8,000 credit; unsigned execution trap. |
| `case-02` | Development | Straightforward authorized execution. |
| `case-03` | Evaluation | Partially signed envelope with a disclosed retry route. |
| `case-04` | Evaluation | Executed $6,000 revision and unsigned $8,000 revision. |
| `case-05` | Evaluation | Credit signatures require an executed deadline extension. |
| `case-06` | Evaluation | Insufficient seller concession requires express cancellation authority. |

The eight original demonstrations are six fixture witnesses and a paired flagship failure/repaired continuation. Seven succeed and the deliberately flawed script fails. These are explanatory scripts using Inspect's `mockllm/model`, not a model benchmark. Their token accounting is not local-model inference usage. `uv run dealroom demo` regenerates the scripted bundle; the supplied logs already contain these records, so rerunning it is unnecessary for viewing or resuming the matched benchmark.

## Engine and reward

All names, properties, messages, organizations, and transaction rules are synthetic. Amounts use integer cents and timestamps require time zones. Read and finish consume no simulated time; requests, drafting, routing, and disposition take five minutes. Structural and authority checks run before coordination, scheduled events are processed, and the action is revalidated at completion. Rejected actions have no contractual effects, but elapsed time and intervening events remain. Curated fixtures declare 80 domain attempts and generated fixtures declare 64. The expanded benchmark explicitly grants 256 actions per episode while retaining the unchanged base fixtures and their fingerprints. Invalid actions count toward the effective limit; exhausting this compute budget is unscored.

Wait processes events in stable timestamp/sequence order. Events at the exact deadline precede expiry; a disposition completing then can commit. An executed extension updates pending envelope deadlines. An informal message or draft cannot extend a deadline. Terminal disposition or expiry prevents further business changes. In the expanded experiment, actual irreversible contractual expiry ends the episode immediately as an objective task failure, without requesting more actions or inventing a `finish` call. Other finish requirements are unchanged; the generic engine still permits reading and final reporting after business termination.

Revisions are immutable, and required parties must sign that exact revision in time. Only counterpart events create signatures. Drafting preserves the old operative agreement; a newly executed credit replaces rather than adds to it. Duplicate requests do not duplicate replies, signatures, credits, or reward. A missing signer can be retried after its earlier response arrives. Authority is scoped and checked at action completion. Initialization history is replayed through the same engine and does not consume the new attempt's action budget.

Binary terminal reward checks acceptable disposition, historical authority, executed terms, exact-revision signatures, deadline, buyer constraints, and structured final claims. Reasons include evidence IDs and timestamps. Free text provides context; structured claims are scored without an LLM judge. Calling `finish` cannot create execution or a disposition. Actual contractual expiry receives reward 0 because the environment objective has become impossible, even if no `finish` call occurred. Its replay retains the real final action and expiry evidence. A wall-clock, token, message, turn, or domain-action limit is instead an operational cutoff with no terminal reward, as are errors and otherwise unfinished attempts.

Every sample receives a fresh engine and lock. The agent sees public observations and seven typed tools: `read`, `request`, `draft_amendment`, `send_for_signature`, `wait_until`, `set_disposition`, and `finish`. The runner exposes no filesystem, shell, arbitrary network, evaluator tools, private witnesses, or targets. This observation boundary does not prevent an unrelated local process from reading repository files. Contractual state and reviewer diagnostics shown in the UI come from Python; JavaScript formats and selects records.

## Other evaluations and exports

Inspect also accepts configured cloud providers; OpenAI/Anthropic SDK installation was checked, but credentialed cloud inference has not been verified. With a provider key already available in the shell and `MODEL` set to its Inspect identifier:

```sh
uv run --extra openai inspect eval dealroom/task.py \
  -T case_id=case-01 --model "$MODEL" \
  --token-limit 100000 --message-limit 120 --time-limit 180 \
  --log-dir logs/model
```

For Anthropic use `--extra anthropic`. Use `-T suite=synthetic` for generated tasks, and optionally `-T split=development`, `-T split=evaluation`, or a matching `-T case_id=synth-001`. Without task arguments, the generic task runs the original six cases. Its defaults are 100,000 total tokens, 120 messages, 180 seconds, and 2,048 output tokens per generation; the matched local runner explicitly overrides these with the settings above.

Export completed samples from native logs:

```sh
uv run dealroom export logs/model/*.eval --output web/public/model-runs.json
```

Select that file with **Import export**. Export uses Inspect's public log API, replays actions through the engine, verifies fixture fingerprints, and checks equality with the canonical score. Changed fixtures fail explicitly. The JSON keeps errors, limits, incomplete samples, and cancellations separate from completed replay runs. Unlabelled mock-provider exports remain labeled validation. No second model transcript is created.

## Implementation and checks

- `dealroom/domain.py`, `score.py`: typed state, transitions, events, and evidence predicates.
- `dealroom/task.py`: Inspect tools, native ReAct solver, sample isolation, and scorer.
- `dealroom/synthesis.py`: seeded task generation and private satisfiability validation.
- `dealroom/benchmark.py`: matched local inference, resume checks, canonical reporting, and live exports.
- `dealroom/demo.py`, `witnesses.py`: scripted examples and replay export.
- `cases/`, `cases/generated/`: curated fixtures and the generated suite with its manifest.
- `web/`: React/TypeScript console, per-model results, replay inspection, and local import.
- `tests/`: domain/adversarial rules, synthesis invariants, real Inspect integration, and reporting/resume checks.

```sh
uv run pytest -q
uv run ruff check dealroom tests
npm --prefix web run build
```

Tests cover authority and deadline boundaries, immutable revision/signature behavior, duplicate actions, limits, sample isolation, public-only observations, witness satisfiability, fixture/score consistency, and outcome classification without dropping failed or unfinished attempts. Reporting checks enforce the complete 96-attempt matrix and matching tool-emulation provenance. Frontend validation checks the scored-only pass-rate denominator, separate planned coverage, the no-score state, missing measurements, contradictory rewards, duplicate attempts, model/task references, and nested native-viewer links. The corrected v4 implementation passes 201 tests and Ruff checks. Tests exercise canonical unscored token, turn, message, time, and domain-action cutoffs, native-log round trips, aggregate exclusion, and valid outcomes at the final allowed action. Final browser verification and run totals are recorded after evaluation completes.

## Interface reference

The console uses public patterns from Idler's [ShelfLife sample viewer](https://idler.ai/collections/shelflife/sample) and [E-Sim rollout viewer](https://idler.ai/collections/shelflife-e-sim/sample): neutral surfaces, compact task navigation, prompts, tool schemas, numbered trajectories, reward, and grading evidence. This is an original DealRoom interface, not a verified reproduction of Idler's private internal UI or an affiliated product. It does not invent reasoning traces, training progress, or model-performance records.

## SilverKey source ledger

Reference: [SilverKey-Inc.](https://github.com/jaycewalzer1/SilverKey-Inc.) at requested and actual commit `4e30020db784144580d92307f4a658e22862ffbe`. The trailing period is part of the repository name. Applicable `AGENTS.md` was read; production scripts were not run and the reference checkout remained unchanged.

SilverKey's root manifest declares **UNLICENSED**, with no standalone LICENSE/COPYING/NOTICE found at that revision. DealRoom copies no SilverKey code, component, production text/data, or dependency. The following are conceptual adaptations implemented as original code and synthetic materials; no permission terms are inferred.

| Inspected SilverKey path | Concept used; changes/exclusions |
| --- | --- |
| `Server/app/services/transactions/insurance/items.py` | Inspection → specialist evidence → repair negotiation → proceed/cancel. Original case prose; omit production URL import and checkoff-relative reminders. |
| `Server/app/services/transactions/offer/items.py`, `escrow/items.py` | Contingency, amendment, parties and deadline vocabulary only; omit financing, title, funds and closing catalogs. |
| `Server/app/services/transactions/checklist_support/checklist_rules.py` | Pure helpers inspected; none needed/copied. No checklist gate bypasses or checkoff-as-evidence. |
| `Server/app/models/documents/agreement.py`, `agreement_revision.py`, `agreement_participant.py`, `agreement_link.py`, `agreement_event.py` | Original immutable revisions, required signers, routing, evidence links and events. Omit Flask/SQLAlchemy/DocuSign; strengthen signatures to bind exact revisions. Highest draft/status/audit rows are not execution proof. |
| `Client/packages/utils/transaction/agreement/contextualAgreementStatus.ts` | Status/routing vocabulary inspected; no function copied. Python-exported authoritative records replace application status shortcuts. |
| `Client/packages/features/checklists/components/roadmap/`, including `BuyerRoadmapChecklistItemCard.tsx` | Compact ordered state and evidence hierarchy, rebuilt with local React props and HTML/CSS; omit hooks, contexts and component packages. |
| `Client/packages/features/documents/components/agreement/AgreementStatusBadge.tsx`, `AgreementDetailModal.tsx` | Original compact badges and document/signer detail composition; omit authentication, integrations, modal infrastructure and generated model imports. |

Harness references: [Inspect agents](https://inspect.aisi.org.uk/agents.html), [tools](https://inspect.aisi.org.uk/tools.html), [native viewer](https://inspect.aisi.org.uk/log-viewer.html), and [task views](https://inspect.aisi.org.uk/task-views.html). APIs were checked against the installed pinned release. No second harness, speculative training adapter, or arbitrary React embedding in Inspect was added.

## Scope of the results

The generated tasks span eight designed behavioral families; variants within a family are related. This is a small synthetic evaluation of two locally quantized models under one constrained configuration, not evidence of broad generalization, independent real-world sampling, or learning improvement. Development/evaluation labels do not establish contamination-proof holdouts. All transaction rules are explicit fictional assumptions, not jurisdictional law. Case plus actions is deterministic; model output need not be. There is no stochastic counterpart or business-day calendar.

The authored scripted failure explains an execution boundary and is not a baseline for model comparison. Real local-model outcomes are reported separately and preserved whether successful or not. The frontend is a local research inspection tool; operating real transaction services, executing signatures, publishing, or sending external messages is outside this project.
