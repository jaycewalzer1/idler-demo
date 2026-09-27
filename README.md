# DealRoom

A synthetic real estate transaction environment for evaluating tool-using agents, with an evidence-first replay inspector. The agent coordinates inspection findings, repair-credit amendments, signatures, and deadlines. Inspect AI supplies the model integrations, ReAct loop, tool execution, evaluation logs, limits, scorer integration, and full transcript viewer.

**The supplied demonstration is offline and scripted.** Six fixture witnesses and one repaired continuation succeed; the deliberately flawed flagship script fails. These are fixture validation and an explanatory demonstration, not model-performance results. No live provider evaluation was run because no provider credentials were available in the build environment.

## Run locally

Tested with Python 3.12.5, uv, Node 22.18.0, and npm 10.9.3. Python 3.12 is selected by `.python-version`; the frontend supports Node `^20.19.0 || >=22.12.0`. Inspect AI is pinned to **0.3.271**. `uv.lock` and `web/package-lock.json` pin the resolved dependencies.

From this project directory:

```sh
uv sync --locked
uv run pytest -q
uv run dealroom demo
npm --prefix web ci
npm --prefix web run dev
```

Open the Vite URL printed in the terminal, normally `http://127.0.0.1:5173/`. The checked-in `web/public/runs.json` already works without Python, model credentials, or a custom server; `dealroom demo` regenerates it through Inspect's actual task/tool/scorer path using `mockllm/model`.

Build and preview the static page:

```sh
npm --prefix web run build
npm --prefix web run preview
```

Open the preview URL, normally `http://127.0.0.1:4173/`. The frontend reads local JSON only. It has no provider keys, model calls, job-launch API, or custom backend. File selection loads a new export in browser memory; nothing is uploaded to a service. Open malformed or empty files to see explicit error/empty states.

Open the canonical evaluation records in Inspect's native viewer:

```sh
uv run inspect view --log-dir logs/demo --host 127.0.0.1 --port 7575
```

The delivered archive also includes `logs/demo/` from the verified build. Regeneration creates new uniquely named Inspect logs; it does not delete older logs. The generated bundle references only its current eight runs. The native sample grid is configured with terminal score labels, tokens, duration, errors, and limits. Its transcripts are not reimplemented in React.

## Two-minute walkthrough

1. Select **18 Alder Lane** and **Scripted failure**. At the initial step, effective credit is $0 and the specialist quote is not yet public.
2. Choose **Go to branch**. The shared prefix obtained the $12,000 specialist quote, negotiated an informal $8,000 credit, and created revision-001. Both required signatures are still absent.
3. Choose **Final step**. The flawed script attempted to proceed and confidently claimed execution. The engine rejected proceeding: $0 credit leaves $12,000 residual against the buyer's $5,000 limit. Reviewer-only diagnostics expose the unsupported signature and final-report claims.
4. Inspect the failed/repaired comparison, then switch to **Repaired script** and its final step. The continuation routed the same immutable revision, waited for both signatures, read the envelope, and proceeded. Effective credit is $8,000; residual is $4,000; terminal reward is 1.
5. Move backward and forward to see documents, messages, signer states, and the simulated clock change. The common prefix is reset and replayed, with full state equality asserted at the branch. Comparison positions align by action offset after that boundary, not by wall-clock time.

All contractual values, execution statuses, signature records, and diagnostics shown by the page come from Python snapshots. JavaScript formats and selects records; it does not decide legal validity. The terminal result badge describes the entire run; the state cards and reviewer predicates describe the selected step. All names, properties, messages, rules, and organizations are synthetic.

## Cases

| ID | Split | Validation scenario |
| --- | --- | --- |
| case-01 | Development | Flagship: $12,000 eligible cost, maximum $5,000 residual, $8,000 seller credit; Oct 5, 2026 09:00 to Oct 9 17:00, America/New_York. |
| case-02 | Development | Straightforward authorized execution with complete evidence and enough time. |
| case-03 | Evaluation | Opens with a partially signed envelope. The disclosed seller availability and retry route permit obtaining the missing signature. |
| case-04 | Evaluation | Opens with executed $6,000 revision-001 and unsigned $8,000 revision-002. The older credit remains operative until the changed exact revision is signed. |
| case-05 | Evaluation | Normal credit-signature latency misses the original deadline. An executed extension creates enough time. |
| case-06 | Evaluation | The disclosed seller concession cannot satisfy the buyer's constraint. Request express cancellation authority and cancel in time. |

Cases 03/04 have short fixture initialization action sequences, replayed through the same transition function at reset. Their historical signatures are real engine records, not manually assigned status labels. Agent attempts start at zero; initialization history is retained for evidence and authority checks. Public handoff materials describe the initial records, while fixture initialization actions remain outside the model input. Witness scripts are case-specific fixture validation, never a third general-purpose solver.

## Domain and evaluation rules

The explicit fictional rules appear in every public case policy. Amounts use integer cents; timestamps require time zones. Read and finish take zero simulated time. Requests, drafting, routing, and disposition take five minutes. Structural and authority checks run before coordination; events during the interval are processed, then the action is revalidated at completion. Proceed readiness uses evidence available at completion, allowing a last signature arriving during those five minutes to count. Rejected actions have no contractual effects, although elapsed time and intervening events are retained. Every call counts toward the 80-attempt limit.

Wait advances to the requested later time. Events use stable timestamp/sequence order. Events at the exact deadline are processed before expiry; a disposition completing then can commit. An executed extension updates pending envelope deadlines. An informal message or draft cannot extend a deadline. A terminal disposition or expiry prevents further business changes; read and final report remain available.

Revisions are immutable. Required parties must sign that exact revision in time; only counterpart events create signatures. A new draft preserves the old operative agreement. The most recently executed credit replaces earlier credit rather than adding amounts. Identical requests do not duplicate replies, revisions, signatures, credits, or reward. A missing signer can be retried once its earlier response has arrived. Buyer authority is scoped and checked at action completion; later instructions affect subsequent actions without retroactively invalidating earlier authorized activity.

Terminal reward is binary. The scorer checks acceptable disposition, historical authority, applicable executed terms, exact-revision signatures, deadline, buyer constraint, and structured final claims. Failures include record IDs and timestamps where applicable. The free-text report provides context; only its structured contractual claims are scored, without an LLM judge. Finishing cannot create execution or a disposition. Cancellation passes only where the case objective allows it.

Each sample gets a fresh engine and lock in its solver closure. Tool calls are explicitly sequential; provider parallel calls are disabled and additionally serialized. Inspect handles cross-sample concurrency. The model sees a public case index and seven typed tools: `read`, `request`, `draft_amendment`, `send_for_signature`, `wait_until`, `set_disposition`, and `finish`. No filesystem, shell, arbitrary network, or evaluator tool is exposed. No targets, witness scripts, private event payloads, or counterpart implementation are supplied to the model. This boundary applies to the supplied runner, not an arbitrary process that can read the repository.

The default limits are 100,000 total model tokens, 120 messages, 180 seconds per sample, 2,048 output tokens per generation, and 80 domain attempts. Tool receipts return compact public indexes; complete evidence is discoverable by typed resource ID. Inspect's built-in loop uses `finish` as its submit tool. A small schema adapter preserves nested discriminated-union constraints in the pinned release's public `ToolParam` format.

## Evaluate a configured model

Use an Inspect provider/model identifier supplied by you; no current model ID is hardcoded. Provider SDKs are optional and locked. For OpenAI or Anthropic, respectively:

```sh
uv sync --locked --extra openai
# Or: uv sync --locked --extra anthropic
```

Make the provider's usual API key available in your shell. Set `MODEL` to your chosen provider/model identifier. Begin with one development sample and the explicit budget below:

```sh
uv run --extra openai inspect eval dealroom/task.py \
  -T case_id=case-01 --model "$MODEL" \
  --token-limit 100000 --message-limit 120 --time-limit 180 \
  --log-dir logs/model
```

For Anthropic, change `--extra openai` to `--extra anthropic`. Other providers can use Inspect's integrations after installing their required SDK; those optional SDKs were not validated here. To evaluate a split, replace `-T case_id=case-01` with `-T split=development` or `-T split=evaluation`. Omitting task arguments runs all six cases. This is a time/token budget, not a dollar guarantee; provider pricing determines cost.

These CLI flags and task loading were exercised with Inspect's mock provider. Provider SDK installation/import was verified; credentialed live inference was **not** verified. The offline regression suite exercises actual ReAct generation requests and tool calls using Inspect's supported mock-model outputs, not a replacement harness or provider adapter.

Export completed samples from native logs:

```sh
uv run dealroom export logs/model/*.eval --output web/public/model-runs.json
```

Open that file using **Open export**, or export to `web/public/runs.json` to replace the default bundle. Export uses Inspect's public `read_eval_log` API and replays recorded actions through the same engine. It verifies the fixture fingerprint and compares replay reward with the canonical Inspect score. Exporting against a changed fixture fails explicitly. Compact actions/events and a fixture hash live in per-sample Inspect metadata; no second model transcript is created.

Exports retain total planned samples, completed domain outcomes, successes, failures, execution errors, budget exhaustion, and incomplete counts, plus run-level errors/cancellations. Errors, exhausted budgets, and unfinished attempts are excluded from ordinary completed-run replays and listed as issues, preserving the denominator. Supplied scripts are clearly labeled; an unlabelled mock-provider export is also labeled validation, never live-model performance.

## Implementation and verification

- `dealroom/domain.py`: typed fixtures/actions/records, reset, public observations, scheduled events, and one transition function.
- `dealroom/score.py`: explicit evidence predicates and diagnostics.
- `dealroom/task.py`: sample-local Inspect tools, built-in ReAct configuration, scorer, and native viewer configuration.
- `dealroom/witnesses.py`: known-fixture action sequences and flagship shared prefix/alternate suffixes.
- `dealroom/demo.py`: offline Inspect runs and derived JSON exports.
- `cases/`: six validated, self-contained synthetic JSON fixtures.
- `web/`: React/TypeScript, Vite, original HTML/CSS, local state, file loading, and shipped run export.
- `tests/`: domain, adversarial, and real Inspect integration checks.

Run all checks:

```sh
uv run pytest -q
uv run ruff check dealroom tests
npm --prefix web run build
```

Build verification on September 27, 2026: **53 tests passed**, Ruff passed, and the production frontend built successfully. Eight canonical Inspect logs contain seven successful scripted outcomes and the expected flagship failure, with zero execution errors, limits, or incomplete attempts.

Verified behavior includes every witness and failed/repaired result; prefix equality; initial trap records; old revision and missing signatures; expired/historical authority and authority changes during coordination; late execution; duplicate requests; exact-deadline ordering; extension effects; action limits; concurrent sample isolation; actual public-only model requests; rejected final-claim correction; fixture-change rejection; score/export equality; error and limit denominators; and all seven tools through Inspect. Browser checks covered desktop and 390px widths, default traces, uploads, malformed/empty exports, keyboard navigation, changing evidence, timeline controls, and comparison switching. No browser console errors were observed.

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

## Limits of the demonstration

These are six curated examples, including two development and four evaluation cases. They are not contamination-proof holdouts, a statistically meaningful benchmark, evidence of broad generalization, or evidence of learning improvements. All transaction rules are explicit fictional assumptions, not jurisdictional law. There is no business-day calendar or stochastic counterpart; case plus actions is deterministic, while live LLM outputs need not be. The failure policy is deliberately authored, not a fair baseline or a manufactured model failure. Reviewer JSON and fixture files contain information an unrestricted local process could read. The frontend is a local replay inspection tool, not a consumer transaction application. Publishing, sending messages to Ivan, real signatures, and operating transaction services are outside this project.
