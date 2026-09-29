# DealRoom: why Llama is failing

Investigation completed September 27, 2026. Scope: the first 16 completed Llama v4 attempts and six separate selected-case diagnostics. The broader 96-attempt comparison is continuing.

**Finding:** the recorded failures are supported by the actual model actions. No dropped calls, missing essential public facts, or scorer rejection of a correct solution was found in the audited records. The tests show sensitivity to interface wording and continuing mistakes in planning and following tool contracts; they do not isolate general model capability.

- Thirteen of the sixteen original attempts explicitly waited to contractual expiry; three finished without resolving the transaction. None hit a compute limit or infrastructure error.
- All 119 original tagged calls matched parsed calls. Prompt usage remained below the served 40,960-token context.
- Eight original attempts received complete public evidence inside a failed-read response. Zero successful document/message reads therefore did not mean zero evidence exposure. This inconsistent response presentation is a real interface weakness, but does not establish the cause of all failures.
- A public-information-only planner solved all sixteen audited cases.
- Clearer read instructions got Llama to a fully signed, correct amendment on synth-007. It then reported proceed without using the tool that commits it. The binary failure score hides this intermediate improvement.

## Controlled diagnostics

Fresh attempts on two selected failed cases: straightforward execution (synth-007) and deadline extension (synth-025). The first two conditions were specified together. The tool-effect condition was added afterward to investigate observed finish/wait mistakes. Each condition used the same downloaded Llama weights, seven tools, solver, engine, scorer, serving context, temperature/seed and numeric budgets. Only the declared public input supplement changed. No private witness was given to the model.

| Condition | Case | Result | Actions / tool errors | Actual terminal cause | Canonical transcript |
| --- | --- | --- | ---: | --- | --- |
| Clearer resource reads | synth-007 | Failure | 15 / 0 | Correct credit fully signed; claimed proceed in finish without committing a disposition. | [Inspect](http://127.0.0.1:7575/#/tasks/diagnostics/llama-interface-20260927T185817Z/2026-09-27T18-58-18-00-00_dealroom_5XVjHHRfcqGqEv3M4MQVMw.eval) |
| Clearer resource reads | synth-025 | Failure | 5 / 3 | Invalid request arguments, then waited to expiry without an extension. | [Inspect](http://127.0.0.1:7575/#/tasks/diagnostics/llama-interface-20260927T185817Z/2026-09-27T18-58-57-00-00_dealroom_FfRAYXT3eYzFbZkgy5eD6T.eval) |
| Public evidence upfront | synth-007 | Failure | 18 / 0 | Negotiated but never drafted or routed credit; attempted proceed with zero effective credit and finished unresolved. | [Inspect](http://127.0.0.1:7575/#/tasks/diagnostics/llama-interface-20260927T185817Z/2026-09-27T18-59-11-00-00_dealroom_SZbDvo8sLQvfNVKfZpNs4D.eval) |
| Public evidence upfront | synth-025 | Failure | 5 / 0 | Negotiated an extension but never drafted or routed it; waited to expiry. | [Inspect](http://127.0.0.1:7575/#/tasks/diagnostics/llama-interface-20260927T185817Z/2026-09-27T19-00-00-00-00_dealroom_2AuYdGEqhFqG79ZUfbBhqS.eval) |
| Explicit tool effects (follow-up) | synth-007 | Failure | 9 / 4 | Correct credit drafted and routed; waited to deadline instead of completing disposition after signatures. | [Inspect](http://127.0.0.1:7575/#/tasks/diagnostics/llama-interface-20260927T190046Z/2026-09-27T19-00-47-00-00_dealroom_LXc3HE37SVwxaATBgvumLj.eval) |
| Explicit tool effects (follow-up) | synth-025 | Failure | 10 / 6 | Read every initial document and message; invalid requests, then waited to expiry without executing extension. | [Inspect](http://127.0.0.1:7575/#/tasks/diagnostics/llama-interface-20260927T190046Z/2026-09-27T19-01-19-00-00_dealroom_Nqi8cHCFYKX2nEkzWFzSuc.eval) |

All six reached objective failures; none hit a compute limit. All 75 diagnostic tagged calls matched parsed calls. Two malformed JSON calls were already malformed in raw model output. The local transcript links require the supplied Inspect viewer on port 7575.

**Interpretation:** clearer interface documentation improved some behavior but did not produce a successful completion in these six trials. Supplying all initial public evidence did not resolve failure. This is evidence against a simple hidden-data, truncated-context, dropped-call, or incorrect-scoring explanation. Small selected-case probes cannot separate every effect of model, prompt, quantization, runtime, and tool format. No general model ranking follows.

## Integrity and reporting

The six probes are stored separately under logs/diagnostics and are excluded from the matched benchmark. The original batch resumed under its original conditions. Its Qwen synth-044 attempt was interrupted to run these diagnostics; its canonical record is retained as incomplete and unscored, without retry. A reporting fix distinguishes canonical cancellation from genuine execution errors. Full Python verification: 201 tests passed; Ruff passed.

## Original sixteen-attempt audit

### Detailed audit

This read-only audit covers the **16 completed Llama 3.1 8B attempts** present in the `dealroom-local-48-text-tools-v4` report snapshot at **2026-09-27 18:52:59 UTC**. It excludes later attempts, Qwen results, and the separately requested diagnostics. Those diagnostics were still underway when this report was prepared; no conclusions from them are included.

The source records are the canonical `.eval` files under `logs/benchmark/dealroom-local-48-text-tools-v4/`, matched by model `ollama/llama3.1:8b` and the task IDs below. Their task source fingerprint is `b078b1a989cde97b63cfae2001d1e7d190d84e20e971edad8d07ca502bd4539b`. The audit compared raw API responses, parsed calls, tool events, recorded actions, deterministic replay, public evidence, and terminal scores. No source, fixture, log, or result was changed, and no model inference was invoked.

## Findings

All 16 canonical results are failures with reward 0. **None has an Inspect cutoff, sample error, or exhausted domain-action budget.** Thirteen explicitly called `wait_until` at the exact contractual deadline and reached real expiry. Three invoked `finish` while the transaction remained unresolved; one of those falsely claimed a successful disposition.

Across these records there were **109 model generations, 119 tagged calls, 119 parsed calls, 119 ToolEvents, and 87 domain actions**. There were 32 parsing/schema ToolErrors and 30 domain rejections. The 32 ToolErrors comprise two malformed JSON calls and 30 argument-schema violations. They are distinct from normal tool responses reporting that a business action was rejected.

The model successfully read **18 summaries and three amendments, but zero documents, messages, or envelopes**. **That action count does not mean the model never received their contents.** A follow-up audit found that failed non-summary reads return a full public observation, including document and message bodies. Eight of these 16 attempts received those bodies in the actual outgoing model requests despite the read being rejected. The other eight had no full buyer/seller message body in their wire inputs, although their initial brief, policy, authority, resource index, and other observations still supplied public facts.

This corrects the earlier interpretation that unsuccessful retrieval established absence of evidence. Seven of the 13 deadline waits occurred **after** full seller facts had been supplied. Failure to obtain those facts cannot explain all 16 outcomes.

## Per-task classification

Task IDs in this table have the `synth-` prefix. The error columns separate tool parsing/schema errors from engine rejections. All deadline waits below were model-produced actions, not wall-clock timeouts.

| Task | Domain actions | Tool errors | Domain rejections | Observed failure trigger |
| --- | ---: | ---: | ---: | --- |
| `synth-001` | 3 | 0 | 0 | Requested the quote, then waited directly to the deadline; the quote arrived during that wait, before expiry. |
| `synth-002` | 3 | 0 | 0 | Requested the quote, then waited directly to expiry. |
| `synth-007` | 15 | 0 | 7 | Treated negotiation as execution. Proceed and cancellation were rejected; finished with `unresolved`. |
| `synth-008` | 5 | 25 | 3 | Repeated missing arguments and zero-credit offers outside authority; finished with `unresolved`. |
| `synth-013` | 5 | 0 | 1 | Retried signatures before seller availability, then waited to expiry without another retry. |
| `synth-014` | 2 | 0 | 0 | Read the summary, then waited to expiry without repairing the missing signature. |
| `synth-019` | 8 | 1 | 2 | Proposed the old credit amount and waited to expiry without executing the changed revision. |
| `synth-020` | 8 | 3 | 3 | Calculated the correct new credit but routed the old revision. Proceed was rejected; then falsely reported a successful transaction. |
| `synth-025` | 2 | 0 | 0 | Read the summary and waited to the deadline without executing the required extension. |
| `synth-026` | 2 | 0 | 0 | Same extension omission followed by a wait to expiry. |
| `synth-031` | 7 | 1 | 3 | Drafted the full repair cost above the seller cap, routed it, then waited to expiry; never requested cancellation authority. |
| `synth-032` | 7 | 2 | 2 | Same cancellation-route omission: drafted above the seller cap, routed, and waited to expiry. |
| `synth-037` | 5 | 0 | 2 | Requested refreshed authority, then waited to expiry. |
| `synth-038` | 4 | 0 | 1 | Requested refreshed authority, then waited to expiry. |
| `synth-043` | 4 | 0 | 1 | Proposed credit above the seller cap, then waited to the tight deadline. |
| `synth-044` | 7 | 0 | 5 | Repeated requests for a quote already in public documents, then waited to expiry. |

## Actual evidence exposure

The following table checks complete initial `message-001` (buyer) and `message-002` (seller) bodies against the **raw outgoing API requests**, not just the action names. Generation numbers are one-based model calls. In all eight exposed cases, both complete message bodies appeared together and remained in every later request. Those failed-read receipts also included all then-public document bodies: four documents in `synth-007`, `synth-019`, `synth-031`, `synth-037`, and `synth-043`; five in `synth-032`, `synth-038`, and `synth-044`.

| Task | First request containing full buyer/seller bodies | Source receipt |
| --- | --- | --- |
| `synth-001` | None | No full-body receipt. |
| `synth-002` | None | No full-body receipt. |
| `synth-007` | Generation 3 | Action 2: rejected `read(resource="document")`, missing `resource_id`. |
| `synth-008` | None | No full-body receipt. |
| `synth-013` | None | No full-body receipt. |
| `synth-014` | None | No full-body receipt. |
| `synth-019` | Generation 5 | Action 4: rejected `read(resource="amendment")`, missing `resource_id`. |
| `synth-020` | None | No full-body receipt. |
| `synth-025` | None | No full-body receipt. |
| `synth-026` | None | No full-body receipt. |
| `synth-031` | Generation 3 | Action 2: rejected `read(resource="document")`, missing `resource_id`. |
| `synth-032` | Generation 3 | Action 2: rejected `read(resource="document")`, missing `resource_id`. |
| `synth-037` | Generation 3 | Action 2: rejected `read(resource="document")`, missing `resource_id`. |
| `synth-038` | Generation 3 | Action 2: rejected `read(resource="document")`, missing `resource_id`. |
| `synth-043` | Generation 3 | Action 2: rejected `read(resource="document")`, missing `resource_id`. |
| `synth-044` | Generation 3 | Action 2: rejected `read(resource="document")`, missing `resource_id`. |

The complete bodies appeared in 40 outgoing requests across those eight attempts. The seller message included the disclosed credit cap, required evidence, and timing; relevant availability or extension terms were also present. This is a confirmed observation inconsistency in the harness: `tools_for()` leaves non-summary read observations uncompacted, while a failed read's fallback observation is the full public state. A rejected read can therefore expose more public material than a successful selected-resource read. The audited receipts contained public material, not private counterpart implementation or witness answers. No code or historical log was changed during this audit.

## Transport, schema, and public-fact checks

- Every raw tagged call corresponded to a parsed call and a ToolEvent, with matching function names and arguments. No dropped call or altered argument was found. The two malformed raw JSON calls surfaced as parsing errors.
- All 109 generations retained the same complete seven-tool schema payload, equal to Inspect's logged `ToolInfo` schemas. Nested request/terms branches and their required enum tags were present, as were the final-claims fields. Native API `tool_calls` were absent as expected for Inspect's configured text emulation; tagged calls were converted into ordinary Inspect ToolEvents.
- Public messages across all 16 fixtures matched the counterpart implementation's seller caps, required evidence IDs, reply/signature timing, availability, extension limits, and offered authority-grant timing. The public resource index exposed the relevant message IDs. This check found no mismatch between those disclosed values and the engine. Discoverability and actual receipt are distinct: the raw-request exposure table above establishes which models received the full bodies.
- Deadline expiry and rejected dispositions were reproduced from the recorded actions. None of the 16 records completed an acceptable disposition that the scorer then incorrectly rejected.

Representative schema errors were missing `document_id`, missing `amount_cents`, missing final `report`, invalid `intent="credit"`, and `read(resource="authority")` when the supported resource enum was `summary`, `document`, `message`, `envelope`, or `amendment`. Examples of exact feedback include:

```text
'authority' is not one of ['summary', 'document', 'message', 'envelope', 'amendment']
{'intent': 'specialist_evidence'} is not valid under any of the given schemas
'report' is a required property
```

The records do not show a systematic cents-versus-dollars conversion error. In `synth-007`, the proposed 970,700-cent credit correctly equals 1,120,700 cents of cost minus the 150,000-cent residual limit; the mistake was treating unexecuted terms as effective. In `synth-020`, the 552,000-cent new credit was correct, but the model routed `revision-001` instead of the changed `revision-002`. The old effective credit left 280,000 cents residual, above the 190,000-cent limit, and the subsequent claim of `proceed` was unsupported.

## Interpretation and limitations

The strongest confirmed outcome pattern is premature deadline waits or premature finalization. The failed-read fallback is a real harness observation inconsistency and invalidates any inference that zero successful evidence reads means zero evidence exposure. It does not establish why the model then failed: seven expiry cases already had full seller facts in their context. No dropped-call, schema-loss, disclosed-value mismatch, or incorrectly rejected completed disposition was found in this bounded audit. That is not proof that the entire implementation is defect-free.

A plausible setup-related hypothesis is interface burden: evidence is normally accessed through an index but can arrive unexpectedly inside an error receipt, several calls use nested schemas, and generic `anyOf` validation feedback may be difficult to recover from. The current observations do not establish causation or isolate model ability from the prompt, tool interface, serving template, and parser. They should not be generalized to Llama's overall capability, and they do not predict the remaining tasks or the pending diagnostics.

