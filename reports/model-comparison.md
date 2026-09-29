# DealRoom local model results

48 synthesized tasks across eight scenario families, attempted once by each downloaded model. All 96 attempts are resolved. Rewards come from the deterministic DealRoom verifier; limits, execution errors, and incomplete attempts remain separate from scored failures.

| Model | Success / scored | Success rate | Scored coverage | Scored failure | Limit | Error | Incomplete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama 3.1 8B | 1 / 46 | 2.2% | 46 / 48 | 45 | 2 | 0 | 0 |
| Qwen3 8B | 13 / 32 | 40.6% | 32 / 48 | 19 | 15 | 0 | 1 |

## Results by scenario family

Each cell shows successes / scored tasks, followed by scored coverage out of six planned tasks.

| Family | Llama 3.1 8B | Qwen3 8B |
| --- | ---: | ---: |
| quote discovery | 0 / 5 scored; coverage 5 / 6 | 0 / 3 scored; coverage 3 / 6 |
| straightforward execution | 0 / 5 scored; coverage 5 / 6 | 0 / 3 scored; coverage 3 / 6 |
| partial signature retry | 1 / 6 scored; coverage 6 / 6 | 1 / 6 scored; coverage 6 / 6 |
| revised credit execution | 0 / 6 scored; coverage 6 / 6 | 6 / 6 scored; coverage 6 / 6 |
| deadline extension | 0 / 6 scored; coverage 6 / 6 | 0 / 2 scored; coverage 2 / 6 |
| cancellation authority | 0 / 6 scored; coverage 6 / 6 | 6 / 6 scored; coverage 6 / 6 |
| credit authority refresh | 0 / 6 scored; coverage 6 / 6 | 0 / 3 scored; coverage 3 / 6 |
| boundary timing | 0 / 6 scored; coverage 6 / 6 | 0 / 3 scored; coverage 3 / 6 |

## Configuration and records

- Runtime: Ollama 0.31.1, Inspect AI 0.3.271.
- Tool interface: inspect_text_emulation. This condition is shared by both models.
- Local Q4_K_M models; 40,960-token context; thinking disabled; temperature 0; seed 20260927.
- Per attempt: 128 turns, 512 messages, 2,000,000 cumulative tokens, 2,048 output tokens per generation, 900 seconds, 256 domain actions.
- Native deterministic history trimming at 28,672 tokens preserves the system/initial task and half of subsequent messages.
- Real irreversible contractual expiry ends the episode as an objective task failure without fabricating a finish or final claims.
- Compute limits, execution errors, and incomplete attempts are excluded from the scored-task success rate. Coverage is reported separately.
- No outcome retries or model generation cache. Prompts, tools, fixtures, and settings frozen across the run.
- Every generated task has a successful private engine witness. Witness actions were excluded from model inputs.
- Canonical transcripts: `logs/benchmark/dealroom-local-48-text-tools-v4/` inside the archive.
- Exact model digests and task/source fingerprints: `web/public/benchmark.json`.
- Scripted demonstrations, infrastructure pilots, six separate Llama interface diagnostics, and stopped native-tool runs are excluded from these counts.
- Qwen synth-044 was interrupted by the operator for the Llama investigation and is retained unscored without retry. See reports/llama-diagnosis.md for the six isolated probes.
- The initial run is retained under `logs/benchmark/dealroom-local-48-v1/`, including its report, replays, 53 native logs, and original core source snapshots.
- The expanded native-tool diagnostic is retained under `logs/benchmark/dealroom-local-48-expanded-v2/`, including 4 native logs, its interrupted report, replays, and core source snapshots.
- The early text-tool run is retained under `logs/benchmark/dealroom-local-48-text-tools-v3/`, including 6 native logs. It was stopped to correct numeric zero scores on unfinished/limited samples in the native viewer; the current condition uses Inspect Score.unscored.

These are results for one small synthetic suite under this configuration. Related variants are not independent real-world tasks, and the results do not establish a general model ranking.
