# DealRoom learning results

Experiment: `dealroom-llama-learning-v2`. Test state: `opened`.

Rates use scored attempts only. Every count below comes from a recorded attempt.

| Condition | Split | Success / scored | Success rate | Scored cases / planned | Limits / errors / incomplete |
|---|---|---:|---:|---:|---:|
| original | validation | 0 / 32 | 0.0% | 32 / 32 | 0 / 0 / 0 |
| original | sealed_test | 0 / 64 | 0.0% | 64 / 64 | 0 / 0 / 0 |
| workflow | validation | 1 / 32 | 3.1% | 32 / 32 | 0 / 0 / 0 |
| workflow | sealed_test | 0 / 39 | 0.0% | 39 / 64 | 0 / 25 / 0 |
| sft | validation | 2 / 32 | 6.2% | 32 / 32 | 0 / 0 / 0 |
| sft | sealed_test | 2 / 64 | 3.1% | 64 / 64 | 0 / 0 / 0 |
| rl | train | 4 / 32 | 12.5% | 8 / 256 | 0 / 0 / 0 |
| rl | validation | 2 / 32 | 6.2% | 32 / 32 | 0 / 0 / 0 |
| rl | sealed_test | 2 / 64 | 3.1% | 64 / 64 | 0 / 0 / 0 |

## Paired held-out changes

Positive changes favor the later condition. Both conditions must have scored the case.

| Comparison | Paired cases / planned | Wins / losses / ties | Change (percentage points) | Bootstrap 95% interval |
|---|---:|---:|---:|---:|
| workflow − original | 39 / 64 | 0 / 0 / 39 | +0.0 | [+0.0, +0.0] |
| sft − workflow | 39 / 64 | 2 / 0 / 37 | +5.1 | [+0.0, +12.8] |
| rl − sft | 64 / 64 | 0 / 0 / 64 | +0.0 | [+0.0, +0.0] |

Percentile bootstrap resamples paired synthetic cases as exchangeable observations. Shared generation templates can violate independence. The interval is conditional on this synthetic suite and the subset scored by both conditions; it is not a general model ranking and does not include training-seed or sampling-seed uncertainty.

## Recorded inference costs

Costs include native agent attempts, excluding model optimization. Missing measurements remain missing.

| Condition | Split | All-attempt tokens | Scored-attempt tokens | All-attempt seconds | Scored-attempt seconds |
|---|---|---:|---:|---:|---:|
| original | validation | 491,726.0 (32 recorded; 0 missing) | 491,726.0 (32 recorded; 0 missing) | 300.0 (32 recorded; 0 missing) | 300.0 (32 recorded; 0 missing) |
| original | sealed_test | 1,261,877.0 (64 recorded; 0 missing) | 1,261,877.0 (64 recorded; 0 missing) | 644.0 (64 recorded; 0 missing) | 644.0 (64 recorded; 0 missing) |
| workflow | validation | 1,322,881.0 (32 recorded; 0 missing) | 1,322,881.0 (32 recorded; 0 missing) | 633.6 (32 recorded; 0 missing) | 633.6 (32 recorded; 0 missing) |
| workflow | sealed_test | 3,412,340.0 (42 recorded; 22 missing) | 3,401,100.0 (39 recorded; 0 missing) | 1,472.4 (64 recorded; 0 missing) | 1,379.1 (39 recorded; 0 missing) |
| sft | validation | 2,209,711.0 (32 recorded; 0 missing) | 2,209,711.0 (32 recorded; 0 missing) | 758.3 (32 recorded; 0 missing) | 758.3 (32 recorded; 0 missing) |
| sft | sealed_test | 4,770,355.0 (64 recorded; 0 missing) | 4,770,355.0 (64 recorded; 0 missing) | 1,586.8 (64 recorded; 0 missing) | 1,586.8 (64 recorded; 0 missing) |
| rl | train | 3,542,107.0 (32 recorded; 0 missing) | 3,542,107.0 (32 recorded; 0 missing) | 1,030.2 (32 recorded; 0 missing) | 1,030.2 (32 recorded; 0 missing) |
| rl | validation | 2,168,972.0 (32 recorded; 0 missing) | 2,168,972.0 (32 recorded; 0 missing) | 695.2 (32 recorded; 0 missing) | 695.2 (32 recorded; 0 missing) |
| rl | sealed_test | 4,730,854.0 (64 recorded; 0 missing) | 4,730,854.0 (64 recorded; 0 missing) | 1,510.7 (64 recorded; 0 missing) | 1,510.7 (64 recorded; 0 missing) |

Success rates use success / (success + failure); unscored attempts are excluded, not assigned zero.
Paired deltas use exactly the same cases scored by both conditions; inspect paired coverage alongside each delta.
Recorded cost totals include only available measurements. All-attempt costs include unscored and active attempts when measured; scored costs exclude them. Optimization compute is not an attempt cost.
Training groups may include repeated stochastic rollouts of one case; training success rates are per rollout and case coverage counts unique cases.
Percentile bootstrap resamples paired synthetic cases as exchangeable observations. Shared generation templates can violate independence. The interval is conditional on this synthetic suite and the subset scored by both conditions; it is not a general model ranking and does not include training-seed or sampling-seed uncertainty.
