export interface Diagnostic {
  predicate: string;
  passed: boolean;
  reason: string;
  evidence: string[];
}

export interface Evidence {
  id: string;
  title: string;
  body: string;
  time?: string;
}

export interface Amendment {
  id: string;
  kind: string;
  credit_cents: number | null;
  deadline: string | null;
  status: string;
  required_signers: string[];
  signatures: { party: string; time: string }[];
}

export interface Snapshot {
  time: string;
  deadline: string;
  disposition: string;
  effective_credit_cents: number;
  residual_cents: number | null;
  documents: Evidence[];
  messages: Evidence[];
  amendments: Amendment[];
  authority: Record<string, unknown>;
}

export interface Step {
  index: number;
  label: string;
  action: Record<string, unknown> | null;
  result: Record<string, unknown> | null;
  state: Snapshot;
  reviewer: { note: string; diagnostics: Diagnostic[] };
}

export interface Run {
  id: string;
  case_id: string;
  case_title: string;
  split: 'development' | 'evaluation';
  kind: 'scripted_failure' | 'repaired_script' | 'fixture_witness' | 'recorded_model';
  label: string;
  description: string;
  branch_step?: number;
  paired_run_id?: string;
  task?: {
    brief?: string;
    policy?: string;
    system_prompt?: string;
    tools?: { name: string; description: string; parameters: Record<string, unknown> }[];
    initial_observation?: Record<string, unknown>;
  };
  provenance?: {
    inspect_log?: string;
    model?: string;
    source?: string;
    usage?: Record<string, { input_tokens: number; output_tokens: number; total_tokens: number }>;
  };
  score: { success: boolean; diagnostics: Diagnostic[] };
  steps: Step[];
}

export interface RunExport {
  schema_version: 1;
  generated_at: string;
  runs: Run[];
  summary?: Record<string, unknown>;
  issues?: { sample_id: string; status: string; reason: string }[];
}

export type AttemptStatus = 'pending' | 'running' | 'success' | 'failure' | 'error' | 'limit' | 'incomplete';

export interface BenchmarkModel {
  id: string;
  label: string;
  provider?: string;
  parameters?: string;
  quantization?: string;
  digest?: string;
}

export interface BenchmarkTask {
  id: string;
  title: string;
  split: 'development' | 'evaluation';
  family?: string;
}

export interface BenchmarkAttempt {
  id: string;
  task_id: string;
  model_id: string;
  status: AttemptStatus;
  run_id?: string;
  inspect_log?: string;
  reason?: string;
  reward?: number | null;
  started_at?: string;
  duration_seconds?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  total_tokens?: number | null;
  actions?: number;
  tool_calls?: number;
  tool_errors?: number;
  generations?: number;
}

export interface BenchmarkExport {
  schema_version: 1;
  id: string;
  title: string;
  generated_at: string;
  status: 'running' | 'complete' | 'interrupted';
  seed: number;
  task_count: number;
  models: BenchmarkModel[];
  tasks: BenchmarkTask[];
  attempts: BenchmarkAttempt[];
  config?: { temperature?: number; max_messages?: number; max_tokens?: number; max_turns?: number; seed?: number; thinking?: boolean; context_window?: number; output_tokens_per_turn?: number; time_limit_seconds?: number; notes?: string };
}

export const attemptStatuses: AttemptStatus[] = ['success', 'failure', 'error', 'limit', 'incomplete', 'running', 'pending'];

export function inspectLogUrl(log: string): string {
  return `http://127.0.0.1:7575/#/tasks/${log.split(/[\\/]/).map(encodeURIComponent).join('/')}`;
}

export function missingBenchmarkReplays(benchmark: BenchmarkExport | null, replay: RunExport | null): string[] {
  if (!benchmark) return [];
  const available = new Set(replay?.runs.map(run => run.id) ?? []);
  return [...new Set(benchmark.attempts.flatMap(attempt => attempt.run_id && !available.has(attempt.run_id) ? [attempt.run_id] : []))];
}

const kinds = ['scripted_failure', 'repaired_script', 'fixture_witness', 'recorded_model'];
const object = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);
const text = (value: unknown): value is string => typeof value === 'string';
const number = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const stringArray = (value: unknown): boolean => Array.isArray(value) && value.every(text);
const date = (value: unknown): boolean => text(value) && !Number.isNaN(Date.parse(value));
const timestamp = (value: unknown): boolean => text(value) && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/i.test(value) && date(value);

function diagnostics(value: unknown): boolean {
  return Array.isArray(value) && value.every(d => object(d) && text(d.predicate) && typeof d.passed === 'boolean' && text(d.reason) && stringArray(d.evidence));
}

function evidence(value: unknown): boolean {
  return Array.isArray(value) && value.every(d => object(d) && text(d.id) && text(d.title) && text(d.body) && (d.time === undefined || date(d.time)));
}

function snapshot(value: unknown): boolean {
  if (!object(value)) return false;
  return date(value.time) && date(value.deadline) && text(value.disposition) && number(value.effective_credit_cents) && (value.residual_cents === null || number(value.residual_cents))
    && evidence(value.documents) && evidence(value.messages) && object(value.authority)
    && Array.isArray(value.amendments) && value.amendments.every(a => object(a) && text(a.id) && text(a.kind) && text(a.status)
      && (a.credit_cents === null || number(a.credit_cents)) && (a.deadline === null || date(a.deadline)) && stringArray(a.required_signers)
      && Array.isArray(a.signatures) && a.signatures.every(s => object(s) && text(s.party) && date(s.time)));
}

function task(value: unknown): boolean {
  if (value === undefined) return true;
  return object(value) && ['brief', 'policy', 'system_prompt'].every(key => value[key] === undefined || text(value[key]))
    && (value.initial_observation === undefined || object(value.initial_observation))
    && (value.tools === undefined || (Array.isArray(value.tools) && value.tools.every(tool => object(tool) && text(tool.name) && text(tool.description) && object(tool.parameters))));
}

function provenance(value: unknown): boolean {
  if (value === undefined) return true;
  return object(value) && ['inspect_log', 'model', 'source'].every(key => value[key] === undefined || text(value[key]))
    && (value.usage === undefined || (object(value.usage) && Object.values(value.usage).every(usage => object(usage)
      && ['input_tokens', 'output_tokens', 'total_tokens'].every(key => number(usage[key]) && Number(usage[key]) >= 0))));
}

export function parseExport(value: unknown): RunExport {
  if (!object(value) || value.schema_version !== 1 || !date(value.generated_at) || !Array.isArray(value.runs)) {
    throw new Error('This file is not a DealRoom replay export. Expected schema_version 1, generated_at, and a runs array.');
  }
  const ids = new Set<string>();
  for (const [index, run] of value.runs.entries()) {
    if (!object(run) || !text(run.id) || !text(run.case_id) || !text(run.case_title) || !text(run.label) || !text(run.description)
      || !kinds.includes(String(run.kind)) || !['development', 'evaluation'].includes(String(run.split))
      || (run.branch_step !== undefined && (!number(run.branch_step) || run.branch_step < 0))
      || (run.paired_run_id !== undefined && !text(run.paired_run_id))
      || !task(run.task) || !provenance(run.provenance)
      || !object(run.score) || typeof run.score.success !== 'boolean' || !diagnostics(run.score.diagnostics)
      || !Array.isArray(run.steps) || run.steps.length === 0 || !run.steps.every(s => object(s) && number(s.index) && text(s.label)
        && (s.action === null || object(s.action)) && (s.result === null || object(s.result)) && snapshot(s.state)
        && object(s.reviewer) && text(s.reviewer.note) && diagnostics(s.reviewer.diagnostics))) {
      throw new Error(`Run ${index + 1} is incomplete or has invalid fields. Re-export the run with the DealRoom Python exporter.`);
    }
    if (ids.has(run.id)) throw new Error(`Duplicate run identifier: ${run.id}. Each run must have a unique ID.`);
    ids.add(run.id);
  }
  return value as unknown as RunExport;
}

export function parseBenchmark(value: unknown): BenchmarkExport {
  if (!object(value) || value.schema_version !== 1 || !text(value.id) || !text(value.title)
    || !date(value.generated_at) || !['running', 'complete', 'interrupted'].includes(String(value.status))
    || !number(value.seed) || !number(value.task_count) || !Array.isArray(value.models) || !value.models.length
    || !Array.isArray(value.tasks) || value.tasks.length !== value.task_count || !Array.isArray(value.attempts)) {
    throw new Error('The benchmark export has invalid metadata. Regenerate benchmark.json with the DealRoom benchmark runner.');
  }
  const modelIds = new Set<string>();
  for (const model of value.models) {
    if (!object(model) || !text(model.id) || !text(model.label) || modelIds.has(model.id)
      || !['provider', 'parameters', 'quantization', 'digest'].every(key => model[key] === undefined || text(model[key]))) {
      throw new Error('The benchmark export has invalid or duplicate models.');
    }
    modelIds.add(model.id);
  }
  const taskIds = new Set<string>();
  for (const task of value.tasks) {
    if (!object(task) || !text(task.id) || !text(task.title) || taskIds.has(task.id)
      || !['development', 'evaluation'].includes(String(task.split)) || (task.family !== undefined && !text(task.family))) {
      throw new Error('The benchmark export has invalid or duplicate tasks.');
    }
    taskIds.add(task.id);
  }
  const attemptIds = new Set<string>();
  const pairs = new Set<string>();
  for (const attempt of value.attempts) {
    if (!object(attempt) || !text(attempt.id) || attemptIds.has(attempt.id)
      || !text(attempt.task_id) || !taskIds.has(attempt.task_id) || !text(attempt.model_id) || !modelIds.has(attempt.model_id)
      || !attemptStatuses.includes(attempt.status as AttemptStatus)
      || (attempt.started_at !== undefined && !timestamp(attempt.started_at))
      || !['actions', 'tool_calls', 'tool_errors', 'generations'].every(key => attempt[key] === undefined || (number(attempt[key]) && Number.isSafeInteger(attempt[key]) && Number(attempt[key]) >= 0))
      || !['run_id', 'inspect_log', 'reason'].every(key => attempt[key] === undefined || text(attempt[key]))
      || !['reward', 'duration_seconds', 'input_tokens', 'output_tokens', 'total_tokens'].every(key => attempt[key] === undefined || attempt[key] === null || (number(attempt[key]) && Number(attempt[key]) >= 0))) {
      throw new Error('The benchmark export has an invalid attempt or references an unknown task or model.');
    }
    if ((attempt.status === 'success' && attempt.reward !== 1)
      || (attempt.status === 'failure' && attempt.reward !== 0)
      || (!['success', 'failure'].includes(String(attempt.status)) && attempt.reward !== undefined && attempt.reward !== null)) {
      throw new Error('Benchmark rewards must match outcomes: success requires reward 1, scored failure requires reward 0, and unscored attempts cannot have a reward.');
    }
    const pair = JSON.stringify([attempt.task_id, attempt.model_id]);
    if (pairs.has(pair)) throw new Error('The benchmark export contains duplicate task/model attempts. Export a single evaluation per pair.');
    pairs.add(pair);
    attemptIds.add(attempt.id);
  }
  const config = value.config;
  if (config !== undefined && (!object(config)
    || !['temperature', 'max_messages', 'max_tokens', 'max_turns', 'seed', 'context_window', 'output_tokens_per_turn', 'time_limit_seconds'].every(key => config[key] === undefined || number(config[key]))
    || (config.thinking !== undefined && typeof config.thinking !== 'boolean')
    || (config.notes !== undefined && !text(config.notes)))) {
    throw new Error('The benchmark configuration is invalid.');
  }
  return value as unknown as BenchmarkExport;
}
