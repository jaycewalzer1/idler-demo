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

const kinds = ['scripted_failure', 'repaired_script', 'fixture_witness', 'recorded_model'];
const object = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);
const text = (value: unknown): value is string => typeof value === 'string';
const number = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const stringArray = (value: unknown): boolean => Array.isArray(value) && value.every(text);
const date = (value: unknown): boolean => text(value) && !Number.isNaN(Date.parse(value));

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
