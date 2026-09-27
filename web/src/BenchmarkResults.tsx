import { useEffect, useMemo, useRef, useState } from 'react';
import { attemptStatuses, inspectLogUrl } from './types';
import type { AttemptStatus, BenchmarkAttempt, BenchmarkExport, BenchmarkModel, BenchmarkTask, Run } from './types';

const labels: Record<AttemptStatus, string> = {
  success: 'Success', failure: 'Scored failure', error: 'Execution error', limit: 'Budget limit',
  incomplete: 'Incomplete', running: 'Running', pending: 'Pending',
};
const integers = new Intl.NumberFormat('en-US');
const decimal = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 });
const duration = (seconds: number) => seconds < 60 ? `${decimal.format(seconds)} s` : `${decimal.format(seconds / 60)} min`;
const runningElapsed = (attempt: BenchmarkAttempt, now: number): number | null => attempt.status === 'running' && attempt.started_at ? Math.max(0, (now - Date.parse(attempt.started_at)) / 1000) : null;
const terminal = (status: AttemptStatus) => !['pending', 'running'].includes(status);
const pct = (numerator: number, denominator: number) => denominator ? `${decimal.format(numerator / denominator * 100)}%` : '—';
const statusCounts = (attempts: BenchmarkAttempt[]) => Object.fromEntries(attemptStatuses.map(status => [status, attempts.filter(attempt => attempt.status === status).length])) as Record<AttemptStatus, number>;

function Outcome({ status }: { status: AttemptStatus }) {
  return <span className={`outcome outcome-${status}`}><span className="status-dot" />{labels[status]}</span>;
}

function ModelCard({ model, attempts, total }: { model: BenchmarkModel; attempts: BenchmarkAttempt[]; total: number }) {
  const counts = statusCounts(attempts);
  const scored = counts.success + counts.failure;
  const durations = attempts.filter(attempt => typeof attempt.duration_seconds === 'number' && terminal(attempt.status));
  const tokenAttempts = attempts.filter(attempt => typeof attempt.total_tokens === 'number');
  const average = durations.length ? durations.reduce((sum, attempt) => sum + attempt.duration_seconds!, 0) / durations.length : null;
  const tokens = tokenAttempts.reduce((sum, attempt) => sum + attempt.total_tokens!, 0);
  return <article className="model-result-card">
    <header><div><div className="section-label">Local model</div><h2>{model.label}</h2><code>{model.id}</code></div><span className="badge">{model.quantization ?? model.provider ?? 'Recorded inference'}</span></header>
    <div className="model-score"><strong>{pct(counts.success, scored)}</strong><div><span>Success / scored tasks</span><b>{counts.success} <span>/ {scored}</span></b></div></div>
    <div className="model-coverage"><span>Scored coverage</span><strong>{scored} / {total} planned tasks</strong><span className="mono">{pct(scored, total)}</span></div>
    <div className="outcome-bar" aria-label={attemptStatuses.map(status => `${labels[status]}: ${counts[status]}`).join(', ')}>{attemptStatuses.map(status => counts[status] > 0 && <span key={status} className={`segment-${status}`} style={{ width: `${counts[status] / Math.max(total, 1) * 100}%` }} title={`${labels[status]}: ${counts[status]}`} />)}</div>
    <dl className="outcome-counts">{attemptStatuses.map(status => <div key={status}><dt><span className={`legend-dot segment-${status}`} />{labels[status]}</dt><dd>{counts[status]}</dd></div>)}</dl>
    <dl className="model-measurements"><div><dt>Mean attempt time</dt><dd>{average === null ? '—' : duration(average)}</dd><span>{durations.length} recorded timings</span></div><div><dt>Total tokens</dt><dd>{tokenAttempts.length ? integers.format(tokens) : '—'}</dd><span>{tokenAttempts.length} recorded counts</span></div></dl>
  </article>;
}

type OutcomeFilter = 'all' | 'completed' | 'different' | AttemptStatus;

export default function BenchmarkResults({ benchmark, runs, onInspect, refreshing, onRefresh }: {
  benchmark: BenchmarkExport; runs: Run[]; onInspect: (run: Run) => void; refreshing: boolean; onRefresh: () => void;
}) {
  const [query, setQuery] = useState('');
  const [modelFilter, setModelFilter] = useState('all');
  const [outcomeFilter, setOutcomeFilter] = useState<OutcomeFilter>('all');
  const [selected, setSelected] = useState<{ taskId: string; modelId: string } | null>(null);
  const detailPanel = useRef<HTMLElement>(null);
  const matrixSearch = useRef<HTMLInputElement>(null);
  const detailTrigger = useRef<HTMLElement | null>(null);
  function showDetails(taskId: string, modelId: string, trigger: HTMLElement) { detailTrigger.current = trigger; setSelected({ taskId, modelId }); }
  function closeDetails() {
    setSelected(null);
    const trigger = detailTrigger.current;
    window.requestAnimationFrame(() => {
      if (trigger?.isConnected) trigger.focus();
      if (!trigger?.isConnected || document.activeElement !== trigger) matrixSearch.current?.focus();
    });
  }
  useEffect(() => { if (selected) detailPanel.current?.focus(); }, [selected]);
  const attempts = useMemo(() => {
    const recorded = new Map(benchmark.attempts.map(attempt => [JSON.stringify([attempt.task_id, attempt.model_id]), attempt]));
    return benchmark.tasks.flatMap(task => benchmark.models.map(model => recorded.get(JSON.stringify([task.id, model.id])) ?? {
      id: `pending:${model.id}:${task.id}`, task_id: task.id, model_id: model.id, status: 'pending' as const,
    }));
  }, [benchmark]);
  const byPair = useMemo(() => new Map(attempts.map(attempt => [JSON.stringify([attempt.task_id, attempt.model_id]), attempt])), [attempts]);
  const attemptFor = (taskId: string, modelId: string) => byPair.get(JSON.stringify([taskId, modelId]))!;
  const models = benchmark.models.filter(model => modelFilter === 'all' || model.id === modelFilter);
  const matchesOutcome = (task: BenchmarkTask) => {
    const outcomes = models.map(model => attemptFor(task.id, model.id).status);
    if (outcomeFilter === 'all') return true;
    if (outcomeFilter === 'completed') return outcomes.some(status => ['success', 'failure'].includes(status));
    if (outcomeFilter === 'different') {
      const allOutcomes = benchmark.models.map(model => attemptFor(task.id, model.id).status);
      return allOutcomes.length >= 2 && allOutcomes.every(terminal) && new Set(allOutcomes).size > 1;
    }
    return outcomes.includes(outcomeFilter);
  };
  const tasks = benchmark.tasks.filter(task => `${task.id} ${task.title} ${task.family ?? ''}`.toLowerCase().includes(query.toLowerCase()) && matchesOutcome(task));
  const counts = statusCounts(attempts);
  const resolved = attempts.filter(attempt => terminal(attempt.status)).length;
  const scored = counts.success + counts.failure;
  const now = Date.now();
  const selectedAttempt = selected ? attemptFor(selected.taskId, selected.modelId) : undefined;
  const selectedElapsed = selectedAttempt ? runningElapsed(selectedAttempt, now) : null;
  const selectedTask = selected ? benchmark.tasks.find(task => task.id === selected.taskId) : undefined;
  const selectedModel = selected ? benchmark.models.find(model => model.id === selected.modelId) : undefined;
  const selectedRun = selectedAttempt?.run_id ? runs.find(run => run.id === selectedAttempt.run_id) : undefined;
  const availableRuns = new Map(runs.map(run => [run.id, run]));
  const settings = benchmark.config;
  return <section className="benchmark-workspace" aria-label="Results by model">
    <header className="benchmark-header"><div><div className="section-label">Recorded evaluation <span> / </span> {benchmark.models.length} local models</div><h1>{benchmark.title}</h1><p>Same synthetic tasks, independently attempted by each model. Objective rewards from the DealRoom verifier.</p></div><div className="benchmark-header-actions"><span className={`benchmark-state ${benchmark.status === 'running' ? 'is-running' : ''}`}><span className="status-dot" />{benchmark.status === 'complete' ? 'Evaluation complete' : benchmark.status === 'interrupted' ? 'Evaluation interrupted' : 'Evaluation running'}</span><button className="button" disabled={refreshing} onClick={onRefresh}>{refreshing ? 'Refreshing…' : 'Refresh results'}</button></div></header>
    <div className="benchmark-progress"><div><strong>{resolved} / {attempts.length}</strong><span>attempts resolved</span><span className="progress-divider" /><strong>{scored}</strong><span>scored completions</span></div><span>{benchmark.tasks.length} tasks · seed <code>{benchmark.seed}</code>{benchmark.status === 'running' && ' · refreshes every 15 seconds'}</span></div>
    <div className="model-results-grid">{benchmark.models.map(model => <ModelCard key={model.id} model={model} attempts={attempts.filter(attempt => attempt.model_id === model.id)} total={benchmark.tasks.length} />)}</div>
    <p className="benchmark-method-note">Success rate counts scored tasks only; limits, errors, and unfinished attempts stay separate. Small or partial scored coverage cannot support a model comparison. Timing and tokens use recorded values; scripts are excluded.</p>
    <section className="outcome-matrix" aria-label="Task outcomes by model"><div className="matrix-heading"><div><h2>Task outcomes</h2><span>{tasks.length} of {benchmark.tasks.length} tasks</span></div><p>Select an outcome to inspect its recorded trajectory.</p></div><div className="matrix-filters"><label className="matrix-search"><span className="visually-hidden">Search benchmark tasks</span><input ref={matrixSearch} aria-label="Search benchmark tasks" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search task, family, or ID…" /></label><label><span>Model</span><select aria-label="Filter benchmark model" value={modelFilter} onChange={event => setModelFilter(event.target.value)}><option value="all">All models</option>{benchmark.models.map(model => <option key={model.id} value={model.id}>{model.label}</option>)}</select></label><label><span>Outcome</span><select aria-label="Filter benchmark outcome" value={outcomeFilter} onChange={event => setOutcomeFilter(event.target.value as OutcomeFilter)}><option value="all">All outcomes</option><option value="completed">Scored completions</option><option value="different">Different resolved outcomes</option>{attemptStatuses.map(status => <option key={status} value={status}>{labels[status]}</option>)}</select></label></div>
      <div className="table-scroll"><table className="outcomes-table"><thead><tr><th scope="col">Task</th>{models.map(model => <th key={model.id} scope="col"><span>{model.label}</span><code>{model.id}</code></th>)}</tr></thead><tbody>{tasks.map(task => <tr key={task.id}><th scope="row"><code>{task.id}</code><strong>{task.title}</strong><span>{task.family ?? task.split}</span></th>{models.map(model => {
        const attempt = attemptFor(task.id, model.id);
        const elapsed = runningElapsed(attempt, now);
        const replay = attempt.run_id ? availableRuns.get(attempt.run_id) : undefined;
        return <td key={model.id}><div className="matrix-outcome"><button className="outcome-button" title={replay ? 'Open recorded trajectory' : 'View attempt details'} aria-label={`${model.label}, ${task.id}: ${labels[attempt.status]}${replay ? '. Open trajectory' : '. View details'}`} onClick={event => replay ? onInspect(replay) : showDetails(task.id, model.id, event.currentTarget)}><Outcome status={attempt.status} />{replay && <span className="outcome-arrow" aria-hidden="true">↗</span>}</button><button className="attempt-details-button" aria-label={`Details for ${model.label}, ${task.id}`} aria-pressed={selected?.taskId === task.id && selected?.modelId === model.id} onClick={event => showDetails(task.id, model.id, event.currentTarget)}>Details</button></div><div className="matrix-attempt-meta">{attempt.reward !== null && attempt.reward !== undefined && <span>reward <code>{attempt.reward.toFixed(3)}</code></span>}{attempt.status !== 'running' && attempt.duration_seconds !== undefined && attempt.duration_seconds !== null && <span>{duration(attempt.duration_seconds)}</span>}{attempt.status === 'pending' && <span>Awaiting execution</span>}{attempt.status === 'running' && <span>{elapsed === null ? 'Inference in progress · start not recorded' : `Running elapsed ${duration(elapsed)}`}</span>}</div></td>;
      })}</tr>)}</tbody></table></div>{!tasks.length && <div className="matrix-empty">No tasks match these filters.<button className="text-link" onClick={() => { setQuery(''); setOutcomeFilter('all'); setModelFilter('all'); }}>Clear filters</button></div>}
    </section>
    {selectedAttempt && selectedTask && selectedModel && <section ref={detailPanel} tabIndex={-1} onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); closeDetails(); } }} className="attempt-detail-panel" aria-label="Selected attempt details"><header><div><div className="section-label">Attempt details</div><h2>{selectedModel.label} <span>/</span> {selectedTask.id}</h2></div><button className="button subtle" onClick={closeDetails}>Close details</button></header><div className="attempt-detail-body"><Outcome status={selectedAttempt.status} /><p>{selectedTask.title}</p>{selectedAttempt.reason && <p className="attempt-reason">{selectedAttempt.reason}</p>}<dl className="attempt-metrics"><div><dt>Reward</dt><dd>{selectedAttempt.reward == null ? 'Not scored' : selectedAttempt.reward.toFixed(3)}</dd></div><div><dt>{selectedAttempt.status === 'running' ? 'Running elapsed' : 'Elapsed time'}</dt><dd>{selectedAttempt.status === 'running' ? selectedElapsed === null ? 'Start not recorded' : duration(selectedElapsed) : selectedAttempt.duration_seconds == null ? 'Not recorded' : duration(selectedAttempt.duration_seconds)}</dd></div><div><dt>Input tokens</dt><dd>{selectedAttempt.input_tokens == null ? 'Not recorded' : integers.format(selectedAttempt.input_tokens)}</dd></div><div><dt>Output tokens</dt><dd>{selectedAttempt.output_tokens == null ? 'Not recorded' : integers.format(selectedAttempt.output_tokens)}</dd></div><div><dt>Total tokens</dt><dd>{selectedAttempt.total_tokens == null ? 'Not recorded' : integers.format(selectedAttempt.total_tokens)}</dd></div>{([{ key: 'actions', label: 'Domain actions' }, { key: 'tool_calls', label: 'Tool calls' }, { key: 'tool_errors', label: 'Tool errors' }, { key: 'generations', label: 'Model responses' }] as const).map(metric => { const value = selectedAttempt[metric.key]; return <div key={metric.key}><dt>{metric.label}</dt><dd>{value === undefined ? 'Not recorded' : integers.format(value)}</dd></div>; })}</dl><div className="attempt-links">{selectedRun && <button className="button" onClick={() => onInspect(selectedRun)}>Open recorded trajectory ↗</button>}{selectedAttempt.inspect_log && <a className="button" href={inspectLogUrl(selectedAttempt.inspect_log)} target="_blank" rel="noreferrer">Inspect full transcript ↗</a>}{!selectedRun && <span>{selectedAttempt.run_id ? 'Replay is not present in the loaded export.' : 'No scored replay is available for this attempt.'}</span>}</div><code className="attempt-id">{selectedAttempt.id}</code></div></section>}
    <details className="benchmark-configuration"><summary>Evaluation configuration & provenance</summary><div><dl className="config-grid"><div><dt>Evaluation ID</dt><dd><code>{benchmark.id}</code></dd></div><div><dt>Last export</dt><dd>{new Date(benchmark.generated_at).toLocaleString()}</dd></div>{settings && Object.entries(settings).filter(([key]) => key !== 'notes').map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd><code>{String(value)}</code></dd></div>)}</dl>{settings?.notes && <p>{settings.notes}</p>}<div className="model-provenance">{benchmark.models.map(model => <div key={model.id}><strong>{model.label}</strong><code>{model.id}</code>{model.parameters && <span>Parameters: {model.parameters}</span>}{model.quantization && <span>Quantization: {model.quantization}</span>}{model.digest && <code className="model-digest">Digest: {model.digest}</code>}</div>)}</div></div></details>
  </section>;
}
