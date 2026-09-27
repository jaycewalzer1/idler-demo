import { useMemo, useState } from 'react';
import { inspectLogUrl, stageEvaluations } from './types';
import type { AttemptStatus, LearningAttempt, LearningEvaluation, LearningExport, LearningSplit, LearningStage, LearningStageId } from './types';

const integers = new Intl.NumberFormat('en-US');
const decimals = new Intl.NumberFormat('en-US', { maximumFractionDigits: 3 });
const percentage = (success: number, scored: number) => scored ? `${decimals.format(success / scored * 100)}%` : '—';
const readable = (value: string) => value.replaceAll('_', ' ').replace(/^\w/, letter => letter.toUpperCase());
const time = (value: string) => new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZoneName: 'short' }).format(new Date(value));
const stageOrder: LearningStageId[] = ['original', 'workflow', 'sft', 'rl'];
const stageLabels = { original: 'Original model', workflow: 'Clearer workflow', sft: 'Supervised fine-tuning', rl: 'Reinforcement learning' };
const stageDescriptions = {
  original: 'Starting model with the original workflow.',
  workflow: 'The same base model with clarified tool and completion instructions.',
  sft: 'An adapter learns from successful training demonstrations.',
  rl: 'The supervised checkpoint learns from environment rewards.',
};
const statusLabels = { preparing: 'Preparing experiment', baseline: 'Running baselines', sft: 'Supervised training', rl: 'Reinforcement learning', evaluating: 'Evaluating checkpoints', complete: 'Experiment complete', blocked: 'Needs attention' };
const attemptLabels: Record<AttemptStatus, string> = { success: 'Success', failure: 'Scored failure', limit: 'Budget limit', error: 'Execution error', incomplete: 'Incomplete', pending: 'Pending', running: 'Running' };
const splitLabels = { train: 'Training rollout', validation: 'Validation', sealed_test: 'Held-out test' };

function Attempts({ attempts }: { attempts: LearningAttempt[] }) {
  const [stage, setStage] = useState<LearningStageId | 'all'>('all');
  const [split, setSplit] = useState<LearningAttempt['split'] | 'all'>('all');
  const [query, setQuery] = useState('');
  const visible = attempts.filter(attempt => (stage === 'all' || attempt.stage === stage)
    && (split === 'all' || attempt.split === split) && attempt.case_id.toLowerCase().includes(query.toLowerCase()));
  return <section className="learning-panel learning-attempts" aria-label="Recorded learning attempts"><header><div><h2>Recorded attempts</h2><p>Open a native Inspect log for the full model transcript and verifier result. Training rollouts are separate from evaluation.</p></div><span className="section-label">{visible.length} / {attempts.length} records</span></header>
    <div className="learning-attempt-filters"><label className="learning-attempt-search">Case<input type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search case ID…" aria-label="Search learning attempt case" /></label><label>Stage<select aria-label="Filter learning attempt stage" value={stage} onChange={event => setStage(event.target.value as LearningStageId | 'all')}><option value="all">All stages</option>{stageOrder.map(id => <option key={id} value={id}>{stageLabels[id]}</option>)}</select></label><label>Split<select aria-label="Filter learning attempt split" value={split} onChange={event => setSplit(event.target.value as LearningAttempt['split'] | 'all')}><option value="all">All splits</option>{(['train', 'validation', 'sealed_test'] as const).map(id => <option key={id} value={id}>{splitLabels[id]}</option>)}</select></label></div>
    {visible.length ? <div className="table-scroll"><table className="learning-attempt-table"><thead><tr><th>Case / stage</th><th>Split</th><th>Outcome</th><th>Terminal reward</th><th>Tokens / time</th><th>Evidence</th></tr></thead><tbody>{visible.map((attempt, index) => <tr key={attempt.id ?? `${attempt.stage}:${attempt.split}:${attempt.case_id}:${index}`}><th scope="row"><code>{attempt.case_id}</code><span>{stageLabels[attempt.stage]}</span>{attempt.id && <small>{attempt.id}</small>}</th><td>{splitLabels[attempt.split]}</td><td><span className={`outcome outcome-${attempt.status}`}><span className="status-dot" />{attemptLabels[attempt.status]}</span>{attempt.reason && <details className="learning-attempt-reason"><summary>Recorded reason</summary><p>{attempt.reason}</p></details>}</td><td className="mono">{attempt.reward == null ? ['success', 'failure'].includes(attempt.status) ? 'Not recorded' : 'Not scored' : attempt.reward.toFixed(3)}</td><td><span>{attempt.tokens == null ? 'Tokens not recorded' : `${integers.format(attempt.tokens)} tokens`}</span><span>{attempt.seconds == null ? 'Time not recorded' : `${decimals.format(attempt.seconds)} s`}</span></td><td>{attempt.inspect_log ? <a className="button subtle" href={inspectLogUrl(attempt.inspect_log)} target="_blank" rel="noreferrer" aria-label={`Inspect ${stageLabels[attempt.stage]}, ${splitLabels[attempt.split]}, ${attempt.case_id}`}>Inspect log ↗</a> : <span className="learning-no-log">{['pending', 'running'].includes(attempt.status) ? 'Awaiting log' : 'Log not recorded'}</span>}</td></tr>)}</tbody></table></div> : <div className="learning-empty"><p>{attempts.length ? 'No recorded attempts match these filters.' : 'No attempt records have been published yet.'}</p>{attempts.length > 0 && <button className="text-link" onClick={() => { setStage('all'); setSplit('all'); setQuery(''); }}>Clear filters</button>}</div>}
  </section>;
}

function Evaluation({ evaluation }: { evaluation?: LearningEvaluation }) {
  if (!evaluation) return <div className="learning-unrecorded"><strong>—</strong><span>No evaluation recorded for this split.</span></div>;
  const scored = evaluation.success + evaluation.failure;
  const resolved = scored + evaluation.limit + evaluation.error + evaluation.incomplete;
  return <>
    <div className="learning-score"><strong>{percentage(evaluation.success, scored)}</strong><span>Success / scored tasks<b>{evaluation.success} / {scored}</b></span></div>
    <div className="learning-coverage"><span>Scored coverage</span><b>{scored} / {evaluation.planned}</b><span>{percentage(scored, evaluation.planned)}</span></div>
    <div className="outcome-bar" aria-label={`${scored} of ${evaluation.planned} planned tasks scored`}>
      {(['success', 'failure', 'limit', 'error', 'incomplete'] as const).map(status => evaluation[status] > 0 && <span key={status} className={`segment-${status}`} style={{ width: `${evaluation[status] / Math.max(1, evaluation.planned) * 100}%` }} title={`${readable(status)}: ${evaluation[status]}`} />)}
    </div>
    <dl className="learning-outcomes">
      <div><dt>Success</dt><dd>{evaluation.success}</dd></div><div><dt>Scored failure</dt><dd>{evaluation.failure}</dd></div>
      <div><dt>Budget limit</dt><dd>{evaluation.limit}</dd></div><div><dt>Execution error</dt><dd>{evaluation.error}</dd></div>
      <div><dt>Incomplete</dt><dd>{evaluation.incomplete}</dd></div><div><dt>Remaining</dt><dd>{evaluation.planned - resolved}</dd></div>
    </dl>
    <dl className="learning-compute"><div><dt>Recorded tokens</dt><dd>{evaluation.tokens === undefined ? '—' : integers.format(evaluation.tokens)}</dd></div><div><dt>Recorded time</dt><dd>{evaluation.seconds === undefined ? '—' : `${decimals.format(evaluation.seconds / 60)} min`}</dd></div></dl>
    {evaluation.dataset_version && <p className="learning-dataset-version">Dataset <code>{evaluation.dataset_version}</code></p>}
  </>;
}

function StageCard({ id, stage, split }: { id: LearningStageId; stage?: LearningStage; split: LearningSplit }) {
  const evaluation = stage && stageEvaluations(stage).find(item => item.split === split);
  const notes = stage?.notes ? typeof stage.notes === 'string' ? [stage.notes] : stage.notes : [];
  const checkpoint = stage?.checkpoint;
  return <article className={`learning-stage learning-stage-${id}`}>
    <header><span className="section-label">Stage {stageOrder.indexOf(id) + 1}</span><span className={`learning-stage-state state-${stage?.status ?? 'pending'}`}><span className="status-dot" />{readable(stage?.status ?? 'pending')}</span></header>
    <h2>{stage?.label ?? stageLabels[id]}</h2><p className="learning-stage-description">{stageDescriptions[id]}</p>
    <Evaluation evaluation={evaluation} />
    <div className="learning-checkpoint"><span className="section-label">Checkpoint</span><code>{typeof checkpoint === 'string' ? checkpoint : checkpoint?.id ?? 'Not recorded'}</code>
      {checkpoint && typeof checkpoint !== 'string' && <details><summary>Checkpoint provenance</summary><dl>{checkpoint.parent && <div><dt>Parent</dt><dd><code>{checkpoint.parent}</code></dd></div>}{checkpoint.digest && <div><dt>Digest</dt><dd><code>{checkpoint.digest}</code></dd></div>}{checkpoint.path && <div><dt>Path</dt><dd><code>{checkpoint.path}</code></dd></div>}{checkpoint.step !== undefined && <div><dt>Training step</dt><dd>{checkpoint.step}</dd></div>}{checkpoint.created_at && <div><dt>Created</dt><dd>{time(checkpoint.created_at)}</dd></div>}</dl></details>}
    </div>
    {notes.length > 0 && <div className="learning-stage-notes">{notes.map((note, index) => <p key={index}>{note}</p>)}</div>}
  </article>;
}

function Curves({ points }: { points: LearningExport['curves'] }) {
  const groups = useMemo(() => [...new Set(points.map(point => JSON.stringify([point.metric, point.split ?? 'unspecified'])))], [points]);
  const [selection, setSelection] = useState('');
  const key = groups.includes(selection) ? selection : groups[0];
  const [metric, split] = key ? JSON.parse(key) as [string, string] : ['', ''];
  const selected = points.filter(point => point.metric === metric && (point.split ?? 'unspecified') === split);
  const series = stageOrder.map(stage => ({ stage, points: selected.filter(point => point.stage === stage).sort((a, b) => a.step - b.step) })).filter(item => item.points.length > 0);
  const minStep = Math.min(...selected.map(point => point.step), 0);
  const maxStep = Math.max(...selected.map(point => point.step), 1);
  const values = selected.map(point => point.value);
  const minValue = values.length ? Math.min(...values) : 0;
  const maxValue = values.length ? Math.max(...values) : 1;
  const margin = maxValue === minValue ? Math.max(Math.abs(minValue) * .1, .1) : (maxValue - minValue) * .1;
  const lower = minValue - margin;
  const upper = maxValue + margin;
  const x = (step: number) => 70 + ((step - minStep) / (maxStep - minStep)) * 680;
  const y = (value: number) => 210 - ((value - lower) / (upper - lower)) * 175;
  return <section className="learning-panel learning-curves"><header><div><h2>Learning curves</h2><p>Logged measurements only. Lines connect recorded points; metrics use their original scale.</p></div>{groups.length > 0 && <label>Metric<select value={key} onChange={event => setSelection(event.target.value)}>{groups.map(group => { const [name, dataset] = JSON.parse(group) as string[]; return <option value={group} key={group}>{readable(name)} · {dataset === 'unspecified' ? 'Split not recorded' : readable(dataset)}</option>; })}</select></label>}</header>
    {selected.length ? <><div className="learning-chart"><svg viewBox="0 0 800 255" role="img" aria-label={`${readable(metric)}, ${split === 'unspecified' ? 'split not recorded' : readable(split)}, across ${selected.length} recorded points`}>
      {[0, .5, 1].map(fraction => { const value = lower + fraction * (upper - lower); return <g key={fraction}><line x1="70" x2="750" y1={y(value)} y2={y(value)} className="learning-gridline" /><text x="58" y={y(value) + 4} textAnchor="end">{decimals.format(value)}</text></g>; })}
      <text x="70" y="234" textAnchor="middle">{minStep}</text><text x="750" y="234" textAnchor="middle">{maxStep}</text><text x="410" y="246" textAnchor="middle">Training step</text>
      {series.map(item => <g key={item.stage} className={`learning-series series-${item.stage}`}><polyline fill="none" strokeWidth="2" points={item.points.map(point => `${x(point.step)},${y(point.value)}`).join(' ')} />{item.points.map((point, index) => <circle key={`${point.step}-${index}`} cx={x(point.step)} cy={y(point.value)} r="3.5"><title>{`${stageLabels[item.stage]} · Step ${point.step}: ${point.value}`}</title></circle>)}</g>)}
    </svg></div><div className="learning-chart-legend">{series.map(item => <span key={item.stage}><i className={`series-${item.stage}`} />{stageLabels[item.stage]} · {item.points.length} points</span>)}</div><details className="learning-raw-points"><summary>Recorded measurements</summary><div className="table-scroll"><table><thead><tr><th>Stage</th><th>Step</th><th>Metric</th><th>Split</th><th>Value</th></tr></thead><tbody>{selected.map((point, index) => <tr key={index}><td>{stageLabels[point.stage]}</td><td>{point.step}</td><td><code>{point.metric}</code></td><td>{point.split ? readable(point.split) : 'Not recorded'}</td><td>{point.value}</td></tr>)}</tbody></table></div></details></> : <div className="learning-empty"><span className="section-label">Awaiting measurements</span><p>Training metrics will appear when a run records them.</p></div>}
  </section>;
}

export default function TrainingResults({ learning, refreshing, onRefresh }: { learning: LearningExport; refreshing: boolean; onRefresh: () => void }) {
  const [split, setSplit] = useState<LearningSplit>('validation');
  const sealed = learning.dataset.test_status === 'sealed';
  const events = [...learning.events].sort((a, b) => Date.parse(b.at) - Date.parse(a.at));
  return <section className="learning-workspace" aria-label="Training experiment">
    <header className="benchmark-header"><div><span className="section-label">DealRoom / learning experiment</span><h1>{learning.title}</h1><p>Separate interface improvements from learning. Compare the base model, a clearer workflow, supervised fine-tuning, and reinforcement learning on recorded evaluations.</p></div><div className="benchmark-header-actions"><span className={`learning-state state-${learning.status}`}><span className="status-dot" />{statusLabels[learning.status]}</span><button className="button" disabled={refreshing} onClick={onRefresh}>{refreshing ? 'Refreshing…' : 'Refresh experiment'}</button></div></header>
    <div className="learning-model"><span>Base learner</span><strong>{learning.model.label}</strong><code>{learning.model.id}</code><span className="badge">{learning.model.quantization}</span></div>
    <section className="learning-dataset" aria-label="Dataset separation"><div><span className="section-label">Training scenarios</span><strong>{integers.format(learning.dataset.train_count)}</strong><span>Used for learning</span></div><div><span className="section-label">Validation scenarios</span><strong>{integers.format(learning.dataset.validation_count)}</strong><span>Used for development and selection</span></div><div><span className="section-label">Held-out test scenarios</span><strong>{integers.format(learning.dataset.test_count)}</strong><span className="learning-test-status">{sealed ? 'Sealed · results not published' : 'Opened · recorded evaluation'}</span></div></section>
    <div className="learning-comparison-heading"><div><h2>Four-stage comparison</h2><p>Budgets, scoring, and data provenance are recorded below.</p></div><div className="learning-split-control" role="group" aria-label="Evaluation split"><button aria-pressed={split === 'validation'} onClick={() => setSplit('validation')}>Validation</button><button aria-pressed={split === 'sealed_test'} onClick={() => setSplit('sealed_test')}>Held-out test{sealed ? ' · sealed' : ''}</button></div></div>
    {split === 'sealed_test' && sealed && <div className="learning-notice">The test set is sealed. No test scores are available; training and checkpoint selection use the training and validation sets.</div>}
    <div className="learning-stage-grid">{stageOrder.map(id => <StageCard key={id} id={id} stage={learning.stages.find(stage => stage.id === id)} split={split} />)}</div>
    <p className="learning-method-note">Success rates use only success and scored failure. Limits, execution errors, and incomplete attempts remain unscored; coverage shows how much of each planned evaluation is scored. Partial coverage or different datasets cannot establish a fair improvement claim. The original two-model benchmark remains in Model results.</p>
    <Curves points={learning.curves} />
    <Attempts attempts={learning.attempts ?? []} />
    <div className="learning-bottom-grid"><section className="learning-panel"><header><div><h2>Experiment activity</h2><p>Recorded events, newest first.</p></div><span className="section-label">{events.length} events</span></header>{events.length ? <ol className="learning-events">{events.map((event, index) => <li key={`${event.at}-${index}`}><time dateTime={event.at}>{time(event.at)}</time><p>{event.message}</p></li>)}</ol> : <div className="learning-empty"><p>No events have been recorded.</p></div>}</section><section className="learning-panel"><header><div><h2>Interpretation and limitations</h2><p>Results are evidence from this recorded experiment.</p></div></header><div className="learning-limitations">{learning.limitations.length ? <ul>{learning.limitations.map((limitation, index) => <li key={index}>{limitation}</li>)}</ul> : <p>No experiment-specific limitations have been recorded.</p>}<p>Supervised training learns example actions. Reinforcement learning updates the policy using environment rewards. An RL improvement requires measured gains beyond the supervised checkpoint on the same held-out evaluation.</p></div></section></div>
    <details className="learning-provenance"><summary>Configuration and provenance</summary><dl><div><dt>Experiment</dt><dd><code>{learning.id}</code></dd></div><div><dt>Last update</dt><dd><time dateTime={learning.updated_at}>{time(learning.updated_at)}</time></dd></div><div><dt>Base revision</dt><dd><code>{learning.model.base_revision}</code></dd></div><div><dt>Dataset version</dt><dd><code>{learning.dataset.version}</code></dd></div><div><dt>Dataset fingerprint</dt><dd><code>{learning.dataset.fingerprint}</code></dd></div></dl><pre className="json" tabIndex={0}>{JSON.stringify(learning.config, null, 2)}</pre></details>
  </section>;
}
