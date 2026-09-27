import { useEffect, useMemo, useRef, useState } from 'react';
import type { ChangeEvent, KeyboardEvent, ReactNode } from 'react';
import { parseExport } from './types';
import type { Amendment, Diagnostic, Evidence, Run, RunExport } from './types';

const kindLabels: Record<Run['kind'], string> = {
  scripted_failure: 'Scripted failure', repaired_script: 'Repaired script', fixture_witness: 'Fixture validation', recorded_model: 'Recorded model run',
};
const dollars = (cents: number) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: cents % 100 === 0 ? 0 : 2 }).format(cents / 100);
const dateTime = (value: string, short = false) => new Intl.DateTimeFormat('en-US', {
  timeZone: 'America/New_York', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', ...(short ? {} : { timeZoneName: 'short' as const }),
}).format(new Date(value));
const humanize = (value: string) => value.replaceAll('_', ' ').replace(/^\w/, c => c.toUpperCase());

function Icon({ name, size = 18 }: { name: 'arrow' | 'upload' | 'document' | 'check' | 'minus' | 'clock' | 'branch' | 'chevron' | 'close'; size?: number }) {
  const paths: Record<typeof name, ReactNode> = {
    arrow: <><path d="M5 12h14M13 6l6 6-6 6" /></>,
    upload: <><path d="M12 16V3m-5 5 5-5 5 5M4 15v5h16v-5" /></>,
    document: <><path d="M7 3h7l5 5v13H5V3h2Zm7 0v6h5M9 13h6M9 17h6" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    minus: <path d="M6 12h12" />,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v6l4 2" /></>,
    branch: <><circle cx="6" cy="5" r="2" /><circle cx="6" cy="19" r="2" /><circle cx="18" cy="5" r="2" /><path d="M6 7v10m0-5h5a7 7 0 0 0 7-5" /></>,
    chevron: <path d="m9 5 7 7-7 7" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function Outcome({ success }: { success: boolean }) {
  return <span className={`outcome ${success ? 'success' : 'failure'}`}><span className="status-dot" />{success ? 'Success' : 'Failed'}</span>;
}

function Diagnostics({ items, compact = false }: { items: Diagnostic[]; compact?: boolean }) {
  if (!items.length) return <p className="muted">No evaluator diagnostics in this snapshot.</p>;
  return <div className={`diagnostics ${compact ? 'compact' : ''}`}>{items.map((item, index) => <details key={`${item.predicate}-${index}`} className={`diagnostic ${item.passed ? 'passed' : 'failed'}`} open={!compact && !item.passed}>
    <summary><span className="predicate-icon"><Icon name={item.passed ? 'check' : 'minus'} size={14} /></span><span>{humanize(item.predicate)}</span><span className="predicate-status">{item.passed ? 'Pass' : 'Fail'}</span></summary>
    <div className="diagnostic-body"><p>{item.reason}</p>{item.evidence.length > 0 && <div className="evidence-ids">{item.evidence.map((id, i) => <code key={`${id}-${i}`}>{id}</code>)}</div>}</div>
  </details>)}</div>;
}

function AmendmentCard({ amendment }: { amendment: Amendment }) {
  return <article className="amendment-card">
    <div className="amendment-heading"><div><div className="eyebrow">Immutable revision</div><h4>{amendment.id}</h4></div><span className="plain-badge">{humanize(amendment.status)}</span></div>
    <div className="amendment-terms"><span>{humanize(amendment.kind)}</span><strong>{amendment.credit_cents !== null ? dollars(amendment.credit_cents) : amendment.deadline ? dateTime(amendment.deadline) : 'See document terms'}</strong></div>
    {amendment.deadline && amendment.credit_cents !== null && <p className="small muted">Deadline term: {dateTime(amendment.deadline)}</p>}
    <div className="signatures"><div className="small muted">Required parties · this revision only</div>{amendment.required_signers.map(party => {
      const signature = amendment.signatures.find(s => s.party === party);
      return <div className={`signer ${signature ? 'signed' : 'unsigned'}`} key={party}><span className="signer-name"><span className="signature-symbol">{signature ? <Icon name="check" size={12} /> : <Icon name="minus" size={12} />}</span>{humanize(party)}</span><span>{signature ? <><strong>Signed</strong><time dateTime={signature.time}>{dateTime(signature.time, true)}</time></> : <strong>Signature absent</strong>}</span></div>;
    })}{amendment.required_signers.length === 0 && <p className="small muted">No required signers recorded.</p>}</div>
  </article>;
}

function EvidenceBrowser({ documents, messages, authority }: { documents: Evidence[]; messages: Evidence[]; authority: Record<string, unknown> }) {
  const [tab, setTab] = useState<'documents' | 'messages' | 'authority'>('documents');
  const [selectedId, setSelectedId] = useState<string>('');
  const tabs = useRef<(HTMLButtonElement | null)[]>([]);
  const items = tab === 'messages' ? messages : documents;
  const selected = items.find(item => item.id === selectedId) ?? items[0];
  const tabNames = ['documents', 'messages', 'authority'] as const;
  function onTabKey(event: KeyboardEvent, index: number) {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? 2 : (index + (event.key === 'ArrowRight' ? 1 : -1) + 3) % 3;
    setTab(tabNames[next]);
    tabs.current[next]?.focus();
  }
  return <section className="panel evidence-panel" aria-label="Public evidence">
    <div className="panel-heading"><div><span className="eyebrow">Agent-visible</span><h3>Public evidence</h3></div><span className="plain-badge">At selected step</span></div>
    <div className="tabs" role="tablist" aria-label="Evidence type">{tabNames.map((name, i) => <button ref={el => { tabs.current[i] = el; }} id={`tab-${name}`} aria-controls="evidence-content" role="tab" aria-selected={tab === name} tabIndex={tab === name ? 0 : -1} className={tab === name ? 'active' : ''} onClick={() => setTab(name)} onKeyDown={e => onTabKey(e, i)} key={name}>{humanize(name)}{name !== 'authority' && <span>{name === 'documents' ? documents.length : messages.length}</span>}</button>)}</div>
    <div id="evidence-content" role="tabpanel" aria-labelledby={`tab-${tab}`} tabIndex={0}>
      {tab === 'authority' ? <div className="authority-content"><p className="small muted">Public buyer authority recorded by the engine at this step.</p><dl className="authority-list">{Object.entries(authority).map(([key, value]) => <div key={key}><dt>{humanize(key)}</dt><dd>{typeof value === 'object' ? <pre>{JSON.stringify(value, null, 2)}</pre> : String(value ?? 'None')}</dd></div>)}</dl>{Object.keys(authority).length === 0 && <p className="muted">No public authority record yet.</p>}</div>
        : items.length === 0 ? <div className="empty-inline"><Icon name="document" size={26} /><h4>No {tab} available yet</h4><p>Move forward in the replay to inspect newly available evidence.</p></div>
          : <div className="evidence-browser"><div className="evidence-list" aria-label={`${humanize(tab)} list`}>{items.map(item => <button className={selected?.id === item.id ? 'selected' : ''} onClick={() => setSelectedId(item.id)} aria-pressed={selected?.id === item.id} key={item.id}><Icon name={tab === 'messages' ? 'clock' : 'document'} size={16} /><span><strong>{item.title}</strong><code>{item.id}</code></span><Icon name="chevron" size={12} /></button>)}</div>{selected && <article className="document-detail"><div className="document-meta"><code>{selected.id}</code>{selected.time && <time dateTime={selected.time}>{dateTime(selected.time)}</time>}</div><h4>{selected.title}</h4><div className="document-body">{selected.body}</div></article>}</div>}
    </div>
  </section>;
}

function Comparison({ run, pair, position, onInspect }: { run: Run; pair: Run; position: number; onInspect: (run: Run, index: number) => void }) {
  const [expanded, setExpanded] = useState(true);
  const branch = run.branch_step ?? pair.branch_step ?? 0;
  const runs = [run, pair].sort((a, b) => (a.kind === 'scripted_failure' ? -1 : b.kind === 'scripted_failure' ? 1 : 0));
  return <section className="panel comparison-panel">
    <div className="panel-heading"><div><span className="eyebrow">Same state. Different continuation.</span><h3><Icon name="branch" />The decision boundary</h3></div><button className="text-button" aria-expanded={expanded} aria-controls="comparison-content" onClick={() => setExpanded(value => !value)}>{expanded ? 'Collapse' : 'Expand'} comparison</button></div>
    {expanded && <div id="comparison-content"><p className="comparison-note">Shared prefix through step {branch}. Each column shows the same number of steps past the branch, stopping at its final step.</p><div className="comparison-grid">{runs.map(item => {
      const itemBranch = item.steps.findIndex(s => s.index === (item.branch_step ?? branch));
      const runBranch = run.steps.findIndex(s => s.index === branch);
      const index = Math.min(item.steps.length - 1, Math.max(0, (itemBranch < 0 ? branch : itemBranch) + position - (runBranch < 0 ? branch : runBranch)));
      const step = item.steps[index];
      const isSelected = item.id === run.id;
      return <article className={`comparison-card ${item.kind === 'scripted_failure' ? 'failure-card' : 'repair-card'}`} key={item.id}><div className="comparison-card-heading"><strong>{kindLabels[item.kind]}</strong><span className="small">Final result <Outcome success={item.score.success} /></span></div><div className="comparison-step">Step {step.index} · {dateTime(step.state.time)}</div><h4>{step.label}</h4><dl className="comparison-facts"><div><dt>Effective credit</dt><dd>{dollars(step.state.effective_credit_cents)}</dd></div><div><dt>Disposition</dt><dd>{humanize(step.state.disposition)}</dd></div></dl><p className="comparison-explanation">{step.reviewer.note || step.reviewer.diagnostics.find(d => !d.passed)?.reason || 'See evaluator predicates for this snapshot.'}</p><button className="comparison-inspect" onClick={() => onInspect(item, index)} aria-pressed={isSelected}>{isSelected ? 'Viewing this continuation' : 'Inspect this continuation'}{!isSelected && <Icon name="arrow" size={16} />}</button></article>;
    })}</div></div>}
  </section>;
}

function Replay({ data, run, position, setRun, setPosition }: { data: RunExport; run: Run; position: number; setRun: (run: Run, position?: number) => void; setPosition: (position: number) => void }) {
  const step = run.steps[Math.min(position, run.steps.length - 1)];
  const [showAction, setShowAction] = useState(false);
  const pair = data.runs.find(item => item.id === run.paired_run_id);
  const isFinal = position === run.steps.length - 1;
  const branch = run.branch_step ?? pair?.branch_step;
  const reachedBranch = branch !== undefined && step.index >= branch;
  return <div className="replay">
    <div className="run-heading"><div><div className="run-kicker"><span className="eyebrow">{humanize(run.split)} case</span><span className="kind-label">{kindLabels[run.kind]}</span></div><h2>{run.case_title}</h2><p>{run.description}</p></div><div className="final-outcome"><span className="eyebrow">Terminal result</span><Outcome success={run.score.success} /></div></div>
    <section className="panel timeline-panel" aria-label="Replay timeline">
      <div className="timeline-top"><div className="step-label"><span className="step-number">{String(step.index).padStart(2, '0')}</span><div><span className="eyebrow">{position === 0 ? 'Initial state' : isFinal ? 'Final recorded step' : 'Selected step'}</span><h3>{step.label}</h3></div></div><div className="timeline-controls"><button className="icon-button" title="First step" aria-label="First step" disabled={position === 0} onClick={() => setPosition(0)}>│‹</button><button className="icon-button" title="Previous step" aria-label="Previous step" disabled={position === 0} onClick={() => setPosition(position - 1)}>‹</button><span className="step-counter">{position + 1}<span> / {run.steps.length}</span></span><button className="icon-button" title="Next step" aria-label="Next step" disabled={isFinal} onClick={() => setPosition(position + 1)}>›</button><button className="icon-button" title="Final step" aria-label="Final step" disabled={isFinal} onClick={() => setPosition(run.steps.length - 1)}>›│</button></div></div>
      <div className="timeline-track"><input type="range" min={0} max={run.steps.length - 1} value={position} onChange={e => setPosition(Number(e.target.value))} aria-label="Replay step" aria-valuetext={`Step ${step.index}: ${step.label}`} style={{ '--progress': `${run.steps.length > 1 ? position / (run.steps.length - 1) * 100 : 0}%` } as React.CSSProperties} /><div className="timeline-ticks">{run.steps.map((item, i) => <button key={`${item.index}-${i}`} className={`${i === position ? 'current' : ''} ${branch === item.index ? 'branch-tick' : ''}`} aria-label={`Go to step ${item.index}: ${item.label}`} aria-current={i === position ? 'step' : undefined} title={`${item.index} · ${item.label}`} onClick={() => setPosition(i)}><span />{branch === item.index && <span className="branch-caption">Branch</span>}</button>)}</div></div>
      <div className="timeline-bottom"><span><Icon name="clock" size={14} /><time dateTime={step.state.time}>{dateTime(step.state.time)}</time></span><span>Contractual deadline <strong>{dateTime(step.state.deadline)}</strong></span><button className="text-button" aria-expanded={showAction} aria-controls="action-detail" onClick={() => setShowAction(v => !v)}>{showAction ? 'Hide' : 'Inspect'} action ↗</button></div>
      {showAction && <div id="action-detail" className="action-detail"><div><span className="eyebrow">Recorded action</span><pre>{step.action ? JSON.stringify(step.action, null, 2) : 'No action. Initial observation.'}</pre></div><div><span className="eyebrow">Engine response</span><pre>{step.result ? JSON.stringify(step.result, null, 2) : 'Initial state.'}</pre></div></div>}
    </section>
    <div className="workspace-grid"><div className="public-column">
      <div className="state-strip" aria-label="Engine state at selected step"><div><span className="eyebrow">Effective credit</span><strong>{dollars(step.state.effective_credit_cents)}</strong><span>Executed terms only</span></div><div><span className="eyebrow">Eligible residual</span><strong className={step.state.residual_cents === null ? 'unknown-value' : ''}>{step.state.residual_cents === null ? 'Awaiting quote' : dollars(step.state.residual_cents)}</strong><span>{step.state.residual_cents === null ? 'No public quote yet' : 'From engine state'}</span></div><div><span className="eyebrow">Disposition</span><strong className="disposition-value">{humanize(step.state.disposition)}</strong><span>At selected step</span></div></div>
      <section className="panel execution-panel"><div className="panel-heading"><div><span className="eyebrow">Execution evidence</span><h3>Amendments & signatures</h3></div><span className="count-badge">{step.state.amendments.length}</span></div>{step.state.amendments.length ? <div className="amendments-grid">{step.state.amendments.map(a => <AmendmentCard key={a.id} amendment={a} />)}</div> : <div className="no-amendment"><span className="empty-document"><Icon name="document" size={22} /></span><div><strong>No amendment revisions yet</strong><p>A conversation can agree on terms. An effective amendment needs execution evidence.</p></div></div>}</section>
      <EvidenceBrowser documents={step.state.documents} messages={step.state.messages} authority={step.state.authority} />
      {pair && reachedBranch && <Comparison run={run} pair={pair} position={position} onInspect={setRun} />}
      {pair && !reachedBranch && <div className="branch-invitation"><Icon name="branch" /><div><strong>A shared beginning</strong><p>The failed and repaired scripts share this exact prefix. Their continuations become comparable at step {branch}.</p></div><button className="text-button" onClick={() => setPosition(Math.max(0, run.steps.findIndex(s => s.index === branch)))}>Go to branch <Icon name="arrow" size={15} /></button></div>}
    </div><aside className="reviewer-panel" aria-label="Reviewer-only explanation"><div className="reviewer-heading"><span className="reviewer-icon">R</span><span className="eyebrow">Reviewer only</span><span className="plain-badge">Not agent-visible</span></div><h3>{isFinal ? 'The outcome, explained.' : 'What the evidence supports.'}</h3><p className="reviewer-note">{step.reviewer.note || 'Evaluator diagnostics below refer to the selected engine state.'}</p><div className="reviewer-divider" /><div className="diagnostics-heading"><span className="eyebrow">{isFinal ? 'Terminal predicates' : 'Predicates at this step'}</span><span className="small muted">Engine evaluated</span></div><Diagnostics items={isFinal ? run.score.diagnostics : step.reviewer.diagnostics} /><p className="reviewer-footnote">These checks inspect underlying records and action history. A status label or final assertion cannot create a signature.</p><div className="run-provenance"><span className="eyebrow">Run provenance</span><strong>{kindLabels[run.kind]}</strong><p>{run.kind === 'fixture_witness' ? 'Fixture validation may use full scenario knowledge. This is not agent performance.' : run.kind === 'recorded_model' ? 'Derived from a recorded Inspect evaluation. Open the native Inspect viewer for the complete model transcript.' : 'Deliberately authored demonstration, executed by the domain engine. This is not evidence of model performance.'}</p><code>{run.id}</code></div></aside></div>
  </div>;
}

export default function App() {
  const [data, setData] = useState<RunExport | null>(null);
  const [runId, setRunId] = useState('');
  const [position, setPosition] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [source, setSource] = useState('Bundled engine export');
  const inputRef = useRef<HTMLInputElement>(null);
  const run = data?.runs.find(item => item.id === runId) ?? data?.runs[0];
  const cases = useMemo(() => data ? [...new Map(data.runs.map(item => [item.case_id, item])).values()] : [], [data]);
  const issueCounts = data?.summary ? ['execution_errors', 'budget_exhaustions', 'incomplete', 'run_errors', 'cancelled_runs'].flatMap(key => {
    const value = data.summary?.[key];
    return typeof value === 'number' && value > 0 ? [`${value} ${humanize(key).toLowerCase()}`] : [];
  }) : [];
  function acceptData(parsed: RunExport, name: string) {
    const first = parsed.runs.find(item => item.kind === 'scripted_failure') ?? parsed.runs[0];
    setData(parsed); setRunId(first?.id ?? ''); setPosition(0); setSource(name); setError('');
  }
  async function loadDefault() {
    setLoading(true);
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}runs.json`);
      if (!response.ok) throw new Error(`Could not load runs.json (HTTP ${response.status}). Generate offline traces or open an exported JSON file.`);
      if (!response.headers.get('content-type')?.includes('json')) throw new Error('The bundled runs.json export is missing. Generate offline traces with python -m dealroom.demo, or open a DealRoom export.');
      acceptData(parseExport(await response.json()), 'Bundled engine export');
    } catch (err) { setError(err instanceof Error ? err.message : 'Could not load replay data.'); }
    finally { setLoading(false); }
  }
  useEffect(() => { void loadDefault(); }, []);
  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      if (file.size > 50 * 1024 * 1024) throw new Error('This export exceeds 50 MB. Export a smaller set of runs and try again.');
      acceptData(parseExport(JSON.parse(await file.text())), file.name);
    } catch (err) { setError(err instanceof SyntaxError ? 'This file is not valid JSON. Select a DealRoom replay export.' : err instanceof Error ? err.message : 'Could not read this export.'); }
    event.target.value = '';
  }
  function selectRun(next: Run, nextPosition = 0) { setRunId(next.id); setPosition(Math.min(nextPosition, next.steps.length - 1)); }
  return <>
    <header className="site-header"><a className="brand" href="#main"><span className="brand-mark"><span /><span /><span /></span><span>DealRoom<span className="brand-period">.</span></span></a><div className="header-context">Transaction intelligence <span>/</span> Evaluation environment</div><button className="button secondary upload-button" onClick={() => inputRef.current?.click()}><Icon name="upload" size={15} />Open export</button><input ref={inputRef} type="file" accept=".json,application/json" onChange={upload} className="visually-hidden" aria-label="Open DealRoom JSON export" /></header>
    <main id="main" className="app-shell"><section className="page-intro"><div><div className="eyebrow intro-eyebrow"><span />Replay inspector</div><h1>From agreement to evidence.</h1><p>Follow the decisions. Inspect the signatures. See what actually executed.</p></div><div className="environment-note"><span className="tiny-dot" />Synthetic transactions<p>Deterministic engine · Objective evaluation</p></div></section>
      {error && <div role="alert" className="error-banner"><div><strong>Replay could not be loaded</strong><p>{error}</p>{data && <p>Your previously loaded export is still available below.</p>}</div><button className="text-button" onClick={() => void loadDefault()}>Reload bundled export</button><button className="icon-button" aria-label="Dismiss error" onClick={() => setError('')}><Icon name="close" size={15} /></button></div>}
      {issueCounts.length > 0 && <div className="run-issues" role="status"><strong>Not all samples reached a completed outcome.</strong><p>{issueCounts.join(' · ')}{typeof data?.summary?.total_samples === 'number' ? ` · ${data.summary.total_samples} total samples` : ''}. These samples are counted separately from scored replay runs.</p></div>}
      {loading && !data ? <div className="empty-page" role="status"><span className="loader" /><h2>Loading engine traces</h2><p>Preparing the recorded evidence and outcomes.</p></div> : !run || !data ? <div className="empty-page"><Icon name="document" size={32} /><h2>{data ? 'No runs in this export' : 'Ready for a replay'}</h2><p>Load a DealRoom JSON export to inspect a completed engine run.</p><button className="button primary" onClick={() => inputRef.current?.click()}>Open export <Icon name="upload" size={16} /></button><p className="small muted">Generate the bundled demonstration with <code>python -m dealroom.demo</code>.</p></div> : <div className="app-grid"><nav className="case-sidebar" aria-label="Cases and runs"><div className="sidebar-heading"><span className="eyebrow">Case library</span><span>{String(cases.length).padStart(2, '0')}</span></div><div className="case-list">{cases.map((item, index) => <button key={item.case_id} className={`case-button ${run.case_id === item.case_id ? 'active' : ''}`} onClick={() => { const candidate = data.runs.find(r => r.case_id === item.case_id && r.kind === 'scripted_failure') ?? data.runs.find(r => r.case_id === item.case_id)!; selectRun(candidate); }} aria-current={run.case_id === item.case_id ? 'true' : undefined}><span className="case-number">{String(index + 1).padStart(2, '0')}</span><span><strong>{item.case_title}</strong><span className="case-split">{item.split}</span></span>{run.case_id === item.case_id && <span className="case-active-dot" />}</button>)}</div><div className="run-selection"><label className="eyebrow" htmlFor="run-select">Recorded run</label><select id="run-select" value={run.id} onChange={e => selectRun(data.runs.find(item => item.id === e.target.value)!)}>{data.runs.filter(item => item.case_id === run.case_id).map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select><p>{kindLabels[run.kind]}</p></div><div className="sidebar-note"><Icon name="branch" size={18} /><strong>Evidence changes the outcome.</strong><p>Use the step controls to follow each action and the records it leaves behind.</p><span>Full model transcripts live in Inspect.</span></div></nav><Replay key={run.id} data={data} run={run} position={position} setRun={selectRun} setPosition={setPosition} /></div>}
      <footer className="site-footer"><span>DealRoom <span> / </span> Inspection, amendments & execution</span><span>{data ? `${data.runs.length} recorded runs · ${source}` : 'Static replay inspector'}{data && <span className="export-date"> · Exported {dateTime(data.generated_at)}</span>}</span></footer>
    </main>
  </>;
}
