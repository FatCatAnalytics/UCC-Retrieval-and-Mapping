import { useEffect, useState } from "react";
import { EngineClient, native } from "../api";
import type { Decision, DecisionCounts, OutputFile, Page, RunSnapshot, Summary } from "../types";

interface Props {
  client: EngineClient;
  run: RunSnapshot;
}

type Tab = "summary" | "review" | "clusters" | "outputs";
const BANDS = ["REVIEW_HIGH", "REVIEW_MEDIUM", "REVIEW_LOW"];

export function ResultsView({ client, run }: Props) {
  const [tab, setTab] = useState<Tab>("summary");
  const [summary, setSummary] = useState<Summary | null>(run.summary);
  const [pending, setPending] = useState<number | null>(null);

  useEffect(() => {
    setSummary(run.summary);
    if (!run.summary && run.status === "succeeded") client.summary(run.id).then(setSummary).catch(() => {});
  }, [client, run.id, run.summary, run.status]);
  useEffect(() => {
    client.decisions(run.id).then((d) => setPending(d.counts.pending)).catch(() => setPending(null));
  }, [client, run.id, tab]);

  return (
    <div className="page">
      <header className="page-head">
        <h1>{run.name}</h1>
        <p className="muted">
          Finished {run.finished_at?.replace("T", " ").slice(0, 19)} · {summary?.total_seconds ?? "—"}s total
          {run.sample_rows ? ` · sample of ${run.sample_rows.toLocaleString()} rows` : ""}
        </p>
      </header>

      <nav className="tabs">
        {(["summary", "review", "clusters", "outputs"] as Tab[]).filter((t) => t !== "clusters" || !summary || summary.clusters > 0).map((t) => (
          <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>
            {t === "summary" && "Summary"}
            {t === "review" && `Review queue${pending != null ? ` (${pending.toLocaleString()} to decide)` : summary ? ` (${summary.review_needed.toLocaleString()})` : ""}`}
            {t === "clusters" && `New-entity clusters${summary ? ` (${summary.clusters.toLocaleString()})` : ""}`}
            {t === "outputs" && "Output files"}
          </button>
        ))}
      </nav>

      {tab === "summary" && <SummaryTab run={run} summary={summary} client={client} />}
      {tab === "review" && <ReviewTab run={run} client={client} />}
      {tab === "clusters" && <ClustersTab run={run} client={client} />}
      {tab === "outputs" && <OutputsTab run={run} client={client} />}
    </div>
  );
}

// ------------------------------------------------------------------ summary
function SummaryTab({ run, summary, client }: { run: RunSnapshot; summary: Summary | null; client: EngineClient }) {
  const [dest, setDest] = useState(run.source.type === "csv" ? run.source.master_path : "");
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [counts, setCounts] = useState<DecisionCounts | null>(null);
  useEffect(() => { client.decisions(run.id).then((d) => setCounts(d.counts)).catch(() => setCounts(null)); }, [client, run.id]);
  if (!summary) return <p className="muted">Summary not available.</p>;

  const pct = (n: number) => (summary.total_records ? `${((n / summary.total_records) * 100).toFixed(1)}%` : "");
  const pendingApproved = counts?.unapplied_approved ?? 0;
  const promote = async () => {
    if (!dest) return;
    const extra = pendingApproved ? `\n\n${pendingApproved} approved review row(s) not yet in the file will be added first.` : "";
    if (!confirm(`Overwrite\n${dest}\nwith this run's updated master? A timestamped backup is kept.${extra}`)) return;
    setBusy(true);
    setMsg(null);
    try {
      const r = await client.promoteMaster(run.id, dest);
      setMsg(`Promoted → ${r.destination}${r.backup ? ` (backup: ${r.backup})` : ""}`);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="cards">
        <Stat label="UCC records" value={summary.total_records} />
        <Stat label="Auto-mapped" value={summary.auto_mapped} sub={pct(summary.auto_mapped)} tone="good" />
        <Stat label="Review needed" value={summary.review_needed}
          sub={counts && counts.pending !== summary.review_needed ? `${counts.pending.toLocaleString()} still to decide` : pct(summary.review_needed)} tone="warn" />
        <Stat label="New entities" value={summary.new_entities}
          sub={summary.clusters > 0 ? `${summary.clusters.toLocaleString()} clusters` : `${pct(summary.new_entities)} · not in master`} />
        <Stat label="With Firm ID" value={summary.records_with_firm_id} sub={`${summary.unique_firm_ids.toLocaleString()} firms`} />
      </div>

      <div className="split">
        <section className="card">
          <h4>By match method</h4>
          <Bars data={summary.by_match_method} total={summary.total_records} />
        </section>
        <section className="card">
          <h4>By confidence</h4>
          <Bars data={summary.by_confidence} total={summary.total_records} />
        </section>
      </div>

      <section className="card">
        <h4>Updated master</h4>
        {summary.updated_master ? (
          <p>
            {summary.updated_master.rows.toLocaleString()} rows ={" "}
            {summary.updated_master.original_rows.toLocaleString()} original +{" "}
            <strong>{summary.updated_master.added_rows.toLocaleString()} newly resolved</strong> (Stages 1–3)
            {summary.updated_master.approved_rows ? (
              <> + <strong>{summary.updated_master.approved_rows.toLocaleString()} approved in review</strong></>
            ) : null}
            {". "}The original master always wins on conflicts.
          </p>
        ) : (
          <p className="muted">Not produced.</p>
        )}
        <div className="row gap">
          <input className="grow" value={dest} onChange={(e) => setDest(e.target.value)} placeholder="Destination Master Mapping CSV" />
          {native.inTauri && (
            <button className="btn" onClick={async () => {
              const p = await native.pickSaveFile("Save updated master as…", "Master Mapping.csv");
              if (p) setDest(p);
            }}>Choose…</button>
          )}
          <button className="btn btn-primary" disabled={!dest || busy || !summary.updated_master} onClick={promote}>
            {busy ? "Promoting…" : "Promote as new master"}
          </button>
        </div>
        {counts && counts.total ? (
          <p className="small muted">
            Review queue: {counts.approved} approved, {counts.rejected} rejected (incl. decisions from earlier runs), {counts.pending} still to decide.
            {pendingApproved ? ` ${pendingApproved} approved row(s) are not in this file yet — promoting adds them.` : ""}
          </p>
        ) : null}
        {msg && <p className="small">{msg}</p>}
      </section>

      <section className="card">
        <h4>Stage timings</h4>
        <Bars data={summary.stage_seconds} total={summary.total_seconds} unit="s" />
      </section>
    </>
  );
}

function Stat({ label, value, sub, tone }: { label: string; value: number; sub?: string; tone?: "good" | "warn" }) {
  return (
    <div className={`stat ${tone ?? ""}`}>
      <div className="stat-value">{value.toLocaleString()}</div>
      <div className="stat-label">{label}</div>
      {sub && <div className="muted small">{sub}</div>}
    </div>
  );
}

function Bars({ data, total, unit = "" }: { data: Record<string, number>; total: number; unit?: string }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...entries.map(([, v]) => v));
  return (
    <div className="bars">
      {entries.map(([k, v]) => (
        <div key={k} className="bar-row">
          <span className="bar-label" title={k}>{k}</span>
          <div className="bar"><div style={{ width: `${(v / max) * 100}%` }} /></div>
          <span className="bar-val">
            {unit === "s" ? `${v.toFixed(1)}s` : v.toLocaleString()}
            {total && unit !== "s" ? <span className="muted"> · {((v / total) * 100).toFixed(1)}%</span> : null}
          </span>
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------- review
type DecisionFilter = "" | "pending" | "approved" | "rejected";

function ReviewTab({ run, client }: { run: RunSnapshot; client: EngineClient }) {
  const [bands, setBands] = useState<string[]>(BANDS);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<DecisionFilter>("pending");
  const [page, setPage] = useState<Page | null>(null);
  const [offset, setOffset] = useState(0);
  const [tick, setTick] = useState(0);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const limit = 100;

  useEffect(() => {
    const t = setTimeout(() => {
      client.review(run.id, { confidence: bands, search, decision: filter, offset, limit }).then(setPage).catch(() => setPage(null));
    }, 200);
    return () => clearTimeout(t);
  }, [client, run.id, bands, search, filter, offset, tick]);

  const counts = page?.decision_counts;
  const refresh = () => setTick((t) => t + 1);

  const decide = async (items: { key: string; decision: Decision | null; mapped_name?: string | null }[]) => {
    setBusy("decide");
    setMsg(null);
    try {
      await client.setDecisions(run.id, items);
      setEdits((e) => { const n = { ...e }; items.forEach((i) => delete n[i.key]); return n; });
      refresh();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  };

  const apply = async () => {
    setBusy("apply");
    setMsg(null);
    try {
      const r = await client.applyDecisions(run.id);
      setMsg(`Added ${r.added.toLocaleString()} approved row(s) to the updated master → ${r.file.split(/[\\/]/).pop()} (${r.rows.toLocaleString()} rows).`);
      refresh();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  };

  const rows = page?.rows ?? [];
  const undecidedShown = rows.filter((r) => !r.decision);
  const str = (v: unknown) => (v == null ? "" : String(v));
  const num = (v: unknown) => (typeof v === "number" ? v.toFixed(3) : str(v));

  return (
    <>
      <div className="row gap wrap">
        {BANDS.map((b) => (
          <label key={b} className="chip">
            <input type="checkbox" checked={bands.includes(b)}
              onChange={(e) => { setOffset(0); setBands(e.target.checked ? [...bands, b] : bands.filter((x) => x !== b)); }} />
            {b}
          </label>
        ))}
        <div className="seg small">
          {([["", "All"], ["pending", "To decide"], ["approved", "Approved"], ["rejected", "Rejected"]] as [DecisionFilter, string][]).map(([v, l]) => (
            <button key={v} className={filter === v ? "on" : ""} onClick={() => { setOffset(0); setFilter(v); }}>
              {l}{counts && v ? ` (${counts[v]})` : ""}
            </button>
          ))}
        </div>
        <input className="grow" placeholder="Search secured party / mapped name…" value={search}
          onChange={(e) => { setOffset(0); setSearch(e.target.value); }} />
      </div>

      <div className="row between wrap decision-bar">
        <span className="small muted">
          Approve a suggestion to add <em>SECP_NAME → mapped name (Firm ID)</em> to the updated master. Edit the mapped name first if the
          suggestion is close but not right; the Firm ID is re-looked-up from the Universe. Decisions are remembered across runs:
          anything you approve or reject here stays out of "To decide" in every later run.
        </span>
        <div className="row gap">
          <button className="btn" disabled={busy !== "" || undecidedShown.length === 0}
            onClick={() => decide(undecidedShown.map((r) => ({ key: str(r.key), decision: "approved" as Decision, mapped_name: edits[str(r.key)] })))}>
            Approve all shown ({undecidedShown.length})
          </button>
          <button className="btn btn-primary" disabled={busy !== "" || !counts?.unapplied_approved} onClick={apply}>
            {busy === "apply" ? "Adding…" : `Add ${counts?.unapplied_approved ?? 0} approved to updated master`}
          </button>
        </div>
      </div>
      {msg && <p className="small">{msg}</p>}

      <div className="table-wrap">
        <table className="table review">
          <thead>
            <tr>
              <th>Secured party (as filed)</th><th>States</th><th>Suggested mapping</th><th>Method · score</th><th>Firm ID</th><th>Decision</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const key = str(r.key);
              const decision = r.decision as Decision | null;
              const suggested = str(r.SECP_NAME_MAPPED);
              const value = edits[key] ?? str(r.decided_mapped_name) ?? "";
              return (
                <tr key={key} className={decision ? `decided ${decision}` : ""}>
                  <td title={str(r.SECP_NAME)}>
                    <div className="cell-main">{str(r.SECP_NAME)}</div>
                    {r.matched_master_raw ? <div className="muted small">master: {str(r.matched_master_raw)}</div> : null}
                  </td>
                  <td className="muted small">{str(r.DEBT_STATE)} → {str(r.SECP_STATE)}</td>
                  <td>
                    <input className="inline" value={value || suggested} placeholder={suggested}
                      disabled={decision === "approved" || busy !== ""}
                      onChange={(e) => { const v = e.target.value; setEdits((prev) => ({ ...prev, [key]: v })); }} />
                    {r.decided_mapped_name ? <div className="small muted">suggested: {suggested}</div> : null}
                    {r.previously_rejected ? (
                      <div className="small warn" title="A different suggestion for this filer was rejected before; this one is new.">
                        previously rejected: {str(r.previously_rejected)}
                      </div>
                    ) : null}
                  </td>
                  <td className="small">
                    <div>{str(r.match_method)}</div>
                    <div className="muted">{num(r.match_score)} · {str(r.confidence).replace("REVIEW_", "")}</div>
                  </td>
                  <td className="small">
                    {str(decision ? r.decided_firm_id : r.Firm_ID) || (
                      <span className="muted" title={decision === "approved" ? "Corrected name not found in the Universe – Firm ID left blank" : undefined}>—</span>
                    )}
                  </td>
                  <td>
                    {decision ? (
                      <div className="row gap">
                        <span className={`pill ${decision}`} title={r.decided_in_run && r.decided_in_run !== run.id ? `decided in run ${str(r.decided_in_run)}` : undefined}>
                          {decision}{r.decided_in_run && r.decided_in_run !== run.id ? " ·  earlier run" : ""}
                        </span>
                        <button className="link small" disabled={busy !== ""} onClick={() => decide([{ key, decision: null }])}>undo</button>
                      </div>
                    ) : (
                      <div className="row gap">
                        <button className="btn ok" disabled={busy !== ""}
                          onClick={() => decide([{ key, decision: "approved", mapped_name: edits[key] }])}>Approve</button>
                        <button className="btn" disabled={busy !== ""} onClick={() => decide([{ key, decision: "rejected" }])}>Reject</button>
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
            {page && rows.length === 0 && <tr><td colSpan={6} className="muted">No rows.</td></tr>}
            {!page && <tr><td colSpan={6} className="muted">Loading…</td></tr>}
          </tbody>
        </table>
      </div>
      {page && (
        <div className="row between small">
          <span className="muted">{page.total.toLocaleString()} rows · showing {Math.min(offset + 1, page.total)}–{Math.min(offset + limit, page.total)}</span>
          <div className="row gap">
            <button className="btn" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>‹ Prev</button>
            <button className="btn" disabled={offset + limit >= page.total} onClick={() => setOffset(offset + limit)}>Next ›</button>
          </div>
        </div>
      )}
    </>
  );
}

// ----------------------------------------------------------------- clusters
function ClustersTab({ run, client }: { run: RunSnapshot; client: EngineClient }) {
  const [search, setSearch] = useState("");
  const [minSize, setMinSize] = useState(2);
  const [page, setPage] = useState<Page | null>(null);
  const [offset, setOffset] = useState(0);
  const limit = 100;

  useEffect(() => {
    const t = setTimeout(() => {
      client.clusters(run.id, { search, min_size: minSize, offset, limit }).then(setPage).catch(() => setPage(null));
    }, 200);
    return () => clearTimeout(t);
  }, [client, run.id, search, minSize, offset]);

  const cols = ["cluster_label", "cluster_size", "states", "suggested_mapping", "suggestion_score", "suggestion_confidence", "sample_names"];
  return (
    <>
      <div className="row gap wrap">
        <label className="chip">min size
          <input type="number" min={1} value={minSize} style={{ width: 64 }}
            onChange={(e) => { setOffset(0); setMinSize(Number(e.target.value) || 1); }} />
        </label>
        <input className="grow" placeholder="Search cluster label / names / suggestion…" value={search}
          onChange={(e) => { setOffset(0); setSearch(e.target.value); }} />
      </div>
      <DataTable page={page} cols={cols} offset={offset} limit={limit} setOffset={setOffset} />
    </>
  );
}

function DataTable({ page, cols, offset, limit, setOffset }: { page: Page | null; cols: string[]; offset: number; limit: number; setOffset: (n: number) => void }) {
  if (!page) return <p className="muted">Loading…</p>;
  const shown = cols.filter((c) => !page.columns || page.columns.includes(c));
  const fmt = (v: unknown) => (typeof v === "number" && !Number.isInteger(v) ? v.toFixed(3) : v == null ? "" : String(v));
  return (
    <>
      <div className="table-wrap">
        <table className="table">
          <thead><tr>{shown.map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {page.rows.map((r, i) => (
              <tr key={i}>{shown.map((c) => <td key={c} title={fmt(r[c])}>{fmt(r[c])}</td>)}</tr>
            ))}
            {page.rows.length === 0 && <tr><td colSpan={shown.length} className="muted">No rows.</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="row between small">
        <span className="muted">{page.total.toLocaleString()} rows · showing {offset + 1}–{Math.min(offset + limit, page.total)}</span>
        <div className="row gap">
          <button className="btn" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>‹ Prev</button>
          <button className="btn" disabled={offset + limit >= page.total} onClick={() => setOffset(offset + limit)}>Next ›</button>
        </div>
      </div>
    </>
  );
}

// ------------------------------------------------------------------ outputs
function OutputsTab({ run, client }: { run: RunSnapshot; client: EngineClient }) {
  const [files, setFiles] = useState<OutputFile[]>([]);
  useEffect(() => { client.outputs(run.id).then(setFiles).catch(() => setFiles([])); }, [client, run.id]);
  const kb = (n: number) => (n > 1_048_576 ? `${(n / 1_048_576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);
  return (
    <section className="card">
      <div className="row between">
        <h4>{run.output_dir}</h4>
        {native.inTauri && <button className="btn" onClick={() => native.open(run.output_dir)}>Open folder</button>}
      </div>
      <table className="table compact">
        <thead><tr><th>File</th><th>Size</th><th>Modified</th><th></th></tr></thead>
        <tbody>
          {files.map((f) => (
            <tr key={f.name}>
              <td>{f.name}</td>
              <td>{kb(f.size_bytes)}</td>
              <td className="muted small">{f.modified.replace("T", " ")}</td>
              <td className="row gap">
                {native.inTauri ? (
                  <>
                    <button className="link" onClick={() => native.open(f.path)}>Open</button>
                    <button className="link" onClick={() => native.reveal(f.path)}>Reveal</button>
                  </>
                ) : (
                  <a className="link" href={`${client.base}/runs/${run.id}/outputs/${encodeURIComponent(f.name)}`}>Download</a>
                )}
              </td>
            </tr>
          ))}
          {files.length === 0 && <tr><td colSpan={4} className="muted">No files.</td></tr>}
        </tbody>
      </table>
    </section>
  );
}
