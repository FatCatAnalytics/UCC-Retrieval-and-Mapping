import { useEffect, useState } from "react";
import { EngineClient, native } from "../api";
import type { OutputFile, Page, RunSnapshot, Summary } from "../types";

interface Props {
  client: EngineClient;
  run: RunSnapshot;
}

type Tab = "summary" | "review" | "clusters" | "outputs";
const BANDS = ["REVIEW_HIGH", "REVIEW_MEDIUM", "REVIEW_LOW"];

export function ResultsView({ client, run }: Props) {
  const [tab, setTab] = useState<Tab>("summary");
  const [summary, setSummary] = useState<Summary | null>(run.summary);

  useEffect(() => {
    setSummary(run.summary);
    if (!run.summary && run.status === "succeeded") client.summary(run.id).then(setSummary).catch(() => {});
  }, [client, run.id, run.summary, run.status]);

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
            {t === "review" && `Review queue${summary ? ` (${summary.review_needed.toLocaleString()})` : ""}`}
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
  if (!summary) return <p className="muted">Summary not available.</p>;

  const pct = (n: number) => (summary.total_records ? `${((n / summary.total_records) * 100).toFixed(1)}%` : "");
  const promote = async () => {
    if (!dest) return;
    if (!confirm(`Overwrite\n${dest}\nwith this run's updated master? A timestamped backup is kept.`)) return;
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
        <Stat label="Review needed" value={summary.review_needed} sub={pct(summary.review_needed)} tone="warn" />
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
            <strong>{summary.updated_master.added_rows.toLocaleString()} newly resolved</strong> (Stages 1–3; the original
            master always wins on conflicts).
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
function ReviewTab({ run, client }: { run: RunSnapshot; client: EngineClient }) {
  const [bands, setBands] = useState<string[]>(BANDS);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState<Page | null>(null);
  const [offset, setOffset] = useState(0);
  const limit = 100;

  useEffect(() => {
    const t = setTimeout(() => {
      client.review(run.id, { confidence: bands, search, offset, limit }).then(setPage).catch(() => setPage(null));
    }, 200);
    return () => clearTimeout(t);
  }, [client, run.id, bands, search, offset]);

  const cols = ["SECP_NAME", "DEBT_STATE", "SECP_STATE", "SECP_NAME_MAPPED", "match_method", "match_score", "confidence", "Firm_ID", "ambiguous_master_candidates"];
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
        <input className="grow" placeholder="Search secured party / mapped name…" value={search}
          onChange={(e) => { setOffset(0); setSearch(e.target.value); }} />
      </div>
      <DataTable page={page} cols={cols} offset={offset} limit={limit} setOffset={setOffset} />
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
