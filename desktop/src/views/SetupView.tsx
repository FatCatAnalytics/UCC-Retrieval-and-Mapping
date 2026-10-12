import { useEffect, useState } from "react";
import { EngineClient, native } from "../api";
import type { CsvSource, Health, RunSnapshot, SourcePreview, SourceSpec, SqlSource } from "../types";

interface Props {
  client: EngineClient;
  health: Health | null;
  initialSource: SourceSpec | null;
  onStarted: (run: RunSnapshot, source: SourceSpec) => void;
}

const LS_KEY = "ucc.lastSource";

const emptyCsv: CsvSource = { type: "csv", ucc_path: "", master_path: "", universe_path: "" };
const emptySql: SqlSource = {
  type: "sql",
  url: "",
  ucc_query: "SELECT DEBT_STATE, SECP_STATE, SECP_NAME FROM ucc_filings",
  master_query: 'SELECT DEBT_STATE, SECP_STATE, SECP_NAME, SECP_NAME_MAPPED, "Firm ID" FROM master_mapping',
  universe_query: 'SELECT "Bank Name", "Firm ID" FROM universe',
};

export function SetupView({ client, health, initialSource, onStarted }: Props) {
  const stored = (() => {
    try {
      return JSON.parse(localStorage.getItem(LS_KEY) ?? "null") as SourceSpec | null;
    } catch {
      return null;
    }
  })();
  const start = initialSource ?? stored;
  const [mode, setMode] = useState<"csv" | "sql">(start?.type ?? "csv");
  const [csv, setCsv] = useState<CsvSource>(start?.type === "csv" ? start : emptyCsv);
  const [sql, setSql] = useState<SqlSource>(start?.type === "sql" ? start : emptySql);
  const [sample, setSample] = useState<number>(0);
  const [name, setName] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  const [defaults, setDefaults] = useState<Record<string, unknown>>({});
  const [preview, setPreview] = useState<SourcePreview | null>(null);
  const [busy, setBusy] = useState<"" | "preview" | "start" | "models">("");
  const [error, setError] = useState<string | null>(null);
  const [modelsMsg, setModelsMsg] = useState<string | null>(null);

  const source: SourceSpec = mode === "csv" ? csv : sql;

  useEffect(() => {
    client.configDefaults().then((d) => setDefaults(d.defaults)).catch(() => {});
  }, [client]);

  const pick = async (field: keyof Omit<CsvSource, "type">, title: string) => {
    const p = await native.pickFile(title);
    if (p) setCsv((c) => ({ ...c, [field]: p }));
  };

  const doPreview = async () => {
    setBusy("preview");
    setError(null);
    try {
      setPreview(await client.previewSource(source));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  };

  const doStart = async () => {
    setBusy("start");
    setError(null);
    try {
      const config: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(overrides)) if (v !== "") config[k] = v;
      const run = await client.startRun({ source, config, sample_rows: sample, name });
      localStorage.setItem(LS_KEY, JSON.stringify(source));
      onStarted(run, source);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  };

  const doPrefetch = async () => {
    setBusy("models");
    setModelsMsg(null);
    try {
      const r = await client.prefetchModels();
      setModelsMsg(`Ready · embedding: ${r.embedding} · reranker: ${r.reranker}`);
    } catch (e) {
      setModelsMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  };

  const csvReady = csv.ucc_path && csv.master_path;
  const sqlReady = sql.url && sql.ucc_query && sql.master_query;
  const ready = mode === "csv" ? csvReady : sqlReady;

  const tunables = [
    "fuzzy_auto_threshold", "fuzzy_review_threshold", "embed_auto_threshold", "embed_high_threshold",
    "embed_medium_threshold", "cluster_sim_threshold", "cluster_merge_threshold", "lead_bank_threshold",
    "advanced_cross_encoder_enabled", "embedding_backend", "cpu_threads",
  ];

  return (
    <div className="page">
      <header className="page-head">
        <h1>New mapping run</h1>
        <p className="muted">
          Inputs: UCC filings, the current Master Mapping, and the Universe (tracked banks). Everything runs
          locally; the two small models load from the on-disk cache.
        </p>
      </header>

      <section className="card">
        <div className="seg">
          <button className={mode === "csv" ? "on" : ""} onClick={() => setMode("csv")}>CSV files</button>
          <button className={mode === "sql" ? "on" : ""} onClick={() => setMode("sql")}>SQL database</button>
        </div>

        {mode === "csv" ? (
          <div className="grid2">
            <PathField label="UCC filings (required)" value={csv.ucc_path}
              onChange={(v) => setCsv({ ...csv, ucc_path: v })} onPick={() => pick("ucc_path", "Choose UCC filings CSV")} />
            <PathField label="Master Mapping (required)" value={csv.master_path}
              onChange={(v) => setCsv({ ...csv, master_path: v })} onPick={() => pick("master_path", "Choose Master Mapping CSV")} />
            <PathField label="Universe / tracked banks (optional)" value={csv.universe_path}
              onChange={(v) => setCsv({ ...csv, universe_path: v })} onPick={() => pick("universe_path", "Choose Universe CSV")} />
          </div>
        ) : (
          <div className="grid1">
            <label className="field">
              <span>SQLAlchemy URL</span>
              <input value={sql.url} placeholder="mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+18+for+SQL+Server"
                onChange={(e) => setSql({ ...sql, url: e.target.value })} />
              <small className="muted">Install the matching driver in the engine environment. Passwords are never persisted.</small>
            </label>
            <label className="field"><span>UCC filings query</span>
              <textarea rows={2} value={sql.ucc_query} onChange={(e) => setSql({ ...sql, ucc_query: e.target.value })} /></label>
            <label className="field"><span>Master Mapping query</span>
              <textarea rows={2} value={sql.master_query} onChange={(e) => setSql({ ...sql, master_query: e.target.value })} /></label>
            <label className="field"><span>Universe query (optional)</span>
              <textarea rows={2} value={sql.universe_query} onChange={(e) => setSql({ ...sql, universe_query: e.target.value })} /></label>
          </div>
        )}

        <div className="row gap">
          <button className="btn" disabled={!ready || busy !== ""} onClick={doPreview}>
            {busy === "preview" ? "Checking…" : "Check inputs"}
          </button>
          <button className="btn" disabled={busy !== ""} onClick={doPrefetch} title="Download the embedding and reranker models now">
            {busy === "models" ? "Preparing models…" : "Prepare models"}
          </button>
          {modelsMsg && <span className="small muted">{modelsMsg}</span>}
        </div>

        {preview && <PreviewTable preview={preview} />}
      </section>

      <section className="card">
        <div className="grid2">
          <label className="field">
            <span>Run name</span>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. June refresh" />
          </label>
          <label className="field">
            <span>Quick test: random sample of UCC rows (0 = all)</span>
            <input type="number" min={0} step={1000} value={sample} onChange={(e) => setSample(Number(e.target.value) || 0)} />
          </label>
        </div>

        <button className="link" onClick={() => setAdvanced((a) => !a)}>
          {advanced ? "Hide" : "Show"} advanced thresholds
        </button>
        {advanced && (
          <div className="grid3">
            {tunables.map((k) => (
              <label className="field" key={k}>
                <span>{k}</span>
                <input value={overrides[k] ?? ""} placeholder={String(defaults[k] ?? "")}
                  onChange={(e) => setOverrides({ ...overrides, [k]: e.target.value })} />
              </label>
            ))}
          </div>
        )}
      </section>

      {error && <div className="alert">{error}</div>}

      <div className="row gap end">
        {health?.active_run && <span className="muted small">A run is already in progress.</span>}
        <button className="btn btn-primary" disabled={!ready || busy !== "" || !!health?.active_run} onClick={doStart}>
          {busy === "start" ? "Starting…" : "Start run"}
        </button>
      </div>
    </div>
  );
}

function PathField({ label, value, onChange, onPick }: { label: string; value: string; onChange: (v: string) => void; onPick: () => void }) {
  return (
    <label className="field">
      <span>{label}</span>
      <div className="row">
        <input value={value} onChange={(e) => onChange(e.target.value)} placeholder="/path/to/file.csv" />
        {native.inTauri && <button className="btn" onClick={onPick} type="button">Browse…</button>}
      </div>
    </label>
  );
}

function PreviewTable({ preview }: { preview: SourcePreview }) {
  const rows: Array<["ucc" | "master" | "universe", string]> = [["ucc", "UCC filings"], ["master", "Master Mapping"], ["universe", "Universe"]];
  return (
    <table className="table compact">
      <thead><tr><th>Input</th><th>Status</th><th>Rows</th><th>Columns</th></tr></thead>
      <tbody>
        {rows.map(([k, label]) => {
          const p = preview.inputs[k];
          const ok = p.present && !p.error && (p.missing_columns?.length ?? 0) === 0;
          return (
            <tr key={k}>
              <td>{label}</td>
              <td>
                {!p.present ? <span className="muted">not provided</span>
                  : p.error ? <span className="bad">{p.error}</span>
                  : ok ? <span className="good">OK</span>
                  : <span className="bad">missing: {p.missing_columns?.join(", ")}</span>}
              </td>
              <td>{p.rows?.toLocaleString() ?? "—"}</td>
              <td className="muted small">{p.columns?.join(", ")}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
