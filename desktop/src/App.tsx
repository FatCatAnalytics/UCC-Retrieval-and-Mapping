import { useCallback, useEffect, useMemo, useState } from "react";
import { EngineClient, resolveEngineUrl } from "./api";
import type { Health, RunSnapshot, SourceSpec } from "./types";
import { SetupView } from "./views/SetupView";
import { RunView } from "./views/RunView";
import { ResultsView } from "./views/ResultsView";

type View = "setup" | "run" | "results";

export default function App() {
  const [client, setClient] = useState<EngineClient | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunSnapshot[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [view, setView] = useState<View>("setup");
  const [lastSource, setLastSource] = useState<SourceSpec | null>(null);

  // Boot: find the engine, then poll health + run list.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const url = await resolveEngineUrl();
        const c = new EngineClient(url);
        // Wait for the HTTP server to accept connections.
        for (let i = 0; i < 100; i++) {
          try {
            const h = await c.health();
            if (cancelled) return;
            setHealth(h);
            break;
          } catch {
            await new Promise((r) => setTimeout(r, 300));
          }
        }
        if (!cancelled) setClient(c);
      } catch (e) {
        if (!cancelled) setBootError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshRuns = useCallback(async () => {
    if (!client) return;
    try {
      const list = await client.listRuns();
      setRuns(list);
      if (!selected && list[0]) setSelected(list[0].id);
    } catch {
      /* transient */
    }
  }, [client, selected]);

  useEffect(() => {
    if (!client) return;
    refreshRuns();
    const t = setInterval(refreshRuns, 2000);
    return () => clearInterval(t);
  }, [client, refreshRuns]);

  const selectedRun = useMemo(() => runs.find((r) => r.id === selected) ?? null, [runs, selected]);

  const onStarted = (run: RunSnapshot) => {
    setRuns((prev) => [run, ...prev]);
    setSelected(run.id);
    setView("run");
  };

  if (bootError) {
    return (
      <div className="boot boot-error">
        <h1>Mapping engine unavailable</h1>
        <pre>{bootError}</pre>
        <p>
          In development, start the engine yourself (<code>ucc-engine serve --port 8765</code>) and launch the app
          with <code>UCC_ENGINE_URL=http://127.0.0.1:8765</code>, or point <code>UCC_ENGINE_CMD</code> at the
          <code> ucc-engine</code> executable.
        </p>
      </div>
    );
  }
  if (!client) {
    return (
      <div className="boot">
        <div className="spinner" />
        <p>Starting the mapping engine…</p>
      </div>
    );
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">UCC</span>
          <span className="brand-text">Mapping</span>
        </div>
        <button className="btn btn-primary wide" onClick={() => setView("setup")}>
          + New run
        </button>
        <div className="runs-list">
          {runs.length === 0 && <p className="muted small">No runs yet.</p>}
          {runs.map((r) => (
            <button
              key={r.id}
              className={`run-item ${r.id === selected ? "active" : ""}`}
              onClick={() => {
                setSelected(r.id);
                setView(r.status === "succeeded" ? "results" : "run");
              }}
            >
              <span className={`dot ${r.status}`} />
              <span className="run-name">{r.name}</span>
              <span className="muted small">
                {r.status === "running" ? `${Math.round(r.progress * 100)}%` : r.status}
                {r.sample_rows ? ` · sample ${r.sample_rows.toLocaleString()}` : ""}
              </span>
            </button>
          ))}
        </div>
        <footer className="sidebar-footer muted small">
          engine v{health?.version} · {health?.backends.fastembed ? "fastembed" : "sentence-transformers"} · local
          CPU
        </footer>
      </aside>

      <main className="content">
        {view === "setup" && (
          <SetupView client={client} health={health} initialSource={lastSource} onStarted={(run, src) => {
            setLastSource(src);
            onStarted(run);
          }} />
        )}
        {view === "run" && selectedRun && (
          <RunView
            client={client}
            run={selectedRun}
            onUpdate={(r) => setRuns((prev) => prev.map((x) => (x.id === r.id ? r : x)))}
            onViewResults={() => setView("results")}
          />
        )}
        {view === "results" && selectedRun && <ResultsView client={client} run={selectedRun} />}
        {(view === "run" || view === "results") && !selectedRun && (
          <div className="empty">Select a run from the left, or start a new one.</div>
        )}
      </main>
    </div>
  );
}
