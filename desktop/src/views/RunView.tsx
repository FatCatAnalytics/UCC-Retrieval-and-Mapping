import { useEffect, useRef, useState } from "react";
import { EngineClient } from "../api";
import type { LogLine, RunSnapshot } from "../types";

interface Props {
  client: EngineClient;
  run: RunSnapshot;
  onUpdate: (run: RunSnapshot) => void;
  onViewResults: () => void;
}

export function RunView({ client, run, onUpdate, onViewResults }: Props) {
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [showDebug, setShowDebug] = useState(false);
  const lastSeq = useRef(0);
  const logBox = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);

  // Reset when switching runs.
  useEffect(() => {
    setLogs([]);
    lastSeq.current = 0;
  }, [run.id]);

  // Poll status + logs while active (and once after finishing to flush).
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const [snap, lg] = await Promise.all([client.getRun(run.id), client.logs(run.id, lastSeq.current)]);
        if (!alive) return;
        onUpdate(snap);
        if (lg.lines.length) {
          lastSeq.current = lg.last_seq;
          setLogs((prev) => [...prev, ...lg.lines].slice(-3000));
        }
      } catch {
        /* transient */
      }
    };
    tick();
    const active = run.status === "running" || run.status === "queued";
    const t = setInterval(tick, active ? 1000 : 5000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [client, run.id, run.status, onUpdate]);

  useEffect(() => {
    if (follow && logBox.current) logBox.current.scrollTop = logBox.current.scrollHeight;
  }, [logs, follow]);

  const pct = Math.round(run.progress * 100);
  const visible = showDebug ? logs : logs.filter((l) => l.level !== "DEBUG");
  const groups = Array.from(new Set(run.stages.map((s) => s.group)));

  return (
    <div className="page">
      <header className="page-head row between">
        <div>
          <h1>{run.name}</h1>
          <p className="muted">
            {run.input_rows.ucc ? `${run.input_rows.ucc.toLocaleString()} UCC rows · ` : ""}
            {run.input_rows.master ? `${run.input_rows.master.toLocaleString()} master rows · ` : ""}
            {run.input_rows.universe ? `${run.input_rows.universe.toLocaleString()} universe rows` : "no universe"}
          </p>
        </div>
        <div className="row gap">
          {(run.status === "running" || run.status === "queued") && (
            <button className="btn btn-danger" onClick={() => client.cancelRun(run.id).then(onUpdate)}>
              Cancel after current stage
            </button>
          )}
          {run.status === "succeeded" && (
            <button className="btn btn-primary" onClick={onViewResults}>View results →</button>
          )}
        </div>
      </header>

      <section className="card">
        <div className="row between">
          <strong className={`status ${run.status}`}>{run.status.toUpperCase()}</strong>
          <span className="muted">{pct}%</span>
        </div>
        <div className="progress"><div style={{ width: `${pct}%` }} /></div>
        {run.error && <div className="alert">{run.error}</div>}
      </section>

      <div className="split">
        <section className="card stages">
          {groups.map((g) => (
            <div key={g}>
              <h4 className="group">{g}</h4>
              {run.stages.filter((s) => s.group === g).map((s) => (
                <div key={s.name} className={`stage ${s.status}`}>
                  <span className={`dot ${s.status}`} />
                  <span className="stage-label">{s.label}</span>
                  <span className="muted small">
                    {s.status === "done" && s.seconds != null ? `${s.seconds.toFixed(1)}s` : ""}
                    {s.rows != null ? ` · ${s.rows.toLocaleString()} rows` : ""}
                    {s.status === "running" ? "running…" : ""}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </section>

        <section className="card logs">
          <div className="row between">
            <h4>Log</h4>
            <div className="row gap small">
              <label><input type="checkbox" checked={showDebug} onChange={(e) => setShowDebug(e.target.checked)} /> debug</label>
              <label><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow</label>
            </div>
          </div>
          <div className="logbox" ref={logBox} onScroll={(e) => {
            const el = e.currentTarget;
            setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 40);
          }}>
            {visible.map((l) => (
              <div key={l.seq} className={`log ${l.level.toLowerCase()}`}>
                <span className="muted">{l.ts.slice(11, 19)}</span> <span className="tag">{l.stage}</span> {l.message}
              </div>
            ))}
            {visible.length === 0 && <p className="muted small">Waiting for output…</p>}
          </div>
        </section>
      </div>
    </div>
  );
}
