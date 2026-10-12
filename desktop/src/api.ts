import { invoke } from "@tauri-apps/api/core";
import type {
  ApplyResult, DecisionCounts, DecisionItem, DecisionsPage, Health, LogLine, OutputFile, Page, RunSnapshot,
  SourcePreview, SourceSpec, StoreCounts, Summary,
} from "./types";

const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

interface EngineInfo {
  url: string | null;
  error: string | null;
  source: string | null;
}

/** Resolve the engine base URL: Tauri sidecar port, or VITE_ENGINE_URL when running plain `vite`. */
export async function resolveEngineUrl(timeoutMs = 60_000): Promise<string> {
  const fallback = import.meta.env.VITE_ENGINE_URL as string | undefined;
  if (!inTauri) {
    if (fallback) return fallback.replace(/\/$/, "");
    throw new Error("Not running inside Tauri and VITE_ENGINE_URL is not set.");
  }
  const started = Date.now();
  let lastError: string | null = null;
  while (Date.now() - started < timeoutMs) {
    const info = await invoke<EngineInfo>("engine_url");
    if (info.url) return info.url;
    if (info.error) lastError = info.error;
    if (lastError && !lastError.startsWith("Engine exited")) break;
    await new Promise((r) => setTimeout(r, 400));
  }
  throw new Error(lastError ?? "Timed out waiting for the mapping engine to start.");
}

export class EngineClient {
  constructor(public readonly base: string) {}

  private async req<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(this.base + path, {
      ...init,
      headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
      } catch {
        /* keep statusText */
      }
      throw new Error(`${res.status} ${detail}`);
    }
    return (await res.json()) as T;
  }

  health = () => this.req<Health>("/health");
  configDefaults = () => this.req<{ defaults: Record<string, unknown>; schema: unknown[] }>("/config/defaults");
  prefetchModels = (overrides: Record<string, unknown> = {}) =>
    this.req<Record<string, string>>("/models/prefetch", { method: "POST", body: JSON.stringify(overrides) });

  previewSource = (source: SourceSpec) =>
    this.req<SourcePreview>("/sources/preview", { method: "POST", body: JSON.stringify({ source }) });

  listRuns = () => this.req<RunSnapshot[]>("/runs");
  getRun = (id: string) => this.req<RunSnapshot>(`/runs/${id}`);
  deleteRun = (id: string) => this.req<{ deleted: boolean }>(`/runs/${id}`, { method: "DELETE" });
  cancelRun = (id: string) => this.req<RunSnapshot>(`/runs/${id}/cancel`, { method: "POST" });
  startRun = (body: { source: SourceSpec; config: Record<string, unknown>; sample_rows: number; name: string }) =>
    this.req<RunSnapshot>("/runs", { method: "POST", body: JSON.stringify(body) });

  logs = (id: string, after = 0) =>
    this.req<{ lines: LogLine[]; last_seq: number; status: string }>(`/runs/${id}/logs?after=${after}&limit=1000`);
  summary = (id: string) => this.req<Summary>(`/runs/${id}/summary`);
  outputs = (id: string) => this.req<OutputFile[]>(`/runs/${id}/outputs`);

  review = (id: string, q: { confidence?: string[]; search?: string; method?: string; decision?: string; offset?: number; limit?: number }) => {
    const p = new URLSearchParams();
    (q.confidence ?? []).forEach((c) => p.append("confidence", c));
    if (q.search) p.set("search", q.search);
    if (q.method) p.set("method", q.method);
    if (q.decision) p.set("decision", q.decision);
    p.set("offset", String(q.offset ?? 0));
    p.set("limit", String(q.limit ?? 100));
    return this.req<Page>(`/runs/${id}/review?${p}`);
  };

  clusters = (id: string, q: { search?: string; min_size?: number; offset?: number; limit?: number }) => {
    const p = new URLSearchParams();
    if (q.search) p.set("search", q.search);
    p.set("min_size", String(q.min_size ?? 1));
    p.set("offset", String(q.offset ?? 0));
    p.set("limit", String(q.limit ?? 100));
    return this.req<Page>(`/runs/${id}/clusters?${p}`);
  };

  decisions = (id: string) => this.req<{ counts: DecisionCounts }>(`/runs/${id}/decisions`);
  setDecisions = (id: string, items: DecisionItem[]) =>
    this.req<{ changed: number; counts: DecisionCounts }>(`/runs/${id}/decisions`, { method: "PUT", body: JSON.stringify({ items }) });
  applyDecisions = (id: string) => this.req<ApplyResult>(`/runs/${id}/apply-decisions`, { method: "POST" });

  // The decisions database (every run)
  allDecisions = (q: { decision?: string; search?: string; run_id?: string; offset?: number; limit?: number }) => {
    const p = new URLSearchParams();
    if (q.decision) p.set("decision", q.decision);
    if (q.search) p.set("search", q.search);
    if (q.run_id) p.set("run_id", q.run_id);
    p.set("offset", String(q.offset ?? 0));
    p.set("limit", String(q.limit ?? 100));
    return this.req<DecisionsPage>(`/decisions?${p}`);
  };
  clearDecisions = (keys: string[]) =>
    this.req<{ cleared: number; counts: StoreCounts }>("/decisions/clear", { method: "POST", body: JSON.stringify({ keys }) });
  decisionsExportUrl = () => `${this.base}/decisions/export`;
  exportDecisions = (destination: string) =>
    this.req<{ path: string; rows: number }>("/decisions/export", { method: "POST", body: JSON.stringify({ destination }) });

  promoteMaster = (id: string, destination: string) =>
    this.req<{ source: string; destination: string; backup: string | null }>(`/runs/${id}/promote-master`, {
      method: "POST",
      body: JSON.stringify({ destination }),
    });
}

// ---- native helpers (Rust commands); no-ops outside Tauri --------------------
export const native = {
  pickFile: (title: string) => (inTauri ? invoke<string | null>("pick_file", { title }) : Promise.resolve(null)),
  pickSaveFile: (title: string, defaultName: string) =>
    inTauri ? invoke<string | null>("pick_save_file", { title, defaultName }) : Promise.resolve(null),
  reveal: (path: string) => (inTauri ? invoke<void>("reveal_path", { path }) : Promise.resolve()),
  open: (path: string) => (inTauri ? invoke<void>("open_path", { path }) : Promise.resolve()),
  inTauri,
};
