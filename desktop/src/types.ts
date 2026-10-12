export type CsvSource = { type: "csv"; ucc_path: string; master_path: string; universe_path: string };
export type SqlSource = { type: "sql"; url: string; ucc_query: string; master_query: string; universe_query: string };
export type SourceSpec = CsvSource | SqlSource;

export type StageStatus = "pending" | "running" | "done" | "failed" | "skipped";
export interface StageState {
  name: string;
  label: string;
  group: string;
  status: StageStatus;
  started_at: string | null;
  finished_at: string | null;
  seconds: number | null;
  rows: number | null;
}

export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "unknown";

export interface Summary {
  total_records: number;
  auto_mapped: number;
  review_needed: number;
  new_entities: number;
  no_master_state_pair: number;
  by_match_method: Record<string, number>;
  by_confidence: Record<string, number>;
  clusters: number;
  avg_cluster_size: number;
  records_with_firm_id: number;
  unique_firm_ids: number;
  updated_master?: { rows: number; original_rows: number; added_rows: number; file: string | null };
  lead_bank?: { rows: number; columns: string[] };
  stage_seconds: Record<string, number>;
  total_seconds: number;
}

export interface RunSnapshot {
  id: string;
  name: string;
  status: RunStatus;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  progress: number;
  current_stage: string | null;
  sample_rows: number;
  input_rows: Record<string, number>;
  source: SourceSpec;
  run_dir: string;
  output_dir: string;
  stages: StageState[];
  summary: Summary | null;
  config?: Record<string, unknown>;
}

export interface LogLine {
  seq: number;
  ts: string;
  level: string;
  stage: string;
  message: string;
}

export interface Page {
  total: number;
  offset: number;
  limit: number;
  columns?: string[];
  rows: Record<string, unknown>[];
}

export interface OutputFile {
  name: string;
  path: string;
  size_bytes: number;
  modified: string;
}

export interface InputPreview {
  present: boolean;
  path?: string;
  error?: string;
  size_bytes?: number;
  rows?: number;
  columns?: string[];
  missing_columns?: string[];
  preview?: Record<string, string>[];
}

export interface SourcePreview {
  type: "csv" | "sql";
  url?: string;
  inputs: Record<"ucc" | "master" | "universe", InputPreview>;
}

export interface Health {
  status: string;
  version: string;
  pid: number;
  data_dir: string;
  backends: Record<string, boolean>;
  active_run: string | null;
}
