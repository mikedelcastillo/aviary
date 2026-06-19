// Shared contract for the model benchmark explorer. Pure types only — NO node/fs
// imports — so this is safe to import from the client component. Mirrors the
// camelCase shape written by training/scripts/benchmark.py into
// data/models/benchmark.json. Metric values are fractions in 0..1 (or null when
// the denominator is 0); the UI renders them as percentages.

export type BenchMetric = "recall" | "precision" | "f1";

/** The three metrics in the order the UI's toggle presents them. */
export const BENCH_METRICS: BenchMetric[] = ["recall", "f1", "precision"];

export interface BenchScore {
  recall: number | null;
  precision: number | null;
  f1: number | null;
  tp: number;
  fp: number;
  fn: number;
  /** Ground-truth boxes for this scope (== tp + fn). */
  gt: number;
}

/** Per-category overall score, plus how many labeled images were scored. */
export interface BenchCategoryScore extends BenchScore {
  images: number;
}

export interface BenchModel {
  /** Model stem, e.g. "live-001". */
  name: string;
  /** Numeric version (the NNN), the x-axis position within its series. */
  version: number;
  /** Weights filename, e.g. "live-001.pt". */
  file: string;
  /** Labeled images scored across all the series' categories. */
  images: number;
  /** Total in-vocabulary GT boxes scored. */
  gtBoxes: number;
  /** GT boxes whose label is outside this model's vocabulary (excluded). */
  naBoxes: number;
  overall: BenchScore;
  /** Keyed by category id (e.g. "day", "ir"). */
  byCategory: Record<string, BenchCategoryScore>;
  /** Keyed by roster label name (e.g. "percy"). */
  labels: Record<string, BenchScore>;
}

export interface BenchSeries {
  /** "live" | "archive". */
  name: string;
  /** Categories this series was scored on (e.g. ["day","ir"] or ["phone"]). */
  categories: string[];
  /** Models, ascending by version (the trend's x-axis). */
  models: BenchModel[];
}

export interface BenchmarkFile {
  schemaVersion: number;
  /** ISO-8601 UTC timestamp of the run. */
  generatedAt: string;
  config: { confidence: number; iou: number; nmsIou: number; imgsz: number };
  series: BenchSeries[];
}
