// Server-only reader for the benchmark results file (data/models/benchmark.json),
// written by scripts/benchmark.*. Never import from client code (touches fs).
import { promises as fs, statSync } from "node:fs";
import { BENCHMARK_PATH } from "./paths";
import type { BenchmarkFile } from "./benchmark-types";

// mtime-validated cache: re-parse only when the file changes (a fresh benchmark
// run rewrites it). A warm request with no change does zero filesystem work
// beyond the stat.
let cache: { mtimeMs: number; value: BenchmarkFile | null } | null = null;

/** True when the file has at least one series carrying at least one model. */
function hasModels(file: BenchmarkFile): boolean {
  return Array.isArray(file.series) && file.series.some((s) => Array.isArray(s.models) && s.models.length > 0);
}

/**
 * Read + parse the benchmark file. Returns null when it's missing, unreadable,
 * corrupt, or carries no models — the homepage treats all of those as the empty
 * state (prompt to run the benchmark).
 */
export async function readBenchmark(): Promise<BenchmarkFile | null> {
  let mtimeMs: number;
  try {
    mtimeMs = statSync(BENCHMARK_PATH).mtimeMs;
  } catch {
    cache = null; // file absent
    return null;
  }
  if (cache && cache.mtimeMs === mtimeMs) return cache.value;

  let value: BenchmarkFile | null = null;
  try {
    const parsed = JSON.parse(await fs.readFile(BENCHMARK_PATH, "utf8")) as BenchmarkFile;
    value = parsed && hasModels(parsed) ? parsed : null;
  } catch {
    value = null; // corrupt / mid-write torn read -> empty state
  }
  cache = { mtimeMs, value };
  return value;
}
