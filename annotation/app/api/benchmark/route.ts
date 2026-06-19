import { readBenchmark } from "@/lib/benchmark";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * GET /api/benchmark -> { benchmark: BenchmarkFile | null }.
 * `null` means no results yet (file missing/empty/corrupt) — the homepage shows
 * the empty state prompting `scripts/benchmark.*`.
 */
export async function GET(): Promise<Response> {
  return Response.json({ benchmark: await readBenchmark() });
}
