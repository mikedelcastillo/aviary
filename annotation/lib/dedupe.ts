// Server-only dedupe orchestration: hash -> cluster -> keep-best metadata.
// Delegates hashing to lib/hash-cache.ts (which uses lib/phash.ts); no sharp here.
import { promises as fs } from "node:fs";
import { sidecarFsPath } from "./paths";
import { CATEGORIES, type CatId, type DedupeCluster, type DedupeMember } from "./types";
import { listImages } from "./annotation-io";
import { type HashInfo } from "./hash-cache";
import { ensureCatHashesForeground } from "./hash-indexer";
import { hamming } from "./phash";

/** True when a non-empty YOLO .txt sidecar exists (real labels; empty = negative). */
async function hasLabels(cat: CatId, name: string): Promise<boolean> {
  try {
    const st = await fs.stat(sidecarFsPath(cat, name, ".txt"));
    return st.size > 0;
  } catch {
    return false;
  }
}

interface MemberMeta {
  name: string;
  hasLabels: boolean;
  sizeBytes: number;
}

/**
 * Keep-best comparator (max wins): prefer a labeled frame, then the larger file
 * (sharpness proxy), then the lexically-latest name on a full tie. Matches the
 * Python prototype's `max(key=keep_rank)` behavior.
 */
function isBetterKeep(a: MemberMeta, b: MemberMeta): boolean {
  if (a.hasLabels !== b.hasLabels) return a.hasLabels && !b.hasLabels;
  if (a.sizeBytes !== b.sizeBytes) return a.sizeBytes > b.sizeBytes;
  return a.name > b.name;
}

interface Cluster {
  anchor: bigint;
  members: { name: string; dist: number }[];
}

/** Cluster one category's images into near-duplicate groups at a Hamming threshold. */
export async function clusterCategory(cat: CatId, threshold: number): Promise<DedupeCluster[]> {
  const names = listImages(cat);
  if (names.length === 0) return [];

  const hashes = await ensureCatHashesForeground(cat);

  // Greedy anchor clustering: attach each image to the existing cluster whose
  // anchor is closest, if within threshold; otherwise start a new cluster.
  const clusters: Cluster[] = [];
  for (const name of names) {
    const info = hashes.get(name);
    if (!info) continue; // unreadable image — skip
    let best: Cluster | null = null;
    let bestDist = Infinity;
    for (const c of clusters) {
      const d = hamming(info.hash, c.anchor);
      if (d < bestDist) {
        best = c;
        bestDist = d;
      }
    }
    if (best && bestDist <= threshold) {
      best.members.push({ name, dist: bestDist });
    } else {
      clusters.push({ anchor: info.hash, members: [{ name, dist: 0 }] });
    }
  }

  // Decorate with metadata + pick the keeper per cluster.
  const out: DedupeCluster[] = [];
  for (const c of clusters) {
    const members: DedupeMember[] = await Promise.all(
      c.members.map(async (m) => {
        const info = hashes.get(m.name) as HashInfo;
        return {
          name: m.name,
          hasLabels: await hasLabels(cat, m.name),
          sizeBytes: info?.size ?? 0,
          dist: m.dist,
        };
      }),
    );
    let keep = members[0];
    for (const m of members) if (isBetterKeep(m, keep)) keep = m;
    out.push({ cat, keepName: keep.name, members });
  }
  return out;
}

/**
 * Near-duplicate clusters across the requested categories. By default only
 * actionable groups (more than one member) are returned; singletons are noise
 * for the review UI.
 */
export async function getDedupeClusters(
  cats: CatId[],
  threshold: number,
  opts: { includeSingletons?: boolean } = {},
): Promise<DedupeCluster[]> {
  const order = CATEGORIES.map((c) => c.id).filter((id) => cats.includes(id));
  const out: DedupeCluster[] = [];
  for (const cat of order) {
    const clusters = await clusterCategory(cat, threshold);
    for (const cl of clusters) {
      if (opts.includeSingletons || cl.members.length > 1) out.push(cl);
    }
  }
  return out;
}
