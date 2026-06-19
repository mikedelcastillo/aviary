// Server-only data layer for the `<image>.suggest.json` sidecar — model output
// the human approves/rejects. Non-authoritative: it is written by the
// scripts/suggest_* Python jobs and only SHRINKS here (a box/label is dropped
// once accepted or rejected). The canonical `.json`/`.txt` are untouched.
import { promises as fs } from "node:fs";
import type { Suggestions } from "./types";
import { sidecarFsPath } from "./paths";
import { withFileLock } from "./file-lock";

const SUGGEST_EXT = ".suggest.json";

function suggestPath(cat: string, name: string): string {
  return sidecarFsPath(cat, name, SUGGEST_EXT);
}

/** Coerce arbitrary parsed JSON to the normalized Suggestions shape. */
function normalize(data: Partial<Suggestions> | null | undefined): Suggestions {
  const boxes = Array.isArray(data?.boxes) ? data!.boxes : [];
  const labels = Array.isArray(data?.labels) ? data!.labels : [];
  const out: Suggestions = { boxes, labels };
  if (typeof data?.boxModel === "string") out.boxModel = data.boxModel;
  if (typeof data?.labelModel === "string") out.labelModel = data.labelModel;
  return out;
}

/**
 * Read an image's suggestion sidecar. Missing/garbled file -> empty suggestions
 * (this feature is purely additive; absence just means "no suggestions yet").
 * NEVER writes on read.
 */
export async function readSuggestions(cat: string, name: string): Promise<Suggestions> {
  try {
    const raw = await fs.readFile(suggestPath(cat, name), "utf8");
    return normalize(JSON.parse(raw) as Partial<Suggestions>);
  } catch {
    return { boxes: [], labels: [] };
  }
}

/** Atomic write (temp + rename) so a concurrent reader never sees a partial file. */
async function persist(path: string, data: Suggestions): Promise<void> {
  const tmp = `${path}.tmp-${process.pid}`;
  await fs.writeFile(tmp, JSON.stringify(data, null, 2), "utf8");
  await fs.rename(tmp, path);
}

/**
 * Read-modify-write the sidecar under a per-file lock so concurrent accept/reject
 * actions on the same image can't lose each other's edits. Returns the new state.
 */
async function mutate(cat: string, name: string, fn: (s: Suggestions) => Suggestions): Promise<Suggestions> {
  const path = suggestPath(cat, name);
  return withFileLock(path, async () => {
    const next = fn(await readSuggestions(cat, name));
    await persist(path, next);
    return next;
  });
}

/** Drop one box proposal (accepted into the .json, or rejected). */
export function removeSuggestedBox(cat: string, name: string, id: string): Promise<Suggestions> {
  return mutate(cat, name, (s) => ({ ...s, boxes: s.boxes.filter((b) => b.id !== id) }));
}

/** Drop one label suggestion for a human box (its label was accepted or overridden). */
export function removeLabelSuggestion(cat: string, name: string, boxId: string): Promise<Suggestions> {
  return mutate(cat, name, (s) => ({ ...s, labels: s.labels.filter((l) => l.boxId !== boxId) }));
}
