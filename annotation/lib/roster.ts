// Server-only roster loader. Parses models/roster.yaml, assigns global class
// indices (= file order), and derives the per-category pill set + shortcuts.
import { readFileSync } from "node:fs";
import { parse as parseYaml } from "yaml";
import { ROSTER_PATH } from "./paths";
import type { Pill, PillRule } from "./types";

export interface RosterEntry {
  name: string;
  kind: string; // individual | species | unknown
  models: string[];
  /** Global class index = position in the YAML file. */
  globalIndex: number;
}

// Keyboard order: number row first, then QWERTY rows. Pills render their key.
const SHORTCUTS = "1234567890qwertyuiopasdfghjklzxcvbnm".split("");

let cache: { mtimeKey: string; entries: RosterEntry[] } | null = null;

export function loadRoster(): RosterEntry[] {
  // Re-read on file change (dev convenience); cheap file, tiny parse.
  let raw: string;
  try {
    raw = readFileSync(ROSTER_PATH, "utf8");
  } catch (e) {
    throw new Error(`Cannot read roster at ${ROSTER_PATH}: ${(e as Error).message}`);
  }
  if (cache && cache.mtimeKey === raw) return cache.entries;

  const data = parseYaml(raw) as { labels?: Array<Record<string, unknown>> };
  const entries: RosterEntry[] = (data.labels ?? []).map((entry, index) => ({
    name: String(entry.name),
    kind: String(entry.kind ?? "individual"),
    models: Array.isArray(entry.models) ? (entry.models as unknown[]).map(String) : [],
    globalIndex: index,
  }));
  cache = { mtimeKey: raw, entries };
  return entries;
}

/** Map of label name -> global class index, for writing YOLO .txt. */
export function nameToIndex(): Map<string, number> {
  const m = new Map<string, number>();
  for (const e of loadRoster()) m.set(e.name, e.globalIndex);
  return m;
}

const UNKNOWN = "unknown_bird";

/**
 * Pills for a category, in roster file order, with unknown_bird always last.
 *  - day:   kind=individual AND models includes "live"   (the living birds)
 *  - ir:    kind=species                                 (cockatiel/lovebird/budgie)
 *  - phone: kind=individual AND models includes "archive"(every individual)
 */
export function pillsForCategory(rule: PillRule): Pill[] {
  const roster = loadRoster();
  let chosen: RosterEntry[];
  if (rule === "day") {
    chosen = roster.filter((e) => e.kind === "individual" && e.models.includes("live"));
  } else if (rule === "ir") {
    chosen = roster.filter((e) => e.kind === "species");
  } else {
    chosen = roster.filter((e) => e.kind === "individual" && e.models.includes("archive"));
  }

  const unknown = roster.find((e) => e.name === UNKNOWN);
  const ordered = [...chosen];
  if (unknown && !ordered.some((e) => e.name === UNKNOWN)) ordered.push(unknown);

  return ordered.map((e, i) => ({
    label: e.name,
    globalIndex: e.globalIndex,
    shortcut: SHORTCUTS[i] ?? "",
    kind: e.kind,
  }));
}
