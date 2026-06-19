// FS-as-DB integrity checker ("doctor"/fsck). Scans the annotation data tree and
// reports drift you can't see at a glance: corrupt JSON sidecars, orphaned
// sidecars (no matching image), un-annotated images, empty .txt exports, and
// leftover .tmp-<pid> crash files.
//
// Usage:  node scripts/doctor-fsdb.mjs [dataRoot]
//   dataRoot defaults to $AVIARY_DATA_ROOT, else <repo>/data/annotation/raw
import { readdirSync, readFileSync, statSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const DATA_ROOT =
  process.argv[2] || process.env.AVIARY_DATA_ROOT || join(repoRoot, "data/annotation/raw");

// Category id -> directory under the data root. KEEP IN SYNC with CATEGORIES in
// annotation/lib/types.ts (this script is intentionally dependency-free).
const CATEGORY_DIRS = { day: "tapo/day", ir: "tapo/ir", phone: "phone" };

const IMG_RE = /\.(jpe?g|png)$/i;
const totals = { images: 0, json: 0, txt: 0, suggest: 0 };
const issues = { corrupt: [], orphanSidecar: [], noAnnotation: [], emptyTxt: [], tmp: [] };

function stemOf(file) {
  if (file.endsWith(".suggest.json")) return file.slice(0, -".suggest.json".length);
  if (file.endsWith(".json")) return file.slice(0, -".json".length);
  if (file.endsWith(".txt")) return file.slice(0, -".txt".length);
  const m = file.match(IMG_RE);
  return m ? file.slice(0, -m[0].length) : null;
}

for (const [cat, rel] of Object.entries(CATEGORY_DIRS)) {
  const dir = join(DATA_ROOT, rel);
  if (!existsSync(dir)) {
    console.log(`! category "${cat}" dir missing: ${dir}`);
    continue;
  }
  const files = readdirSync(dir);
  const imageStems = new Set();
  for (const f of files) if (IMG_RE.test(f)) imageStems.add(stemOf(f));

  for (const f of files) {
    const full = join(dir, f);
    if (f.includes(".tmp-")) {
      issues.tmp.push(`${cat}/${f}`);
      continue;
    }
    if (IMG_RE.test(f)) {
      totals.images++;
      const stem = stemOf(f);
      if (!existsSync(join(dir, `${stem}.json`))) issues.noAnnotation.push(`${cat}/${f}`);
      continue;
    }
    const stem = stemOf(f);
    if (f.endsWith(".suggest.json")) {
      totals.suggest++;
      if (!imageStems.has(stem)) issues.orphanSidecar.push(`${cat}/${f}`);
    } else if (f.endsWith(".json")) {
      totals.json++;
      if (!imageStems.has(stem)) issues.orphanSidecar.push(`${cat}/${f}`);
      try {
        JSON.parse(readFileSync(full, "utf8"));
      } catch (e) {
        issues.corrupt.push(`${cat}/${f}: ${e.message}`);
      }
    } else if (f.endsWith(".txt")) {
      totals.txt++;
      if (!imageStems.has(stem)) issues.orphanSidecar.push(`${cat}/${f}`);
      else if (statSync(full).size === 0) issues.emptyTxt.push(`${cat}/${f}`);
    }
  }
}

function report(title, list, { sample = 10 } = {}) {
  const n = list.length;
  const flag = n > 0 ? "✗" : "✓";
  console.log(`${flag} ${title}: ${n}`);
  for (const item of list.slice(0, sample)) console.log(`    - ${item}`);
  if (n > sample) console.log(`    … and ${n - sample} more`);
}

console.log(`\nFS-as-DB doctor — ${DATA_ROOT}\n`);
console.log(
  `totals: ${totals.images} images · ${totals.json} .json · ${totals.txt} .txt · ${totals.suggest} .suggest.json\n`,
);
report("corrupt .json (parse failures)", issues.corrupt);
report("orphaned sidecars (no matching image)", issues.orphanSidecar);
report("leftover .tmp-<pid> crash files", issues.tmp);
report("empty .txt (reviewed-negative — usually fine)", issues.emptyTxt, { sample: 0 });
report("images with no .json (un-annotated — informational)", issues.noAnnotation, { sample: 0 });

const hardFails = issues.corrupt.length + issues.orphanSidecar.length + issues.tmp.length;
console.log(`\n${hardFails === 0 ? "OK — no integrity problems." : `${hardFails} issue(s) need attention.`}`);
process.exit(hardFails === 0 ? 0 : 1);
