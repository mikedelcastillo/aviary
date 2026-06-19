// Global ceiling on concurrent filesystem reads/stats.
//
// The home page fans out several endpoints at once (progress, queue, box-queue,
// entry, label-stats), and each scans the WHOLE manifest with an unbounded
// `Promise.all(manifest.map(...))` — thousands of sidecar reads apiece. Add the
// dedupe hashing/decoration on top and the process tries to open tens of
// thousands of file handles simultaneously; the OS rejects them with EMFILE
// ("too many open files"). Funnel the hot reads through this gate so at most MAX
// run concurrently and the rest queue. Small JSON reads are fast, so the cap
// costs little throughput while making EMFILE impossible.
import { promises as fs } from "node:fs";
import type { Stats } from "node:fs";

const MAX = 64;
let active = 0;
const waiters: (() => void)[] = [];

function acquire(): Promise<void> {
  if (active < MAX) {
    active++;
    return Promise.resolve();
  }
  // At capacity — park until a slot is handed over by release().
  return new Promise<void>((resolve) => waiters.push(resolve));
}

function release(): void {
  const next = waiters.shift();
  if (next) {
    next(); // transfer the slot directly to the next waiter (active stays put)
  } else {
    active--; // no one waiting — free the slot
  }
}

/** `fs.readFile(path, "utf8")` under the global concurrency cap. */
export async function readTextLimited(p: string): Promise<string> {
  await acquire();
  try {
    return await fs.readFile(p, "utf8");
  } finally {
    release();
  }
}

/** `fs.stat(path)` under the global concurrency cap. */
export async function statLimited(p: string): Promise<Stats> {
  await acquire();
  try {
    return await fs.stat(p);
  } finally {
    release();
  }
}
