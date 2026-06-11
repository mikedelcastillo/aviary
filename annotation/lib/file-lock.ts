// Server-only: serialize async work per key so concurrent read-modify-write
// cycles on the same sidecar file can't lose updates. The atomic temp+rename in
// writeAnnotation guarantees a file is never half-written, but it does NOT stop
// two callers from each reading the same base and the second clobbering the
// first. Keying on the JSON sidecar path makes ops on the same image run in
// series while different images proceed in parallel.
//
// In-memory and per-process only — fine for this single-writer, filesystem-as-DB
// design (one Next dev/start process owns the data).
const chains = new Map<string, Promise<unknown>>();

export function withFileLock<T>(key: string, fn: () => Promise<T>): Promise<T> {
  const prev = chains.get(key) ?? Promise.resolve();
  // Run fn whether the previous link resolved or rejected — never block the
  // chain on an earlier failure.
  const run = prev.then(fn, fn);
  // Park a non-throwing tail as the chain head so later waiters don't see a
  // rejection from an unrelated earlier op.
  const tail = run.then(
    () => undefined,
    () => undefined,
  );
  chains.set(key, tail);
  // Drop the entry once this is the last link, so the map doesn't grow forever.
  tail.finally(() => {
    if (chains.get(key) === tail) chains.delete(key);
  });
  return run;
}
