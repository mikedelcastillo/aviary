// Server-only perceptual hashing (pHash). Pure compute + a sharp decode helper —
// no fs, no caching here (that lives in lib/hash-cache.ts).
//
// Mirrors imagehash.phash(image, hash_size=8) so Hamming thresholds transfer from
// the original Python prototype:
//   grey -> resize 32x32 -> 2D DCT-II -> top-left 8x8 -> bit = coeff > median.
import sharp from "sharp";

const HASH_SIZE = 8; // 8x8 low-freq block -> 64-bit hash
const IMG_SIZE = 32; // HASH_SIZE * highfreq_factor(4), matching imagehash defaults

// Precomputed DCT-II cosine basis: BASIS[k][n] = cos(pi/IMG_SIZE * (n + 0.5) * k).
// Unnormalized (norm=None), exactly like scipy.fftpack.dct used by imagehash.
const BASIS: Float64Array[] = (() => {
  const rows: Float64Array[] = [];
  for (let k = 0; k < IMG_SIZE; k++) {
    const row = new Float64Array(IMG_SIZE);
    for (let n = 0; n < IMG_SIZE; n++) {
      row[n] = Math.cos((Math.PI / IMG_SIZE) * (n + 0.5) * k);
    }
    rows.push(row);
  }
  return rows;
})();

/** 1D DCT-II of a length-IMG_SIZE vector using the precomputed basis. */
function dct1d(input: Float64Array, out: Float64Array): void {
  for (let k = 0; k < IMG_SIZE; k++) {
    const basisK = BASIS[k];
    let sum = 0;
    for (let n = 0; n < IMG_SIZE; n++) sum += input[n] * basisK[n];
    out[k] = sum;
  }
}

/**
 * Compute a 64-bit pHash from a 32x32 grayscale buffer (row-major, length 1024).
 * Pure + deterministic — the unit-testable core (no sharp/fs).
 */
export function phashFromGray(gray: Uint8Array): bigint {
  if (gray.length !== IMG_SIZE * IMG_SIZE) {
    throw new Error(`phashFromGray expects ${IMG_SIZE * IMG_SIZE} bytes, got ${gray.length}`);
  }

  // DCT along rows, then along columns (scipy does axis=0 then axis=1; order is
  // symmetric for the separable 2D transform, so rows-then-cols is equivalent).
  const rowsDct: Float64Array[] = [];
  const tmp = new Float64Array(IMG_SIZE);
  const rowOut = () => new Float64Array(IMG_SIZE);
  for (let r = 0; r < IMG_SIZE; r++) {
    for (let c = 0; c < IMG_SIZE; c++) tmp[c] = gray[r * IMG_SIZE + c];
    const o = rowOut();
    dct1d(tmp, o);
    rowsDct.push(o);
  }
  // DCT down each column.
  const colIn = new Float64Array(IMG_SIZE);
  const colOut = new Float64Array(IMG_SIZE);
  // We only need the top-left 8x8 of the final matrix, so compute all columns but
  // keep just the first HASH_SIZE rows of each column's transform.
  const low: number[] = []; // HASH_SIZE x HASH_SIZE low-freq coefficients, row-major
  const lowCols: Float64Array[] = [];
  for (let c = 0; c < HASH_SIZE; c++) {
    for (let r = 0; r < IMG_SIZE; r++) colIn[r] = rowsDct[r][c];
    dct1d(colIn, colOut);
    lowCols.push(colOut.slice(0, HASH_SIZE));
  }
  for (let r = 0; r < HASH_SIZE; r++) {
    for (let c = 0; c < HASH_SIZE; c++) low.push(lowCols[c][r]);
  }

  // Median of the 64 coefficients (numpy.median = mean of the two middle values).
  const sorted = [...low].sort((a, b) => a - b);
  const median = (sorted[31] + sorted[32]) / 2;

  // bit = coeff > median, row-major MSB-first.
  let hash = 0n;
  for (let i = 0; i < low.length; i++) {
    hash = (hash << 1n) | (low[i] > median ? 1n : 0n);
  }
  return hash;
}

/** Decode an image file to a 64-bit pHash. */
export async function phashFromFile(absPath: string): Promise<bigint> {
  const gray = await sharp(absPath)
    .greyscale()
    .resize(IMG_SIZE, IMG_SIZE, { fit: "fill", kernel: sharp.kernel.lanczos3 })
    .raw()
    .toBuffer();
  return phashFromGray(new Uint8Array(gray.buffer, gray.byteOffset, gray.byteLength));
}

/** Population count of a non-negative 64-bit-ish bigint. */
export function popcount64(x: bigint): number {
  let n = x;
  let count = 0;
  while (n > 0n) {
    n &= n - 1n; // clear lowest set bit
    count++;
  }
  return count;
}

/** Hamming distance between two pHashes. */
export function hamming(a: bigint, b: bigint): number {
  return popcount64(a ^ b);
}
