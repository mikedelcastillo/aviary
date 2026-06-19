// Regenerates the PWA / home-screen icon set from a single vector source.
//
//   node scripts/generate-icons.mjs   (or: npm run gen:icons)
//
// The mark: a bird framed by green detection-box corner brackets — the app's
// "draw a bounding box around a bird" concept, in the dark theme's box-green.
// Output PNGs land in public/ and public/icons/ and are committed; this script
// only needs re-running to change the artwork.
import sharp from "sharp";
import { mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const PUBLIC = join(here, "..", "public");
const ICONS = join(PUBLIC, "icons");

// Theme tokens mirrored from app/globals.css.
const BG0 = "#0a0a0a"; // --color-bg
const BG1 = "#18181c"; // subtle lift toward --color-surface-2
const GREEN = "#00e676"; // --color-box
const FG = "#ededed"; // --color-fg
const WING = "#c8c8cf"; // dimmer fg for the wing, for a touch of depth

// All geometry is authored in a fixed 512×512 space; outputs rasterize at their
// own pixel size (sharp renders the SVG natively, so every size is crisp).
const VB = 512;

// Green detection-box corner brackets.
const ARM = 72;
const A = 118; // top-left corner
const B = VB - A; // bottom-right corner
const brackets = `
  <g fill="none" stroke="${GREEN}" stroke-width="26" stroke-linecap="round" stroke-linejoin="round">
    <path d="M ${A} ${A + ARM} L ${A} ${A} L ${A + ARM} ${A}" />
    <path d="M ${B - ARM} ${A} L ${B} ${A} L ${B} ${A + ARM}" />
    <path d="M ${B} ${B - ARM} L ${B} ${B} L ${B - ARM} ${B}" />
    <path d="M ${A + ARM} ${B} L ${A} ${B} L ${A} ${B - ARM}" />
  </g>`;

// A simple side-profile songbird, built from overlapping primitives that share a
// fill so they read as one silhouette. Facing right, perched, tail down-left.
const bird = `
  <g>
    <polygon points="208,286 150,334 216,308" fill="${FG}" />
    <ellipse cx="250" cy="272" rx="66" ry="46" fill="${FG}" transform="rotate(-20 250 272)" />
    <circle cx="316" cy="220" r="33" fill="${FG}" />
    <polygon points="344,206 386,202 346,228" fill="${GREEN}" />
    <ellipse cx="252" cy="268" rx="40" ry="22" fill="${WING}" transform="rotate(-20 252 268)" />
    <circle cx="328" cy="214" r="6" fill="${BG0}" />
  </g>`;

/**
 * Build one SVG variant.
 * @param {number} size           output pixel size
 * @param {object} opts
 * @param {boolean} [opts.rounded] rounded-square background (for unmasked use)
 * @param {boolean} [opts.transparent] no background fill (favicon)
 * @param {number} [opts.scale]    shrink content toward center (maskable safe zone)
 */
function svg(size, { rounded = false, transparent = false, scale = 1 } = {}) {
  const r = rounded ? 104 : 0;
  const bg = transparent
    ? ""
    : `<rect x="0" y="0" width="${VB}" height="${VB}" rx="${r}" ry="${r}" fill="url(#g)" />`;
  const content =
    scale === 1
      ? `${brackets}${bird}`
      : `<g transform="translate(${VB / 2} ${VB / 2}) scale(${scale}) translate(${-VB / 2} ${-VB / 2})">${brackets}${bird}</g>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${VB} ${VB}">
  <defs>
    <radialGradient id="g" cx="50%" cy="38%" r="75%">
      <stop offset="0%" stop-color="${BG1}" />
      <stop offset="100%" stop-color="${BG0}" />
    </radialGradient>
  </defs>
  ${bg}
  ${content}
</svg>`;
}

async function png(svgStr, size, outPath) {
  await sharp(Buffer.from(svgStr)).resize(size, size).png().toFile(outPath);
  console.log("  ✓", outPath.replace(PUBLIC, "public"));
}

async function main() {
  await mkdir(ICONS, { recursive: true });

  // Manifest icons ("any" → rounded square so they look right unmasked).
  await png(svg(192, { rounded: true }), 192, join(ICONS, "icon-192.png"));
  await png(svg(512, { rounded: true }), 512, join(ICONS, "icon-512.png"));
  // Maskable → full-bleed square, content pulled into the 80% safe circle.
  await png(svg(512, { scale: 0.78 }), 512, join(ICONS, "maskable-512.png"));
  // Favicon-sized PNG fallback.
  await png(svg(32, { rounded: true }), 32, join(ICONS, "icon-32.png"));

  // iOS home-screen icon — opaque square (iOS rounds the corners itself).
  await png(svg(180), 180, join(PUBLIC, "apple-touch-icon.png"));

  // Crisp vector favicon.
  const { writeFile } = await import("node:fs/promises");
  await writeFile(join(PUBLIC, "icon.svg"), svg(512, { rounded: true }), "utf8");
  console.log("  ✓ public/icon.svg");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
