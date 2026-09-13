#!/usr/bin/env node

// Plot the PARSEC CMD points used to motivate the warm-star CMD likelihood cut.
// This small Node script is kept dependency-light so the diagnostic can also be
// generated on machines where the project's Conda environment is unavailable.

const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const repoRoot = path.resolve(__dirname, "../..");
const inputPath = path.join(
  repoRoot,
  "data/PARSEC_logAge_6to10_0p5_MH_n1to0p6_0p2.csv",
);
const outputDir = path.join(repoRoot, "results/parsec_teff_threshold_cmd");
const svgPath = path.join(outputDir, "parsec_cmd_teff_4700_4800.svg");
const pngPath = path.join(outputDir, "parsec_cmd_teff_4700_4800.png");

const lines = fs.readFileSync(inputPath, "utf8").trim().split(/\r?\n/);
const header = lines[0].split(",");
const index = Object.fromEntries(header.map((name, i) => [name, i]));
for (const name of ["MH", "logAge", "logTe", "Gmag", "G_BPmag", "G_RPmag"]) {
  if (!(name in index)) throw new Error(`Missing PARSEC column: ${name}`);
}

const points = [];
for (const line of lines.slice(1)) {
  const row = line.split(",");
  const logAge = Number(row[index.logAge]);
  const g = Number(row[index.Gmag]);
  const bp = Number(row[index.G_BPmag]);
  const rp = Number(row[index.G_RPmag]);
  const mh = Number(row[index.MH]);
  const teff = 10 ** Number(row[index.logTe]);
  if (
    Number.isFinite(logAge) &&
    Number.isFinite(g) &&
    Number.isFinite(bp) &&
    Number.isFinite(rp) &&
    Number.isFinite(mh) &&
    Number.isFinite(teff) &&
    logAge > 9.3 &&
    logAge < 10.0 &&
    g > 3.0 &&
    g < 15.0 &&
    teff >= 4700
  ) {
    points.push({ color: bp - rp, g, mh, teff });
  }
}

const thresholds = [4700, 4800];
const panels = thresholds.map((threshold) => points.filter((point) => point.teff >= threshold));
if (panels.some((panel) => panel.length === 0)) throw new Error("A temperature panel is empty.");

const all = panels[0];
const xValues = all.map((point) => point.color);
const yValues = all.map((point) => point.g);
const xRawMin = Math.min(...xValues);
const xRawMax = Math.max(...xValues);
const yRawMin = Math.min(...yValues);
const yRawMax = Math.max(...yValues);
const xPad = 0.05 * (xRawMax - xRawMin);
const yPad = 0.04 * (yRawMax - yRawMin);
const xMin = xRawMin - xPad;
const xMax = xRawMax + xPad;
const yMin = yRawMin - yPad;
const yMax = yRawMax + yPad;
const mhMin = -1.0;
const mhMax = 0.6;

const width = 1600;
const height = 760;
const left = 105;
const right = 145;
const top = 108;
const bottom = 95;
const gap = 85;
const panelWidth = (width - left - right - gap) / 2;
const panelHeight = height - top - bottom;

const viridisStops = [
  [0.00, [68, 1, 84]],
  [0.25, [59, 82, 139]],
  [0.50, [33, 145, 140]],
  [0.75, [94, 201, 98]],
  [1.00, [253, 231, 37]],
];

function interpolateColor(value) {
  const t = Math.max(0, Math.min(1, (value - mhMin) / (mhMax - mhMin)));
  let upper = 1;
  while (upper < viridisStops.length && viridisStops[upper][0] < t) upper += 1;
  const [t0, c0] = viridisStops[upper - 1];
  const [t1, c1] = viridisStops[Math.min(upper, viridisStops.length - 1)];
  const fraction = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
  const rgb = c0.map((channel, i) => Math.round(channel + fraction * (c1[i] - channel)));
  return `rgb(${rgb.join(",")})`;
}

function xPixel(value, panelIndex) {
  return left + panelIndex * (panelWidth + gap) + ((value - xMin) / (xMax - xMin)) * panelWidth;
}

function yPixel(value) {
  return top + ((value - yMin) / (yMax - yMin)) * panelHeight;
}

function escapeText(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

const svg = [];
svg.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`);
svg.push(`<rect width="100%" height="100%" fill="white"/>`);
svg.push(`<style>
  text { font-family: Arial, Helvetica, sans-serif; fill: #222; }
  .axis { stroke: #333; stroke-width: 1.5; }
  .grid { stroke: #d8d8d8; stroke-width: 1; stroke-dasharray: 4 5; }
  .tick { font-size: 19px; }
  .label { font-size: 23px; }
  .title { font-size: 27px; font-weight: 600; }
  .subtitle { font-size: 18px; fill: #555; }
</style>`);
svg.push(`<text x="${width / 2}" y="38" text-anchor="middle" class="title">PARSEC warm main-sequence CMD</text>`);
svg.push(`<text x="${width / 2}" y="70" text-anchor="middle" class="subtitle">9.3 &lt; log(Age/yr) &lt; 10.0; point colour gives PARSEC [M/H]</text>`);

const xTicks = [];
for (let tick = Math.ceil(xMin * 5) / 5; tick <= xMax + 1e-9; tick += 0.2) xTicks.push(tick);
const yTicks = [];
for (let tick = Math.ceil(yMin * 2) / 2; tick <= yMax + 1e-9; tick += 0.5) yTicks.push(tick);

for (let panelIndex = 0; panelIndex < 2; panelIndex += 1) {
  const panelLeft = left + panelIndex * (panelWidth + gap);
  const panelRight = panelLeft + panelWidth;
  svg.push(`<defs><clipPath id="clip${panelIndex}"><rect x="${panelLeft}" y="${top}" width="${panelWidth}" height="${panelHeight}"/></clipPath></defs>`);
  for (const tick of xTicks) {
    const x = xPixel(tick, panelIndex);
    svg.push(`<line x1="${x}" y1="${top}" x2="${x}" y2="${top + panelHeight}" class="grid"/>`);
    svg.push(`<text x="${x}" y="${top + panelHeight + 31}" text-anchor="middle" class="tick">${tick.toFixed(1)}</text>`);
  }
  for (const tick of yTicks) {
    const y = yPixel(tick);
    svg.push(`<line x1="${panelLeft}" y1="${y}" x2="${panelRight}" y2="${y}" class="grid"/>`);
    if (panelIndex === 0) {
      svg.push(`<text x="${panelLeft - 15}" y="${y + 7}" text-anchor="end" class="tick">${tick.toFixed(1)}</text>`);
    }
  }
  svg.push(`<g clip-path="url(#clip${panelIndex})">`);
  for (const point of panels[panelIndex]) {
    svg.push(`<circle cx="${xPixel(point.color, panelIndex).toFixed(2)}" cy="${yPixel(point.g).toFixed(2)}" r="2.0" fill="${interpolateColor(point.mh)}" fill-opacity="0.62"/>`);
  }
  svg.push(`</g>`);
  svg.push(`<rect x="${panelLeft}" y="${top}" width="${panelWidth}" height="${panelHeight}" fill="none" class="axis"/>`);
  svg.push(`<text x="${(panelLeft + panelRight) / 2}" y="${top - 20}" text-anchor="middle" class="label">T<tspan baseline-shift="sub" font-size="16">eff</tspan> ≥ ${thresholds[panelIndex]} K  (${panels[panelIndex].length.toLocaleString()} points)</text>`);
  svg.push(`<text x="${(panelLeft + panelRight) / 2}" y="${height - 30}" text-anchor="middle" class="label">G<tspan baseline-shift="sub" font-size="16">BP</tspan> − G<tspan baseline-shift="sub" font-size="16">RP</tspan></text>`);
}

svg.push(`<text x="30" y="${top + panelHeight / 2}" text-anchor="middle" class="label" transform="rotate(-90 30 ${top + panelHeight / 2})">PARSEC M<tspan baseline-shift="sub" font-size="16">G</tspan></text>`);

const colorbarX = width - 78;
const colorbarY = top;
const colorbarWidth = 27;
const colorbarHeight = panelHeight;
svg.push(`<defs><linearGradient id="viridis" x1="0" y1="1" x2="0" y2="0">`);
for (const [position, rgb] of viridisStops) {
  svg.push(`<stop offset="${position * 100}%" stop-color="rgb(${rgb.join(",")})"/>`);
}
svg.push(`</linearGradient></defs>`);
svg.push(`<rect x="${colorbarX}" y="${colorbarY}" width="${colorbarWidth}" height="${colorbarHeight}" fill="url(#viridis)" stroke="#333"/>`);
for (const tick of [-1.0, -0.6, -0.2, 0.2, 0.6]) {
  const y = colorbarY + (1 - (tick - mhMin) / (mhMax - mhMin)) * colorbarHeight;
  svg.push(`<line x1="${colorbarX + colorbarWidth}" y1="${y}" x2="${colorbarX + colorbarWidth + 8}" y2="${y}" class="axis"/>`);
  svg.push(`<text x="${colorbarX + colorbarWidth + 15}" y="${y + 7}" class="tick">${tick.toFixed(1)}</text>`);
}
svg.push(`<text x="${colorbarX + colorbarWidth / 2}" y="${colorbarY - 20}" text-anchor="middle" class="label">[M/H]</text>`);
svg.push(`</svg>`);

fs.mkdirSync(outputDir, { recursive: true });
fs.writeFileSync(svgPath, svg.join("\n"));
sharp(Buffer.from(svg.join("\n"))).png().toFile(pngPath).then(() => {
  console.log(JSON.stringify({ inputPath, svgPath, pngPath, xRange: [xMin, xMax], yRange: [yMin, yMax], counts: panels.map((panel) => panel.length) }, null, 2));
});
