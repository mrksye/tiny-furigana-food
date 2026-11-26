// Pareto frontier of total distributed bytes vs accuracy for systems A/B/C/D -> research/results/pareto.md + pareto.svg
import { readFileSync, writeFileSync } from "node:fs";

const root = new URL("../..", import.meta.url).pathname;
const JS = { lattice: readFileSync(root + "dist/tiny-g2p.min.js").length, neural: readFileSync(root + "research/models/tiny-g2p-neural.min.js").length };
const latest = new Map<string, any>();
for (const l of readFileSync(root + "research/results/results.jsonl", "utf8").split("\n").filter(Boolean)) { const r = JSON.parse(l); latest.set(r.name, r); }

const SYSTEMS: Record<string, { label: string; js: number; color: string; points: string[] }> = {
  A: { label: "A Symbolic only", js: JS.lattice, color: "#8e8e93", points: ["A-16k", "A-32k"] },
  B: { label: "B Neural only (BiGRU)", js: JS.neural, color: "#d1495b", points: ["B-t8-q4", "B-t16-q6", "B-t32-q6", "B-t64-q6", "B-64k-q6"] },
  C: { label: "C Neuro-symbolic", js: JS.lattice, color: "#0a66c2", points: ["C-16k-d", "C-32k-a", "C-64k-a", "C-100k-mq"] },
  D: { label: "D Compressed dictionary", js: JS.lattice, color: "#2a9d8f", points: ["D-16k", "D-32k", "D-64k", "D-100k"] },
};
const METRICS = [["words_unknown_common", "unknown words"], ["compounds_unknown", "unknown compounds"], ["words_known", "known words"], ["places_unknown", "unknown places"], ["place_products_synth", "place+product (synthetic)"]] as const;
const acc = (r: any, set: string) => r.results.find((x: any) => x.name === set)?.exact ?? NaN;

let md = `| system | experiment | model bytes | JS bytes | total bytes | ${METRICS.map(([, l]) => l).join(" | ")} |\n|${"---|".repeat(5 + METRICS.length)}\n`;
for (const [k, s] of Object.entries(SYSTEMS)) for (const p of s.points) {
  const r = latest.get(p);
  if (!r) continue;
  md += `| ${k} | ${p} | ${r.modelBytes} | ${s.js} | ${r.modelBytes + s.js} | ${METRICS.map(([m]) => (acc(r, m) * 100).toFixed(1)).join(" | ")} |\n`;
}
writeFileSync(root + "research/results/pareto.md", md);

// SVG: x = total bytes (log2), y = accuracy on unknown words; tier lines at 8/16/32/64 KiB and 100 kB
const W = 720, H = 420, L = 64, R = 24, T = 28, B = 56;
const xs = (b: number) => L + ((Math.log2(b) - Math.log2(4096)) / (Math.log2(131072) - Math.log2(4096))) * (W - L - R);
const ys = (a: number) => T + (1 - a) * (H - T - B);
let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="system-ui, sans-serif" font-size="12">\n<rect width="${W}" height="${H}" fill="#fff"/>\n`;
for (let a = 0; a <= 1.0001; a += 0.2) svg += `<line x1="${L}" x2="${W - R}" y1="${ys(a)}" y2="${ys(a)}" stroke="#e5e5ea"/><text x="${L - 8}" y="${ys(a) + 4}" text-anchor="end" fill="#6e6e73">${Math.round(a * 100)}%</text>\n`;
for (const [b, lab] of [[8192, "8 KiB"], [16384, "16 KiB"], [32768, "32 KiB"], [65536, "64 KiB"], [100000, "100 kB"]] as const)
  svg += `<line x1="${xs(b)}" x2="${xs(b)}" y1="${T}" y2="${H - B}" stroke="#c7c7cc" stroke-dasharray="3 3"/><text x="${xs(b)}" y="${H - B + 16}" text-anchor="middle" fill="#6e6e73">${lab}</text>\n`;
svg += `<text x="${(L + W - R) / 2}" y="${H - 14}" text-anchor="middle" fill="#1d1d1f">total distributed bytes (minified JS + model), log scale</text>\n`;
svg += `<text x="16" y="${(T + H - B) / 2}" transform="rotate(-90 16 ${(T + H - B) / 2})" text-anchor="middle" fill="#1d1d1f">exact match, unknown common words</text>\n`;
let ly = T + 6;
for (const [, s] of Object.entries(SYSTEMS)) {
  const pts = s.points.map((p) => latest.get(p)).filter(Boolean).map((r) => [xs(r.modelBytes + s.js), ys(acc(r, "words_unknown_common"))]);
  svg += `<polyline fill="none" stroke="${s.color}" stroke-width="2" points="${pts.map((p) => p.join(",")).join(" ")}"/>\n`;
  for (const [x, y] of pts) svg += `<circle cx="${x}" cy="${y}" r="4" fill="${s.color}"/>\n`;
  svg += `<rect x="${L + 12}" y="${ly}" width="10" height="10" fill="${s.color}"/><text x="${L + 28}" y="${ly + 9}" fill="#1d1d1f">${s.label}</text>\n`;
  ly += 18;
}
svg += "</svg>\n";
writeFileSync(root + "research/results/pareto.svg", svg);
console.log(md);
