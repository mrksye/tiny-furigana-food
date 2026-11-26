// Summarise research/results/results.jsonl (latest row per experiment name) as markdown tables.
// usage: bun run research/tools/summary.ts [name-regex]

const rx = new RegExp(process.argv[2] ?? ".");
const lines = (await Bun.file(new URL("../results/results.jsonl", import.meta.url).pathname).text()).split("\n").filter(Boolean);
const latest = new Map<string, any>();
for (const l of lines) { const r = JSON.parse(l); latest.set(r.name, r); }
const sets = ["words_known", "words_unknown_common", "words_unknown_rare", "compounds_unknown", "food_unknown", "places_unknown", "places_known_famous", "place_products_synth", "kenall_towns_unknown", "kenall_municipal_known", "kenall_products_synth", "spec_examples", "menu_items", "menu_items_clean"];
const short = ["known", "unk-common", "unk-rare", "unk-compound", "food", "places", "famous-places(known)", "place+product", "KEN towns(unk)", "KEN municipal(known)", "KEN product(synth)", "spec(18)", "menu(76,private)", "menu-clean(64,private)"];
const pct = (x: number) => (x * 100).toFixed(1);
console.log(`| name | model bytes | dev | ${short.join(" | ")} | CER unk-common | oracle unk-common |`);
console.log(`|${"---|".repeat(short.length + 5)}`);
for (const [name, r] of [...latest].filter(([n]) => rx.test(n)).sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
  const by = Object.fromEntries(r.results.map((x: any) => [x.name, x]));
  const dev = r.breakdown?.devAcc !== undefined ? pct(r.breakdown.devAcc) : "–";
  const uc = by["words_unknown_common"];
  console.log(`| ${name} | ${r.modelBytes} | ${dev} | ${sets.map((s) => (by[s] ? pct(by[s].exact) : "–")).join(" | ")} | ${uc ? pct(uc.cer) : "–"} | ${uc && uc.oracle !== null && !Number.isNaN(uc.oracle) ? pct(uc.oracle) : "–"} |`);
}
