import { existsSync, readFileSync, readdirSync } from "node:fs";
import { z } from "zod";
import { loadMacModelRegistry } from "../src/models-mac.js";
import { LAB_ROOT, atomicWrite, resolveRunDirectory, valueAfter } from "../src/run.js";

// Renders the mac bench report from retained eval rows (SPEC §28.5: reports
// are generated from rows, never hand-assembled). Numbers come exclusively
// from the run directory's mac-eval outputs; words come from a narrative JSON
// an agent authors separately.
//
//   npm run mac:report                                  # current run, default narrative
//   npm run mac:report -- --narrative path.json --out out.html
//
// Narrative JSON: prose blocks (verdict, tiles, ledes, dossiers, ablation,
// caveats), display metadata (model order/name/palette slot/runtime), and
// optional scatter label offsets. See templates/mac-eval-narrative.example.json.

const displaySchema = z.object({
  profileKey: z.string(),
  name: z.string(),
  // 1–8 = validated categorical slots; 0 = neutral "retired serving" treatment.
  slot: z.number().int().min(0).max(8),
  runtime: z.string(),
});

const narrativeSchema = z.object({
  eyebrow: z.string(),
  title: z.string(),
  subtitle: z.string(),
  meta: z.array(z.string()).default([]),
  display: z.array(displaySchema).min(1),
  verdict: z.array(z.string()),
  tiles: z.array(z.object({ lab: z.string(), val: z.string(), sub: z.string() })),
  scoreboard: z.object({ title: z.string(), lede: z.string() }),
  quality: z.object({ title: z.string(), lede: z.string() }),
  scatter: z.object({ title: z.string(), lede: z.string() }),
  grid: z.object({ title: z.string(), lede: z.string() }),
  dossierTitle: z.string().default("Signature behaviors"),
  dossiers: z.array(z.object({ key: z.string(), title: z.string(), facts: z.string(), body: z.string(), snip: z.string() })).default([]),
  ablation: z.object({
    kicker: z.string(),
    title: z.string(),
    lede: z.string(),
    cards: z.array(z.object({ lab: z.string(), big: z.string(), body: z.string() })),
  }).default({ kicker: "", title: "", lede: "", cards: [] }),
  caveatTitle: z.string().default("What this bench is, and is not"),
  caveats: z.array(z.string()).default([]),
  scatterLabelOffsets: z.record(z.string(), z.tuple([z.number(), z.number()])).optional(),
  episodeShorts: z.record(z.string(), z.string()).optional(),
});

const summarySchema = z.object({
  profileKey: z.string(),
  variantId: z.string(),
  loadTimeMs: z.number(),
  episodes: z.number(),
  decisionsProduced: z.number(),
  metrics: z.object({
    classAccuracy: z.number().nullable(),
    severityAccuracy: z.number().nullable(),
    actionAccuracy: z.number().nullable(),
    tripleAccuracy: z.number().nullable(),
    firstPassContractRate: z.number().nullable(),
    legalContractRate: z.number().nullable(),
    requiredToolFirstRate: z.number().nullable(),
    toolArgsCorrectRate: z.number().nullable(),
    spuriousToolRate: z.number().nullable(),
    meanEpisodeLatencyMs: z.number(),
    decodeTokensPerSecond: z.number().nullable(),
  }),
});

const runDirectory = resolveRunDirectory();
const runId = runDirectory.split("/").at(-1)!;
const episodeFile = JSON.parse(readFileSync(
  valueAfter("--episodes") ?? `${LAB_ROOT}/config/mac-eval-episodes.json`, "utf8",
)) as { name: string; episodes: Array<{ episodeId: string; expectedTool?: unknown }> };

const narrativePath = valueAfter("--narrative")
  ?? [`${runDirectory}/reports/mac-eval-narrative.json`, `${LAB_ROOT}/templates/mac-eval-narrative.example.json`]
    .find((candidate) => existsSync(candidate));
if (!narrativePath) throw new Error("No narrative JSON found; pass --narrative or add reports/mac-eval-narrative.json to the run.");
const narrative = narrativeSchema.parse(JSON.parse(readFileSync(narrativePath, "utf8")));

const registry = loadMacModelRegistry();
const metricsDir = `${runDirectory}/metrics`;
const available = new Set(readdirSync(metricsDir)
  .filter((name) => /^mac-eval-.*\.json$/.test(name))
  .map((name) => name.replace(/^mac-eval-/, "").replace(/\.json$/, "")));

interface GridState { [profileKey: string]: string[] }
const episodeIds = episodeFile.episodes.map((episode) => episode.episodeId);

function gridFor(profileKey: string): string[] {
  const path = `${runDirectory}/tables/mac-eval/${profileKey}.jsonl`;
  const states = new Map<string, string>();
  if (existsSync(path)) {
    for (const line of readFileSync(path, "utf8").split("\n").filter(Boolean)) {
      const row = JSON.parse(line) as {
        episodeId: string;
        decision: unknown;
        scores: { classCorrect: boolean; severityCorrect: boolean; actionCorrect: boolean };
      };
      const state = row.decision === null ? "none"
        : row.scores.classCorrect && row.scores.severityCorrect && row.scores.actionCorrect ? "triple"
        : row.scores.classCorrect ? "class" : "wrong";
      states.set(row.episodeId, state);
    }
  }
  return episodeIds.map((id) => states.get(id) ?? "none");
}

const models = [];
const grid: GridState = {};
for (const display of narrative.display) {
  if (!available.has(display.profileKey)) {
    console.warn(`No metrics for ${display.profileKey} in this run; skipping.`);
    continue;
  }
  const summary = summarySchema.parse(JSON.parse(readFileSync(`${metricsDir}/mac-eval-${display.profileKey}.json`, "utf8")));
  const profile = registry.profiles[display.profileKey];
  const m = summary.metrics;
  models.push({
    key: display.profileKey,
    name: display.name,
    slot: display.slot,
    runtime: display.runtime,
    gb: profile ? Number((profile.fileSizeMb / 1000).toFixed(1)) : null,
    cls: m.classAccuracy, sev: m.severityAccuracy, act: m.actionAccuracy, triple: m.tripleAccuracy,
    fp: m.firstPassContractRate, legal: m.legalContractRate,
    toolFirst: m.requiredToolFirstRate, args: m.toolArgsCorrectRate, spur: m.spuriousToolRate,
    tps: m.decodeTokensPerSecond,
    lat: Number((m.meanEpisodeLatencyMs / 1000).toFixed(1)),
    load: summary.loadTimeMs > 0 ? Number((summary.loadTimeMs / 1000).toFixed(1)) : null,
    dec: `${summary.decisionsProduced}/${summary.episodes}`,
  });
  grid[display.profileKey] = gridFor(display.profileKey);
}
if (models.length === 0) throw new Error("No narrative display profile has metrics in this run.");

const episodes = episodeFile.episodes.map((episode) => ({
  id: episode.episodeId.slice(0, 4),
  short: narrative.episodeShorts?.[episode.episodeId]
    ?? episode.episodeId.replace(/^ep\d+-/, "").replace(/-/g, " "),
  tool: episode.expectedTool !== undefined,
}));

const data = {
  generatedAt: new Date().toISOString(),
  runId,
  episodeSet: episodeFile.name,
  meta: narrative.meta.length > 0 ? narrative.meta : [runId, `episodes ${episodeFile.name}`],
  models,
  episodes,
  grid,
  narrative,
};

const template = readFileSync(`${LAB_ROOT}/templates/mac-eval-bench.template.html`, "utf8");
const html = template.replace("__DATA_JSON__", JSON.stringify(data, null, 1));
const outPath = valueAfter("--out") ?? `${runDirectory}/reports/mac-eval-bench.html`;
await atomicWrite(outPath, html);
await atomicWrite(`${runDirectory}/reports/mac-eval-bench-data.json`, `${JSON.stringify(data, null, 2)}\n`);
console.log(JSON.stringify({
  outPath,
  dataPath: `${runDirectory}/reports/mac-eval-bench-data.json`,
  narrativePath,
  models: models.map((model) => model.key),
}, null, 2));
