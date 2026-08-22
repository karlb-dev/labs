import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { parse } from "csv-parse/sync";
import { hashFile, hashJson, sha256 } from "../src/hash.js";
import {
  assignSplit,
  charShingles,
  jaccard,
  loadCarriers,
  promptVariantId,
  relationCompletionToQuestion,
  renderCarrier,
  scanSelfName,
} from "../src/prompt-bank.js";
import type { ModelPrintSplit, PromptGroup, PromptVariant } from "../src/types.js";

interface ContentSeed {
  text: string;
  carrierIds: string[];
  context?: string;
  evaluationOnly?: boolean;
  suite?: string;
  metadata?: Record<string, unknown>;
}

interface GroupSeed {
  sourceId: string;
  sourceRowId: string;
  splitKey: string;
  family: string;
  domain: string;
  stratum: string;
  canonicalText: string;
  metadata: Record<string, unknown>;
  contents: ContentSeed[];
}

interface JsonPrompt {
  source_id: string;
  source_row_id: string;
  family: string;
  domain: string;
  stratum: string;
  prompt: string;
  metadata: Record<string, unknown>;
}

const here = new URL(".", import.meta.url);
const labRoot = fileURLToPath(new URL("../", here));
const repoRoot = fileURLToPath(new URL("../../../", here));
const outRoot = `${labRoot}/data/prompt-bank`;
const manifestRoot = `${labRoot}/data/manifests`;
const auditRoot = `${labRoot}/data/audits`;
const carriers = loadCarriers();

function stableOrder(value: string): string {
  return sha256(`modelprint-bank-v2:${value}`);
}

function groupId(sourceId: string, sourceRowId: string): string {
  const safe = sourceRowId.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-|-$/g, "").slice(0, 72);
  return `pg_${sourceId}_${safe}_${sha256(`${sourceId}:${sourceRowId}`).slice(0, 10)}`;
}

async function csv(relative: string): Promise<Record<string, string>[]> {
  const text = await readFile(`${repoRoot}/${relative}`, "utf8");
  if (relative.endsWith("belief_revision_dialogues.csv")) {
    const records = parse(text, { skip_empty_lines: true, bom: true, relax_column_count: true }) as string[][];
    const header = records.shift();
    if (!header) return [];
    return records.map((cells) => {
      const fixed = cells.length > header.length
        ? [...cells.slice(0, 3), cells.slice(3, cells.length - 5).join(","), ...cells.slice(-5)]
        : cells;
      return Object.fromEntries(header.map((column, index) => [column, fixed[index] ?? ""]));
    });
  }
  return parse(text, { columns: true, skip_empty_lines: true, bom: true, relax_quotes: true }) as Record<string, string>[];
}

async function jsonl<T>(path: string): Promise<T[]> {
  return (await readFile(path, "utf8")).split(/\n/).filter(Boolean).map((line) => JSON.parse(line) as T);
}

function relationFamily(row: Record<string, string>): string {
  const match = /(?:^|;)relation=([^;]+)/.exec(row.note ?? "");
  const value = row.family || match?.[1] || row.category || "relation";
  // Co-locate semantically equivalent templates that use different labels in
  // the two repository relation banks. This is intentionally broader than
  // source-row grouping: a template cannot leak merely because its source
  // called it `opposite_of` instead of `antonym`.
  return ({
    opposite_of: "antonym",
    color_of: "object_color",
    language_of: "country_language",
    material_of: "material",
    home_of: "habitat",
  } as Record<string, string>)[value] ?? value;
}

function coding(text: string, metadata: Record<string, unknown>): boolean {
  return metadata.task_kind === "coding" || /\b(?:code|python|javascript|typescript|sql|function|program|algorithm)\b/i.test(text);
}

function externalSecondary(row: JsonPrompt, split: ModelPrintSplit): string {
  if (split !== "train" && split !== "calibration") return "structured-v1";
  if (coding(row.prompt, row.metadata)) return "code-explain-v1";
  if (row.domain === "creative_writing") return "expand-v1";
  return "explain-v1";
}

async function buildSeeds(): Promise<GroupSeed[]> {
  const seeds: GroupSeed[] = [];
  const canonicalSeen = new Set<string>();
  const add = (seed: GroupSeed, deduplicate = true) => {
    const key = seed.canonicalText.trim().toLowerCase().replace(/\s+/g, " ");
    if (deduplicate && canonicalSeen.has(key)) return;
    canonicalSeen.add(key);
    seeds.push(seed);
  };

  // Advanced relation rows are loaded first so the 20 overlaps in the basic
  // set resolve to the richer source record.
  for (const row of await csv("interpretability/data/advanced_relation_geometry.csv")) {
    const question = relationCompletionToQuestion(row.prompt!);
    const family = relationFamily(row);
    const split = assignSplit(`relation-template:${family}`, "advanced-relations");
    const primary = split === "test_id" ? "structured-v1" : "explain-v1";
    add({
      sourceId: "advanced-relations", sourceRowId: row.item_id!, splitKey: `relation-template:${family}`,
      family: `relation-${family}`, domain: row.entity_group || "relations", stratum: "controlled-relation",
      canonicalText: question, metadata: { template: row.template, swap_group: row.swap_group, subject: row.subject },
      contents: [{ text: question, carrierIds: [primary], metadata: { converted_from_completion: row.prompt } }],
    }, true);
  }
  for (const row of await csv("interpretability/data/relation_probes_lab1.csv")) {
    const question = relationCompletionToQuestion(row.prompt!);
    const family = relationFamily(row);
    const splitKey = `relation-template:${family}`;
    const split = assignSplit(splitKey, "relation-probes");
    add({
      sourceId: "relation-probes", sourceRowId: row.example_id!, splitKey, family: `relation-${family}`,
      domain: row.category || "relations", stratum: "controlled-relation", canonicalText: question,
      metadata: { note: row.note, target: row.target?.trim(), distractor: row.distractor?.trim() },
      contents: [{ text: question, carrierIds: [split === "test_id" ? "structured-v1" : "explain-v1"], metadata: { converted_from_completion: row.prompt } }],
    }, true);
  }

  // Exactly 100 relation groups also carry the micro direct view.
  for (const seed of seeds.filter((row) => row.stratum === "controlled-relation").sort((a, b) => stableOrder(a.sourceRowId).localeCompare(stableOrder(b.sourceRowId))).slice(0, 100)) {
    seed.contents.push({ text: seed.canonicalText, carrierIds: ["direct-v1"], suite: "short-text" });
  }

  for (const row of await csv("interpretability/data/sae_feature_corpus.csv")) {
    add({ sourceId: "sae-corpus", sourceRowId: row.text_id!, splitKey: `sae:${row.text_id}`, family: "sae-expansion",
      domain: row.domain || "misc", stratum: "controlled-expansion", canonicalText: row.text!, metadata: {},
      contents: [{ text: row.text!, carrierIds: ["expand-v1"] }] });
  }

  const certainty = new Map<string, GroupSeed>();
  for (const row of await csv("interpretability/data/certainty_calibration_items.csv")) {
    const key = row.topic!;
    const seed = certainty.get(key) ?? {
      sourceId: "certainty", sourceRowId: key, splitKey: `certainty:${key}`, family: `certainty-${row.family}`,
      domain: "reasoning", stratum: "controlled-uncertainty", canonicalText: row.question!, metadata: { topic: key }, contents: [],
    };
    seed.contents.push({ text: row.question!, carrierIds: ["direct-v1", assignSplit(seed.splitKey, seed.sourceId) === "test_id" ? "structured-v1" : "explain-v1"], metadata: { item_id: row.item_id, answerable: row.answerable } });
    certainty.set(key, seed);
  }
  for (const seed of certainty.values()) add(seed);

  for (const row of await csv("interpretability/data/steering_eval_prompts.csv")) {
    const splitKey = `steering:${row.prompt_id}`;
    add({ sourceId: "steering", sourceRowId: row.prompt_id!, splitKey, family: "steering-open",
      domain: "conversation", stratum: "controlled-open", canonicalText: row.prompt!, metadata: {},
      contents: [{ text: row.prompt!, carrierIds: ["direct-v1", assignSplit(splitKey, "steering") === "test_id" ? "structured-v1" : "explain-v1"] }] });
  }

  const persona = new Map<string, GroupSeed>();
  for (const row of await csv("interpretability/data/persona_register_pairs.csv")) {
    const key = row.content_question!;
    const id = row.topic!;
    const seed = persona.get(key) ?? {
      sourceId: "persona", sourceRowId: id, splitKey: `persona-content:${id}`, family: "persona-content",
      domain: row.task_kind || "conversation", stratum: "controlled-persona-content", canonicalText: key,
      metadata: { topic: id, task_kind: row.task_kind }, contents: [{ text: key, carrierIds: ["direct-v1", row.task_kind === "coding" ? "code-explain-v1" : "explain-v1"] }],
    };
    seed.contents.push({ text: row.prompt_positive!, carrierIds: ["persona-v1"], evaluationOnly: true, suite: "persona", metadata: { trait: row.trait, item_id: row.item_id } });
    persona.set(key, seed);
  }
  for (const seed of persona.values()) add(seed);

  const pressure = new Map<string, GroupSeed>();
  for (const row of await csv("interpretability/data/sycophancy_pressure_items.csv")) {
    const key = row.base_id!;
    const seed = pressure.get(key) ?? {
      sourceId: "sycophancy", sourceRowId: key, splitKey: `sycophancy:${key}`, family: "sycophancy-pressure",
      domain: row.domain || "reasoning", stratum: "controlled-pressure", canonicalText: row.question!, metadata: { topic: row.topic }, contents: [],
    };
    if (row.condition === "neutral") {
      seed.contents.push({ text: row.question!, carrierIds: ["direct-v1", assignSplit(seed.splitKey, seed.sourceId) === "test_id" ? "structured-v1" : "explain-v1"], metadata: { condition: "neutral" } });
    } else {
      seed.contents.push({ text: row.user_message!, carrierIds: ["direct-v1"], evaluationOnly: true, suite: "pressure", metadata: { condition: row.condition, pressure_level: row.pressure_level } });
    }
    pressure.set(key, seed);
  }
  for (const seed of pressure.values()) add(seed);

  for (const row of await csv("interpretability/data/belief_revision_dialogues.csv")) {
    const splitKey = `belief:${row.item_id}`;
    add({ sourceId: "belief-revision", sourceRowId: row.item_id!, splitKey, family: `belief-${row.family}`,
      domain: row.family || "reasoning", stratum: "controlled-belief", canonicalText: row.question!,
      metadata: { original_split_ignored: row.split },
      contents: [{ text: row.question!, carrierIds: ["direct-v1", assignSplit(splitKey, "belief-revision") === "test_id" ? "structured-v1" : "explain-v1"] }] });
  }

  const g1Rows = await jsonl<{ prompt_id: string; stratum: string; family: string; text: string }>(`${repoRoot}/interpretability/jspaces/sidelines/gemma/data/g1_prompts_v1.jsonl`);
  for (const row of g1Rows) {
    const splitKey = `g1:${row.prompt_id}`;
    add({ sourceId: "g1", sourceRowId: row.prompt_id, splitKey, family: `g1-${row.family}`, domain: row.stratum,
      stratum: "controlled-g1", canonicalText: row.text, metadata: {},
      contents: [{ text: row.text, carrierIds: ["direct-v1", assignSplit(splitKey, "g1") === "test_id" ? "structured-v1" : "explain-v1"] }] });
  }

  for (const source of ["dolly", "oasst1"] as const) {
    const rows = await jsonl<JsonPrompt>(`${labRoot}/data/external/${source}/prompts.jsonl`);
    for (const row of rows) {
      const split = assignSplit(`${source}:${row.source_row_id}`, source);
      const carriersForRow = ["direct-v1"];
      if (source === "dolly" || Number.parseInt(stableOrder(row.source_row_id).slice(0, 2), 16) < 128) {
        carriersForRow.push(externalSecondary(row, split));
      }
      add({ sourceId: source, sourceRowId: row.source_row_id, splitKey: `${source}:${row.source_row_id}`,
        family: row.family, domain: row.domain, stratum: row.stratum, canonicalText: row.prompt,
        metadata: row.metadata, contents: [{ text: row.prompt, carrierIds: carriersForRow }] });
    }
  }

  const knowledge = JSON.parse(await readFile(`${repoRoot}/aidataapps/rag/data/knowledge.json`, "utf8")) as Array<{
    documentId: string; title: string; chunks: Array<{ chunkId: string; heading: string; content: string }>;
  }>;
  const benchmark = JSON.parse(await readFile(`${repoRoot}/aidataapps/rag/data/benchmark-cases.json`, "utf8")) as Array<{ id: string; query: string }>;
  for (const item of benchmark) {
    add({ sourceId: "rag-lab1", sourceRowId: `benchmark-${item.id}`, splitKey: `rag:${item.id}`, family: "rag-benchmark",
      domain: "northstar-bikes", stratum: "rag-grounded", canonicalText: item.query, metadata: { benchmark_case: item.id },
      contents: [{ text: item.query, context: knowledge.flatMap((doc) => doc.chunks).slice(0, 3).map((chunk) => chunk.content).join("\n\n"), carrierIds: ["rag-grounded-v1"], evaluationOnly: true, suite: "rag-grounded" }] });
  }
  const ragSeeds: GroupSeed[] = [];
  for (const document of knowledge) for (const chunk of document.chunks) {
    const questions = [
      `Explain the practical guidance under “${chunk.heading}”.`,
      `What should a Northstar Bikes operator remember about ${chunk.heading.toLowerCase()}?`,
      `Summarize the evidence about ${chunk.heading.toLowerCase()} and name one limitation.`,
      `Using the context, answer a service-desk question about ${chunk.heading.toLowerCase()}.`,
    ];
    questions.forEach((question, index) => ragSeeds.push({ sourceId: "rag-lab1", sourceRowId: `${chunk.chunkId}-${index}`,
      splitKey: `rag:${chunk.chunkId}:${index}`, family: "rag-knowledge", domain: "northstar-bikes", stratum: "rag-grounded",
      canonicalText: question, metadata: { document_id: document.documentId, chunk_id: chunk.chunkId },
      contents: [{ text: question, context: chunk.content, carrierIds: ["rag-grounded-v1"], evaluationOnly: true, suite: "rag-grounded" }] }));
  }
  for (const seed of ragSeeds.sort((a, b) => stableOrder(a.sourceRowId).localeCompare(stableOrder(b.sourceRowId))).slice(0, 56)) add(seed);
  return seeds;
}

function roundRobinGroups(groups: PromptGroup[]): PromptGroup[] {
  const buckets = new Map<string, PromptGroup[]>();
  for (const group of groups) {
    const values = buckets.get(group.sourceId) ?? [];
    values.push(group);
    buckets.set(group.sourceId, values);
  }
  for (const values of buckets.values()) values.sort((a, b) => stableOrder(a.promptGroupId).localeCompare(stableOrder(b.promptGroupId)));
  const result: PromptGroup[] = [];
  while ([...buckets.values()].some((values) => values.length)) {
    for (const key of [...buckets.keys()].sort()) {
      const value = buckets.get(key)?.shift();
      if (value) result.push(value);
    }
  }
  return result;
}

function csvText(rows: Array<Record<string, unknown>>, columns: string[]): string {
  const escape = (value: unknown) => {
    const text = value === null || value === undefined ? "" : String(value);
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return `${columns.join(",")}\n${rows.map((row) => columns.map((column) => escape(row[column])).join(",")).join("\n")}\n`;
}

async function main() {
  const seeds = await buildSeeds();
  const groups: PromptGroup[] = seeds.map((seed) => ({
    promptGroupId: groupId(seed.sourceId, seed.sourceRowId), sourceId: seed.sourceId, sourceRowId: seed.sourceRowId,
    family: seed.family, domain: seed.domain, stratum: seed.stratum, canonicalText: seed.canonicalText,
    split: assignSplit(seed.splitKey, seed.sourceId), metadata: { ...seed.metadata, split_key: seed.splitKey },
  }));
  const groupById = new Map(groups.map((group) => [group.promptGroupId, group]));
  const variants: PromptVariant[] = [];
  for (const seed of seeds) {
    const id = groupId(seed.sourceId, seed.sourceRowId);
    for (const content of seed.contents) for (const carrierId of content.carrierIds) {
      const carrier = carriers.get(carrierId);
      if (!carrier) throw new Error(`Unknown carrier ${carrierId}`);
      const renderedText = renderCarrier(carrier, content.text, content.context);
      variants.push({ promptVariantId: promptVariantId(id, carrierId, renderedText), promptGroupId: id, carrierId,
        renderedText, renderSha256: sha256(renderedText), maxTokens: carrier.maxTokens,
        evaluationOnly: Boolean(content.evaluationOnly || carrier.evaluationOnly), tierMembership: [],
        metadata: { ...content.metadata, suite: content.suite ?? null } });
    }
  }

  const orderedGroups = roundRobinGroups(groups);
  const tierLimits = { smoke: [32, 32], dev: [200, 320], standard: [1000, 1536], full: [1650, 2500] } as const;
  const tierManifests: Record<string, { groupIds: string[]; variantIds: string[]; hash: string }> = {};
  for (const [tier, [groupLimit, variantLimit]] of Object.entries(tierLimits)) {
    const selectedGroups = orderedGroups.slice(0, groupLimit);
    const selectedIds = new Set(selectedGroups.map((group) => group.promptGroupId));
    const candidates = variants.filter((variant) => selectedIds.has(variant.promptGroupId) && !variant.evaluationOnly);
    const firstByGroup = new Map<string, PromptVariant>();
    for (const variant of candidates.sort((a, b) => stableOrder(`${a.promptGroupId}:${a.carrierId}`).localeCompare(stableOrder(`${b.promptGroupId}:${b.carrierId}`)))) {
      if (!firstByGroup.has(variant.promptGroupId)) firstByGroup.set(variant.promptGroupId, variant);
    }
    const selectedVariants = [...firstByGroup.values()];
    const selectedVariantIds = new Set(selectedVariants.map((variant) => variant.promptVariantId));
    for (const variant of candidates.sort((a, b) => stableOrder(a.promptVariantId).localeCompare(stableOrder(b.promptVariantId)))) {
      if (selectedVariants.length >= variantLimit) break;
      if (!selectedVariantIds.has(variant.promptVariantId)) { selectedVariants.push(variant); selectedVariantIds.add(variant.promptVariantId); }
    }
    for (const variant of selectedVariants) variant.tierMembership.push(tier);
    const body = { groupIds: selectedGroups.map((group) => group.promptGroupId).sort(), variantIds: selectedVariants.map((variant) => variant.promptVariantId).sort() };
    tierManifests[tier] = { ...body, hash: hashJson(body) };
  }

  // Robustness rows are frozen but live outside the 2,500-row main cap.
  for (const variant of variants.filter((row) => row.evaluationOnly)) variant.tierMembership.push("robustness");

  const groupSplits = new Map<string, Set<string>>();
  for (const group of groups) (groupSplits.get(group.promptGroupId) ?? groupSplits.set(group.promptGroupId, new Set()).get(group.promptGroupId))!.add(group.split);
  const overlap = [...groupSplits].filter(([, splits]) => splits.size > 1);
  if (overlap.length) throw new Error(`STOP_DATA: ${overlap.length} prompt groups cross splits`);
  for (const name of ["smoke", "dev", "standard"] as const) {
    const current = new Set(tierManifests[name]!.groupIds);
    const next = new Set(tierManifests[name === "smoke" ? "dev" : name === "dev" ? "standard" : "full"]!.groupIds);
    if ([...current].some((id) => !next.has(id))) throw new Error(`STOP_DATA: tier ${name} is not nested`);
  }

  const canonical = new Map<string, PromptGroup[]>();
  for (const group of groups) {
    const key = group.canonicalText.toLowerCase().replace(/\s+/g, " ").trim();
    const rows = canonical.get(key) ?? []; rows.push(group); canonical.set(key, rows);
  }
  const duplicates = [...canonical.entries()].filter(([, rows]) => rows.length > 1);
  const nearPairs: Array<Record<string, unknown>> = [];
  const buckets = new Map<string, Array<{ group: PromptGroup; grams: Set<string> }>>();
  for (const group of groups) {
    const key = group.canonicalText.toLowerCase().trim().slice(0, 2);
    const grams = charShingles(group.canonicalText);
    for (const other of buckets.get(key) ?? []) {
      const score = jaccard(grams, other.grams);
      if (score >= 0.8 && group.canonicalText !== other.group.canonicalText) nearPairs.push({ left: other.group.promptGroupId, right: group.promptGroupId, jaccard: score });
    }
    const rows = buckets.get(key) ?? []; rows.push({ group, grams }); buckets.set(key, rows);
  }
  const crossSplitNearPairs = nearPairs.filter((pair) => {
    const left = groupById.get(String(pair.left));
    const right = groupById.get(String(pair.right));
    return left?.split !== right?.split;
  });
  if (crossSplitNearPairs.length) throw new Error(`STOP_DATA: ${crossSplitNearPairs.length} near-duplicate prompt pairs cross splits`);

  await Promise.all([mkdir(outRoot, { recursive: true }), mkdir(manifestRoot, { recursive: true }), mkdir(auditRoot, { recursive: true })]);
  const writeJsonl = async (path: string, rows: unknown[]) => writeFile(path, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`);
  await writeJsonl(`${outRoot}/groups.jsonl`, groups);
  await writeJsonl(`${outRoot}/variants.jsonl`, variants);
  for (const [tier, manifest] of Object.entries(tierManifests)) await writeFile(`${manifestRoot}/tier-${tier}.json`, `${JSON.stringify(manifest, null, 2)}\n`);

  const sourcePaths: Record<string, string> = {
    "relation-probes": "interpretability/data/relation_probes_lab1.csv", "advanced-relations": "interpretability/data/advanced_relation_geometry.csv",
    certainty: "interpretability/data/certainty_calibration_items.csv", steering: "interpretability/data/steering_eval_prompts.csv",
    "sae-corpus": "interpretability/data/sae_feature_corpus.csv", persona: "interpretability/data/persona_register_pairs.csv",
    sycophancy: "interpretability/data/sycophancy_pressure_items.csv", "belief-revision": "interpretability/data/belief_revision_dialogues.csv",
    g1: "interpretability/jspaces/sidelines/gemma/data/g1_prompts_v1.jsonl",
  };
  const promptSources: Record<string, unknown>[] = [];
  for (const [id, relative] of Object.entries(sourcePaths)) promptSources.push({ id, path: relative, sha256: await hashFile(`${repoRoot}/${relative}`), kind: id === "rag-lab1" ? "rag" : "repository" });
  const ragInputs = ["aidataapps/rag/data/knowledge.json", "aidataapps/rag/data/benchmark-cases.json"];
  const ragInputHashes = Object.fromEntries(await Promise.all(ragInputs.map(async (relative) => [relative, await hashFile(`${repoRoot}/${relative}`)])));
  promptSources.push({ id: "rag-lab1", kind: "rag", inputs: ragInputHashes, sha256: hashJson(ragInputHashes) });
  const externalManifest = JSON.parse(await readFile(`${labRoot}/data/external/source-manifest.json`, "utf8"));
  promptSources.push({ id: "dolly", kind: "external", ...externalManifest.sources.dolly }, { id: "oasst1", kind: "external", ...externalManifest.sources.oasst1 });
  const sourceManifest = { schemaVersion: 1, sources: promptSources, groupCount: groups.length, variantCount: variants.length, bankHash: hashJson({ groups, variants, tierManifests }) };
  await writeFile(`${manifestRoot}/prompt-sources.json`, `${JSON.stringify(sourceManifest, null, 2)}\n`);
  await writeFile(`${manifestRoot}/prompt-bank.json`, `${JSON.stringify({ groupCount: groups.length, variantCount: variants.length, tiers: tierManifests, bankHash: sourceManifest.bankHash }, null, 2)}\n`);

  const aggregate = (field: keyof PromptGroup) => {
    const counts = new Map<string, number>();
    for (const group of groups) { const key = `${String(group[field])}\u0000${group.split}`; counts.set(key, (counts.get(key) ?? 0) + 1); }
    return [...counts].map(([key, count]) => { const [value, split] = key.split("\u0000"); return { [field]: value, split, count }; });
  };
  const humanControls = await jsonl<{ prompt_group_source_id: string; prompt_group_source_row_id: string; control_id: string }>(`${labRoot}/data/ood/human-authored-controls.jsonl`);
  const groupBySourceRow = new Map(groups.map((group) => [`${group.sourceId}\u0000${group.sourceRowId}`, group]));
  const humanControlAudit = humanControls.map((control) => {
    const group = groupBySourceRow.get(`${control.prompt_group_source_id}\u0000${control.prompt_group_source_row_id}`);
    if (!group) throw new Error(`STOP_DATA: human control ${control.control_id} has no prompt group`);
    return { control_id: control.control_id, prompt_group_id: group.promptGroupId, split: group.split };
  });
  await Promise.all([
    writeFile(`${auditRoot}/prompt_group_overlap.csv`, csvText(overlap.map(([id, splits]) => ({ prompt_group_id: id, splits: [...splits].join("|") })), ["prompt_group_id", "splits"])),
    writeFile(`${auditRoot}/canonical_text_duplicates.csv`, csvText(duplicates.map(([text, rows]) => ({ canonical_sha256: sha256(text), count: rows.length, groups: rows.map((row) => row.promptGroupId).join("|") })), ["canonical_sha256", "count", "groups"])),
    writeFile(`${auditRoot}/near_duplicate_pairs.csv`, csvText(nearPairs, ["left", "right", "jaccard"])),
    writeFile(`${auditRoot}/source_by_split.csv`, csvText(aggregate("sourceId"), ["sourceId", "split", "count"])),
    writeFile(`${auditRoot}/family_by_split.csv`, csvText(aggregate("family"), ["family", "split", "count"])),
    writeFile(`${auditRoot}/carrier_by_split.csv`, csvText(variants.map((variant) => ({ carrier: variant.carrierId, split: groupById.get(variant.promptGroupId)!.split, evaluation_only: variant.evaluationOnly })), ["carrier", "split", "evaluation_only"])),
    writeFile(`${auditRoot}/label_balance_by_split.csv`, csvText(aggregate("split"), ["split", "count"])),
    writeFile(`${auditRoot}/prompt_name_leakage.csv`, csvText(groups.filter((group) => scanSelfName(group.canonicalText).selfNameFound).map((group) => ({ prompt_group_id: group.promptGroupId, names: scanSelfName(group.canonicalText).names.join("|") })), ["prompt_group_id", "names"])),
    writeFile(`${auditRoot}/template_family_by_split.csv`, csvText(groups.filter((group) => group.stratum === "controlled-relation").map((group) => ({ family: group.family, split: group.split, prompt_group_id: group.promptGroupId })), ["family", "split", "prompt_group_id"])),
    writeFile(`${auditRoot}/external_source_by_split.csv`, csvText(groups.filter((group) => ["dolly", "oasst1"].includes(group.sourceId)).map((group) => ({ source: group.sourceId, split: group.split, prompt_group_id: group.promptGroupId })), ["source", "split", "prompt_group_id"])),
    writeFile(`${auditRoot}/human_control_by_split.csv`, csvText(humanControlAudit, ["control_id", "prompt_group_id", "split"])),
    writeFile(`${auditRoot}/persona_suite_isolation.csv`, csvText(variants.filter((variant) => variant.carrierId === "persona-v1").map((variant) => ({ prompt_variant_id: variant.promptVariantId, evaluation_only: variant.evaluationOnly, in_train: false })), ["prompt_variant_id", "evaluation_only", "in_train"])),
  ]);
  const summary = { groupCount: groups.length, variantCount: variants.length, primaryVariants: variants.filter((row) => !row.evaluationOnly).length,
    robustnessVariants: variants.filter((row) => row.evaluationOnly).length, splits: Object.fromEntries(["train", "calibration", "test_id", "test_source_holdout"].map((split) => [split, groups.filter((group) => group.split === split).length])),
    duplicates: duplicates.length, nearDuplicates: nearPairs.length, tiers: Object.fromEntries(Object.entries(tierManifests).map(([tier, value]) => [tier, { groups: value.groupIds.length, variants: value.variantIds.length, hash: value.hash }])), bankHash: sourceManifest.bankHash };
  await writeFile(`${auditRoot}/SUMMARY.json`, `${JSON.stringify(summary, null, 2)}\n`);
  console.log(JSON.stringify(summary, null, 2));
}

await main();
