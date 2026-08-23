import { readFile, writeFile } from "node:fs/promises";
import sql from "mssql";
import { loadConfig } from "../src/config.js";
import { hashJson } from "../src/hash.js";
import { resolveRunDirectory } from "../src/run.js";
import { STYLE_SCHEMA_HASH } from "../src/style.js";
import { repairUnpairedSurrogates } from "./embedding-compat.js";

type CountRow = Record<string, number>;

const runDirectory = resolveRunDirectory();
const freeze = JSON.parse(await readFile(`${runDirectory}/manifests/campaign-freeze.json`, "utf8")) as { campaignId: number; campaignHash: string };
const robustness = await readFile(`${runDirectory}/manifests/robustness-freeze.json`, "utf8")
  .then((value) => JSON.parse(value) as { campaignId: number }).catch(() => null);
const campaignIds = [freeze.campaignId, ...(robustness ? [robustness.campaignId] : [])];
if (!campaignIds.every(Number.isSafeInteger)) throw new Error("Invalid campaign ID in run manifests");
const campaignSql = campaignIds.join(",");
const config = loadConfig();
const pool = await new sql.ConnectionPool({ server: config.database.server, port: config.database.port, user: config.database.user,
  password: config.database.password, database: config.database.database,
  options: { encrypt: false, trustServerCertificate: true }, requestTimeout: 900_000 }).connect();

const count = (row: CountRow, key: string) => Number(row[key] ?? 0);

try {
  const base = await pool.request().query<CountRow>(`
    WITH cg AS (SELECT generation_id FROM dbo.generations WHERE campaign_id IN (${campaignSql})),
    ca AS (SELECT DISTINCT t.text_artifact_id,t.text_view_id,t.artifact_text FROM cg
      JOIN dbo.generation_text_artifacts m ON m.generation_id=cg.generation_id
      JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id),
    raw AS (SELECT * FROM ca WHERE text_view_id='raw-final-v1')
    SELECT
      (SELECT COUNT_BIG(*) FROM cg) AS generations,
      (SELECT COUNT_BIG(*) FROM cg JOIN dbo.generations g ON g.generation_id=cg.generation_id WHERE g.reference_token_count IS NOT NULL) AS referenceTokenized,
      (SELECT COUNT_BIG(*) FROM ca) AS artifacts,
      (SELECT COUNT_BIG(*) FROM raw) AS rawArtifacts,
      (SELECT COUNT_BIG(*) FROM raw WHERE LEN(LTRIM(RTRIM(artifact_text)))>0) AS nonEmptyRawArtifacts,
      (SELECT COUNT_BIG(*) FROM ca JOIN dbo.style_vectors s ON s.text_artifact_id=ca.text_artifact_id AND s.representation_id='style512-v1') AS styleVectors,
      (SELECT COUNT_BIG(*) FROM ca JOIN dbo.output_scalar_features f ON f.text_artifact_id=ca.text_artifact_id AND f.feature_schema_hash='${STYLE_SCHEMA_HASH}') AS scalarFeatures,
      (SELECT COUNT_BIG(*) FROM raw JOIN dbo.output_segments s ON s.text_artifact_id=raw.text_artifact_id) AS segments,
      (SELECT COUNT_BIG(*) FROM raw JOIN dbo.output_segments s ON s.text_artifact_id=raw.text_artifact_id WHERE s.is_primary_eligible=1) AS eligibleSegments;`);
  const baseCounts = base.recordset[0]!;

  const views = await pool.request().query<{ textView: string; artifacts: number }>(`
    WITH ca AS (SELECT DISTINCT t.text_artifact_id,t.text_view_id FROM dbo.generations g
      JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
      JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id WHERE g.campaign_id IN (${campaignSql}))
    SELECT text_view_id AS textView,COUNT_BIG(*) AS artifacts FROM ca GROUP BY text_view_id ORDER BY text_view_id;`);

  const segmenterRows = await pool.request().query<{ segmenter: string; segments: number; artifacts: number; eligibleSegments: number }>(`
    WITH raw AS (SELECT DISTINCT t.text_artifact_id FROM dbo.generations g
      JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
      JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id
      WHERE g.campaign_id IN (${campaignSql}) AND t.text_view_id='raw-final-v1')
    SELECT s.segmenter_id AS segmenter,COUNT_BIG(*) AS segments,COUNT_BIG(DISTINCT s.text_artifact_id) AS artifacts,
      SUM(CONVERT(bigint,s.is_primary_eligible)) AS eligibleSegments
    FROM raw JOIN dbo.output_segments s ON s.text_artifact_id=raw.text_artifact_id
    GROUP BY s.segmenter_id ORDER BY s.segmenter_id;`);

  const segmentUnicodeRows = await pool.request().query<{ id: number; segmenter: string; text: string }>(`
    WITH raw AS (SELECT DISTINCT t.text_artifact_id FROM dbo.generations g
      JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
      JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id
      WHERE g.campaign_id IN (${campaignSql}) AND t.text_view_id='raw-final-v1')
    SELECT s.segment_id AS id,s.segmenter_id AS segmenter,s.segment_text AS text
    FROM raw JOIN dbo.output_segments s ON s.text_artifact_id=raw.text_artifact_id WHERE s.is_primary_eligible=1 ORDER BY s.segment_id;`);
  const unicodeInputRepairs = segmentUnicodeRows.recordset.flatMap((row) => {
    const repaired = repairUnpairedSurrogates(row.text);
    return repaired.replacedCodeUnits ? [{ segmentId: Number(row.id), segmenter: row.segmenter, replacedCodeUnits: repaired.replacedCodeUnits }] : [];
  });

  const profiles = [] as Array<Record<string, unknown>>;
  for (const profile of ["qwen3-embedding-0.6b", "bge-large-en-v1.5"]) {
    const result = await pool.request().input("profile", sql.VarChar(80), profile).query<CountRow>(`
      WITH cg AS (SELECT generation_id FROM dbo.generations WHERE campaign_id IN (${campaignSql})),
      ca AS (SELECT DISTINCT t.text_artifact_id,t.text_view_id FROM cg
        JOIN dbo.generation_text_artifacts m ON m.generation_id=cg.generation_id
        JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id),
      cp AS (SELECT DISTINCT j.prompt_variant_id FROM dbo.generation_jobs j WHERE j.campaign_id IN (${campaignSql})),
      seg AS (SELECT DISTINCT s.segment_id,s.segmenter_id FROM cg
        JOIN dbo.generation_text_artifacts m ON m.generation_id=cg.generation_id
        JOIN dbo.output_segments s ON s.text_artifact_id=m.text_artifact_id WHERE s.is_primary_eligible=1)
      SELECT
        (SELECT COUNT_BIG(*) FROM cp) AS expectedPrompts,
        (SELECT COUNT_BIG(*) FROM cp JOIN dbo.prompt_embeddings p ON p.prompt_variant_id=cp.prompt_variant_id AND p.embedding_profile_id=@profile) AS promptEmbeddings,
        (SELECT COUNT_BIG(*) FROM ca) AS expectedWhole,
        (SELECT COUNT_BIG(*) FROM ca JOIN dbo.semantic_vectors v ON v.text_artifact_id=ca.text_artifact_id AND v.embedding_profile_id=@profile
          AND v.representation_id=CONCAT('whole-',ca.text_view_id)) AS wholeEmbeddings,
        (SELECT COUNT_BIG(*) FROM seg) AS expectedEligibleSegments,
        (SELECT COUNT_BIG(*) FROM seg JOIN dbo.semantic_vectors v ON v.segment_id=seg.segment_id AND v.embedding_profile_id=@profile
          AND v.representation_id=CONCAT('segment-',seg.segmenter_id)) AS segmentEmbeddings;`);
    const row = result.recordset[0]!;
    profiles.push({ profile, expectedPrompts: count(row, "expectedPrompts"), promptEmbeddings: count(row, "promptEmbeddings"),
      expectedWhole: count(row, "expectedWhole"), wholeEmbeddings: count(row, "wholeEmbeddings"),
      expectedEligibleSegments: profile === "qwen3-embedding-0.6b" ? count(row, "expectedEligibleSegments") : 0,
      segmentEmbeddings: count(row, "segmentEmbeddings") });
  }

  const requiredSegmenters = ["paragraph-v1", "sentence-pack-128-v1", "sql-chunks-v1", "token-window-96-24-v1", "whole-v1"];
  const nonEmptyRawArtifacts = count(baseCounts, "nonEmptyRawArtifacts");
  const segmenters = requiredSegmenters.map((name) => {
    const row = segmenterRows.recordset.find((candidate) => candidate.segmenter === name);
    return { segmenter: name, segments: Number(row?.segments ?? 0), artifacts: Number(row?.artifacts ?? 0),
      eligibleSegments: Number(row?.eligibleSegments ?? 0), expectedArtifacts: nonEmptyRawArtifacts,
      complete: Number(row?.artifacts ?? 0) === nonEmptyRawArtifacts };
  });
  const artifactCount = count(baseCounts, "artifacts");
  const complete = count(baseCounts, "generations") === count(baseCounts, "referenceTokenized")
    && artifactCount === count(baseCounts, "styleVectors") && artifactCount === count(baseCounts, "scalarFeatures")
    && segmenters.every((row) => row.complete)
    && profiles.every((row) => row.expectedPrompts === row.promptEmbeddings && row.expectedWhole === row.wholeEmbeddings
      && row.expectedEligibleSegments === row.segmentEmbeddings);
  const summary = {
    schemaVersion: 1,
    runId: runDirectory.split("/").at(-1),
    campaignIds,
    campaignHash: freeze.campaignHash,
    auditedAt: new Date().toISOString(),
    status: complete ? "COMPLETE" : "INCOMPLETE",
    counts: {
      generations: count(baseCounts, "generations"), referenceTokenized: count(baseCounts, "referenceTokenized"),
      artifacts: artifactCount, rawArtifacts: count(baseCounts, "rawArtifacts"), nonEmptyRawArtifacts,
      styleVectors: count(baseCounts, "styleVectors"), scalarFeatures: count(baseCounts, "scalarFeatures"),
      segments: count(baseCounts, "segments"), eligibleSegments: count(baseCounts, "eligibleSegments"),
    },
    artifactsByTextView: views.recordset.map((row) => ({ textView: row.textView, artifacts: Number(row.artifacts) })),
    segmenters,
    embeddingProfiles: profiles,
    embeddingInputPolicies: {
      unicodeRepair: { unpairedSurrogateCodeUnits: "replace-with-U+FFFD" },
      "qwen3-embedding-0.6b": { truncation: null },
      "bge-large-en-v1.5": { truncatePromptTokens: 512, truncationSide: "right" },
    },
    unicodeInputRepairs,
  };
  const artifact = { ...summary, manifestHash: hashJson(summary) };
  await writeFile(`${runDirectory}/metrics/features-completeness.json`, `${JSON.stringify(artifact, null, 2)}\n`);
  await writeFile(`${runDirectory}/manifests/features-complete.json`, `${JSON.stringify(artifact, null, 2)}\n`);
  console.log(JSON.stringify(artifact, null, 2));
  if (!complete) process.exitCode = 2;
} finally {
  await pool.close();
}
