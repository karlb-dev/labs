import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";
import { hashJson, sha256 } from "../src/hash.js";
import { loadModelRegistry } from "../src/models.js";
import { LAB_ROOT } from "../src/run.js";

dotenv.config({ quiet: true });
if (process.env.CONTAINER_RUNTIME_PROFILE === "colab-rootless" && !process.env.DOCKER_HOST) process.env.DOCKER_HOST = "unix:///run/user/1000/docker.sock";

const registry = loadModelRegistry();
const sourcePath = fileURLToPath(new URL("../../rag/config/models.json", import.meta.url));
const source = JSON.parse(readFileSync(sourcePath, "utf8")) as { profiles: Record<string, { modelId: string; revision: string; vllmImage: string }> };
const token = process.env.HF_TOKEN;

function dockerInspect(tag: string) {
  const result = spawnSync("docker", ["inspect", "--format", "{{json .RepoDigests}}|{{.Id}}", tag], { encoding: "utf8" });
  if (result.status !== 0) throw new Error(`Required image is not local: ${tag}: ${result.stderr.trim()}`);
  const [digestsJson, imageId] = result.stdout.trim().split("|");
  return { repoDigests: JSON.parse(digestsJson ?? "[]") as string[], imageId };
}

async function hfJson(repo: string, revision: string, filename: string): Promise<Record<string, unknown> | null> {
  const response = await fetch(`https://huggingface.co/${repo}/resolve/${revision}/${filename}`, {
    headers: token ? { authorization: `Bearer ${token}` } : {}, redirect: "follow",
  });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`HF access failed for ${repo}@${revision}/${filename}: ${response.status}`);
  return JSON.parse(await response.text()) as Record<string, unknown>;
}

async function hfMetadata(repo: string, revision: string) {
  const response = await fetch(`https://huggingface.co/api/models/${repo}/revision/${revision}?blobs=true`, {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });
  if (!response.ok) throw new Error(`HF access failed for ${repo}@${revision}: ${response.status}`);
  const data = await response.json() as { sha?: string; gated?: boolean | string; siblings?: Array<{ rfilename?: string; size?: number; lfs?: { size?: number } }> };
  const weightBytes = (data.siblings ?? []).filter((row) => /\.(?:safetensors|bin)$/.test(row.rfilename ?? "")).reduce((sum, row) => sum + (row.size ?? row.lfs?.size ?? 0), 0);
  return { resolvedRevision: data.sha, gated: data.gated ?? false, weightBytes };
}

const profiles: Record<string, unknown> = {};
for (const [key, profile] of Object.entries(registry.profiles)) {
  const upstream = source.profiles[key];
  if (!upstream || upstream.modelId !== profile.modelId || upstream.revision !== profile.revision) throw new Error(`Lab 1 registry drift for ${key}`);
  const inspected = dockerInspect(profile.vllmSourceTag);
  const expectedDigest = profile.vllmImage.split("@")[1];
  if (!inspected.repoDigests.some((value) => value.endsWith(expectedDigest!))) throw new Error(`Image digest drift for ${key}`);
  const metadata = await hfMetadata(profile.modelId, profile.revision);
  if (metadata.resolvedRevision !== profile.revision) throw new Error(`Revision did not resolve exactly for ${key}`);
  const tokenizer = await hfJson(profile.modelId, profile.revision, "tokenizer_config.json");
  const generation = await hfJson(profile.modelId, profile.revision, "generation_config.json");
  const chatTemplate = tokenizer?.chat_template;
  profiles[key] = {
    ...profile,
    profileHash: hashJson(profile),
    imageId: inspected.imageId,
    repoDigests: inspected.repoDigests,
    hf: metadata,
    chatTemplateSha256: chatTemplate === undefined ? null : sha256(typeof chatTemplate === "string" ? chatTemplate : JSON.stringify(chatTemplate)),
    tokenizerConfigSha256: tokenizer ? hashJson(tokenizer) : null,
    generationConfig: generation,
    generationConfigSha256: generation ? hashJson(generation) : null,
  };
}

const embeddings: Record<string, unknown> = {};
for (const [key, profile] of Object.entries(registry.embeddingProfiles)) {
  const metadata = await hfMetadata(profile.modelId, profile.revision);
  if (metadata.resolvedRevision !== profile.revision) throw new Error(`Embedding revision did not resolve exactly for ${key}`);
  const tokenizer = await hfJson(profile.modelId, profile.revision, "tokenizer_config.json");
  embeddings[key] = { ...profile, profileHash: hashJson(profile), hf: metadata, tokenizerConfigSha256: tokenizer ? hashJson(tokenizer) : null };
}

const snapshot = {
  schemaVersion: 1,
  sourcePath: "aidataapps/rag/config/models.json",
  sourceCommit: registry.sourceCommit,
  sourceSha256: sha256(readFileSync(sourcePath)),
  profiles,
  embeddings,
};
const final = { ...snapshot, snapshotHash: hashJson(snapshot) };
const destination = `${LAB_ROOT}/data/manifests/model-registry-snapshot.json`;
await mkdir(`${LAB_ROOT}/data/manifests`, { recursive: true });
if (existsSync(destination)) {
  const previous = JSON.parse(readFileSync(destination, "utf8")) as { snapshotHash?: string };
  if (previous.snapshotHash && previous.snapshotHash !== final.snapshotHash && !process.argv.includes("--refresh")) {
    throw new Error(`Frozen model registry drift: ${previous.snapshotHash} -> ${final.snapshotHash}. Use --refresh only before MP-2 freeze.`);
  }
}
await writeFile(destination, `${JSON.stringify(final, null, 2)}\n`);
console.log(JSON.stringify({ snapshotHash: final.snapshotHash, profiles: Object.keys(profiles), embeddings: Object.keys(embeddings) }, null, 2));
