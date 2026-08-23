import { chmod, copyFile, mkdir, readFile, rm, stat } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { hashFile, hashJson } from "../src/hash.js";
import { atomicWrite, resolveRunDirectory, valueAfter } from "../src/run.js";

interface BackupArtifact { kind: "control" | "workload"; fileName: string; byteCount: number; sha256: string }
interface BackupReceipt { runId: string; artifacts: BackupArtifact[]; receiptSha256: string; [key: string]: unknown }

const runDirectory = resolveRunDirectory();
const stageDirectory = resolve(`${runDirectory}/repro/restore-backups`);
if (stageDirectory !== resolve(runDirectory, "repro/restore-backups")) throw new Error("Unsafe restore staging path");
if (process.argv.includes("--cleanup")) {
  await rm(stageDirectory, { recursive: true, force: true });
  console.log(JSON.stringify({ runId: basename(runDirectory), stageDirectory, disposition: "REMOVED" }, null, 2));
  process.exit(0);
}

const receiptPath = valueAfter("--receipt") ?? `${runDirectory}/database/backup-receipt.json`;
const receipt = JSON.parse(await readFile(receiptPath, "utf8")) as BackupReceipt;
const { receiptSha256, ...receiptBody } = receipt;
if (hashJson(receiptBody) !== receiptSha256 || receipt.runId !== basename(runDirectory)) throw new Error("Backup receipt validation failed");
if (receipt.artifacts.length !== 2 || new Set(receipt.artifacts.map((item) => item.kind)).size !== 2) throw new Error("Expected two distinct backup artifacts");
await rm(stageDirectory, { recursive: true, force: true });
await mkdir(stageDirectory, { recursive: true, mode: 0o700 });
const staged = [];
for (const artifact of [...receipt.artifacts].sort((left, right) => left.kind.localeCompare(right.kind))) {
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\.bak$/.test(artifact.fileName)) throw new Error(`Unsafe backup filename: ${artifact.fileName}`);
  const source = `${runDirectory}/database/backups/${artifact.fileName}`;
  const information = await stat(source);
  if (information.size !== artifact.byteCount || await hashFile(source) !== artifact.sha256) throw new Error(`Backup artifact drift: ${artifact.fileName}`);
  const target = `${stageDirectory}/${artifact.fileName}`;
  await copyFile(source, target);
  await chmod(target, 0o444);
  staged.push({ kind: artifact.kind, fileName: artifact.fileName, byteCount: artifact.byteCount, sha256: artifact.sha256 });
}
await chmod(stageDirectory, 0o555);
const body = { schemaVersion: 1, runId: receipt.runId, sourceBackupReceiptSha256: receiptSha256, staged, disposition: "PASS" };
const result = { ...body, receiptSha256: hashJson(body) };
await atomicWrite(`${runDirectory}/repro/restore-backup-stage.json`, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ ...result, stageDirectory }, null, 2));
