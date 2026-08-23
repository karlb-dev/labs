import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile, readdir, stat, unlink } from "node:fs/promises";
import { basename, relative, resolve } from "node:path";
import { promisify } from "node:util";
import { retainedCheckpointReceiptNames } from "../src/checkpoint-retention.js";
import { loadConfig } from "../src/config.js";
import { hashFile, hashJson } from "../src/hash.js";
import { atomicWrite, resolveRunDirectory, valueAfter } from "../src/run.js";

interface BackupArtifact {
  fileName: string;
  localPath: string;
  sha256: string;
}

interface BackupReceipt {
  runId: string;
  createdAtUtc: string;
  artifacts: BackupArtifact[];
  receiptSha256: string;
}

interface PinFile {
  schemaVersion: 1;
  runId: string;
  pins: Array<{ label: string; receiptSha256: string; receiptPath: string; createdAtUtc: string }>;
}

interface DeletionTarget {
  plane: "local" | "drive" | "sql-staging";
  path: string;
  relativePath: string;
  bytes: number;
  sha256: string;
}

const apply = process.argv.includes("--apply");
const keepLatest = integerArgument("--keep-latest") ?? 2;
const pinLatest = valueAfter("--pin-latest");
if (!Number.isSafeInteger(keepLatest) || keepLatest < 1 || keepLatest > 20) throw new Error("--keep-latest must be 1..20");
if (pinLatest !== undefined && !/^[a-z0-9][a-z0-9_.-]{0,79}$/i.test(pinLatest)) throw new Error("Unsafe --pin-latest label");

const config = loadConfig();
const runDirectory = resolveRunDirectory();
const runId = basename(runDirectory);
if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$/.test(runId)) throw new Error(`Unsafe run ID: ${runId}`);
const mirrorDirectory = resolve(config.runsMirror, runId);
if (basename(mirrorDirectory) !== runId || !existsSync(mirrorDirectory)) throw new Error(`Run mirror is missing: ${mirrorDirectory}`);
const checkpointDirectory = `${runDirectory}/database/checkpoints`;
const pinPath = `${runDirectory}/database/checkpoint-pins.json`;
const freeze = JSON.parse(await readFile(`${runDirectory}/manifests/freeze.json`, "utf8")) as { databaseCheckpointId?: string };
if (typeof freeze.databaseCheckpointId !== "string") throw new Error("Freeze manifest omits databaseCheckpointId");

const receiptNames = (await readdir(checkpointDirectory)).filter((name) => /^\d{8}T\d{9}Z-backup-receipt\.json$/.test(name)).sort();
const receipts = await Promise.all(receiptNames.map(async (name) => ({
  name,
  value: JSON.parse(await readFile(`${checkpointDirectory}/${name}`, "utf8")) as BackupReceipt,
})));
if (receipts.length < 1 || receipts.some(({ value }) => value.runId !== runId || !Array.isArray(value.artifacts))) {
  throw new Error("Checkpoint receipt set is empty or run-drifted");
}
let pins: PinFile = existsSync(pinPath)
  ? JSON.parse(await readFile(pinPath, "utf8")) as PinFile
  : { schemaVersion: 1, runId, pins: [] };
if (pins.schemaVersion !== 1 || pins.runId !== runId || !Array.isArray(pins.pins)) throw new Error("Checkpoint pin file is invalid");
if (pinLatest !== undefined) {
  const latest = receipts.at(-1)!;
  const prior = pins.pins.find((pin) => pin.label === pinLatest);
  if (prior !== undefined && prior.receiptSha256 !== latest.value.receiptSha256) throw new Error(`Checkpoint label ${pinLatest} is already pinned to another receipt`);
  if (prior === undefined) {
    pins = { ...pins, pins: [...pins.pins, {
      label: pinLatest, receiptSha256: latest.value.receiptSha256,
      receiptPath: `database/checkpoints/${latest.name}`, createdAtUtc: latest.value.createdAtUtc,
    }].sort((left, right) => left.label.localeCompare(right.label)) };
  }
}
const pinnedHashes = new Set([freeze.databaseCheckpointId, ...pins.pins.map((pin) => pin.receiptSha256)]);
const retainedReceiptNames = retainedCheckpointReceiptNames(
  receipts.map(({ name, value }) => ({ name, receiptSha256: value.receiptSha256 })), pinnedHashes, keepLatest,
);
for (const pin of pins.pins) {
  if (!receipts.some(({ value }) => value.receiptSha256 === pin.receiptSha256)) throw new Error(`Pinned checkpoint is absent: ${pin.label}`);
}

const execFileAsync = promisify(execFile);
const composeProject = process.env.COMPOSE_PROJECT_NAME ?? "aidataapps-logwarden";
const inspected = await execFileAsync("docker", ["volume", "inspect", `${composeProject}-sqlserver-data`, "--format", "{{.Mountpoint}}"], { encoding: "utf8" });
const sqlVolumePath = inspected.stdout.trim();
if (!sqlVolumePath.startsWith("/") || basename(sqlVolumePath) !== "_data") throw new Error("SQL data-volume mountpoint failed validation");
const sqlBackupDirectory = `${sqlVolumePath}/backup`;

const deletions: DeletionTarget[] = [];
for (const receipt of receipts.filter(({ name }) => !retainedReceiptNames.has(name))) {
  await addVerifiedPair(`database/checkpoints/${receipt.name}`);
  for (const artifact of receipt.value.artifacts) {
    if (basename(artifact.fileName) !== artifact.fileName) throw new Error(`Unsafe backup file name: ${artifact.fileName}`);
    const backupRelative = `database/backups/${artifact.fileName}`;
    if (resolve(artifact.localPath) !== resolve(runDirectory, backupRelative)) throw new Error(`Backup local path drift: ${artifact.fileName}`);
    await addVerifiedPair(backupRelative, artifact.sha256);
    const sqlPath = resolve(sqlBackupDirectory, artifact.fileName);
    assertInside(sqlBackupDirectory, sqlPath);
    if (existsSync(sqlPath)) await addTarget("sql-staging", sqlPath, `sql-staging/${artifact.fileName}`, artifact.sha256);
  }
}

const recoveryDirectory = `${runDirectory}/recovery`;
const recoveryNames = (await readdir(recoveryDirectory)).filter((name) => /^(?:aidataapps-logwarden-|bundle-verify-|worktree-|untracked-)\d{8}T\d{6}Z\.(?:bundle|txt|patch|list|tar\.gz)$/.test(name));
const recoveryStamps = [...new Set(recoveryNames.map((name) => name.match(/(\d{8}T\d{6}Z)/)?.[1]).filter((value): value is string => value !== undefined))].sort();
const retainedRecoveryStamps = new Set(recoveryStamps.slice(-2));
for (const name of recoveryNames.filter((value) => {
  const stamp = value.match(/(\d{8}T\d{6}Z)/)?.[1];
  return stamp !== undefined && !retainedRecoveryStamps.has(stamp);
})) await addVerifiedPair(`recovery/${name}`);

const deletionBytes = deletions.reduce((sum, target) => sum + target.bytes, 0);
const summary = {
  schemaVersion: 1, runId, mode: apply ? "apply" : "dry-run", keepLatest,
  freezeCheckpointReceiptSha256: freeze.databaseCheckpointId,
  pinnedCheckpoints: pins.pins,
  retainedCheckpointReceipts: [...retainedReceiptNames].sort(),
  retainedRecoveryStamps: [...retainedRecoveryStamps].sort(),
  deletionFileCount: deletions.length, deletionBytes,
  deletionBytesByPlane: Object.fromEntries(["local", "drive", "sql-staging"].map((plane) => [plane,
    deletions.filter((target) => target.plane === plane).reduce((sum, target) => sum + target.bytes, 0)])),
  deletionSetSha256: hashJson(deletions.map((target) => ({ plane: target.plane, relativePath: target.relativePath, bytes: target.bytes, sha256: target.sha256 }))),
};
if (apply) {
  for (const target of deletions) await unlink(target.path);
  await atomicWrite(pinPath, `${JSON.stringify(pins, null, 2)}\n`, 0o600);
  const stamp = new Date().toISOString().replaceAll(/[-:.]/g, "");
  const receiptBody = { ...summary, appliedAtUtc: new Date().toISOString(), disposition: "PASS" };
  const receipt = { ...receiptBody, receiptSha256: hashJson(receiptBody) };
  await atomicWrite(`${runDirectory}/metrics/checkpoint-retention-${stamp}.json`, `${JSON.stringify(receipt, null, 2)}\n`, 0o600);
  console.log(JSON.stringify(receipt, null, 2));
} else {
  console.log(JSON.stringify({ ...summary, disposition: "DRY_RUN" }, null, 2));
}

async function addVerifiedPair(relativePath: string, expectedSha256?: string): Promise<void> {
  if (relativePath.startsWith("/") || relativePath.includes("..")) throw new Error(`Unsafe relative path: ${relativePath}`);
  const localPath = resolve(runDirectory, relativePath);
  const drivePath = resolve(mirrorDirectory, relativePath);
  assertInside(runDirectory, localPath);
  assertInside(mirrorDirectory, drivePath);
  if (!existsSync(localPath) || !existsSync(drivePath)) throw new Error(`Checkpoint mirror pair is incomplete: ${relativePath}`);
  const [localHash, driveHash] = await Promise.all([hashFile(localPath), hashFile(drivePath)]);
  if (localHash !== driveHash || (expectedSha256 !== undefined && localHash !== expectedSha256)) throw new Error(`Checkpoint mirror hash mismatch: ${relativePath}`);
  await addTarget("local", localPath, relativePath, localHash);
  await addTarget("drive", drivePath, relativePath, driveHash);
}

async function addTarget(plane: DeletionTarget["plane"], path: string, relativePath: string, expectedSha256: string): Promise<void> {
  const info = await stat(path);
  if (!info.isFile()) throw new Error(`Retention target is not a file: ${path}`);
  const observed = await hashFile(path);
  if (observed !== expectedSha256) throw new Error(`Retention target hash mismatch: ${path}`);
  deletions.push({ plane, path, relativePath, bytes: info.size, sha256: observed });
}

function assertInside(root: string, path: string): void {
  const value = relative(resolve(root), resolve(path));
  if (value === "" || value.startsWith("..") || value.startsWith("/")) throw new Error(`Path escapes or equals retention root: ${path}`);
}

function integerArgument(name: string): number | undefined {
  const value = valueAfter(name);
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}
