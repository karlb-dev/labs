import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { basename } from "node:path";
import { atomicWrite, resolveRunDirectory } from "../src/run.js";

const driveOnly = process.argv.includes("--drive-only");
const runDirectory = resolveRunDirectory();
const runId = basename(runDirectory);
const labRoot = new URL("../..", import.meta.url).pathname.replace(/\/$/, "");
const repoHandoff = `${labRoot}/inprogress_lab3.md`;
const driveHandoff = "/content/drive/MyDrive/aidataapps/inprogress_lab3.md";
const head = (await command("git", ["rev-parse", "HEAD"], `${labRoot}/..`)).trim();
const branch = (await command("git", ["branch", "--show-current"], `${labRoot}/..`)).trim();
const dirty = (await command("git", ["status", "--porcelain"], `${labRoot}/..`)).trim() !== "";
const atUtc = new Date().toISOString();
let backupReceipt: { receiptSha256?: string } = {};
try { backupReceipt = JSON.parse(await readFile(`${runDirectory}/database/backup-receipt.json`, "utf8")); }
catch { /* the status remains explicit below */ }

const block = [
  "<!-- lab3-watchdog-status:start -->",
  `- Last watchdog checkpoint: ${atUtc}`,
  `- Last watchdog Git head: \`${head}\` on \`${branch}\``,
  `- Last watchdog disposition: ${dirty ? "captured dirty recovery patch; no automatic source commit" : "clean source checkpoint"}`,
  `- Last watchdog database receipt: \`${backupReceipt.receiptSha256 ?? "unavailable"}\``,
  `- Last watchdog run: \`${runId}\``,
  "<!-- lab3-watchdog-status:end -->",
].join("\n");

const source = await readFile(repoHandoff, "utf8");
const updated = replaceBlock(source, block);
if (!driveOnly) await atomicWrite(repoHandoff, updated);
await atomicWrite(driveHandoff, updated);

const resume = `# Resume ${runId}

Last watchdog checkpoint: ${atUtc}

- branch: \`${branch}\`
- Git head: \`${head}\`
- source state at checkpoint: ${dirty ? "dirty; inspect recovery/worktree.patch and untracked archive" : "clean"}
- database backup receipt: \`${backupReceipt.receiptSha256 ?? "unavailable"}\`

\`\`\`bash
cd ${labRoot}/logwarden
source scripts/runtime-env.sh
export LOGWARDEN_RUN_DIR=${runDirectory}
npm ci
npm run db:setup
npm run doctor
npm run check
\`\`\`

No target-model campaign may resume unless its freeze and observability gate
receipts are present and passing.
`;
await atomicWrite(`${runDirectory}/RESUME.md`, resume);
console.log(JSON.stringify({ atUtc, runId, branch, head, dirty, driveOnly, backupReceiptSha256: backupReceipt.receiptSha256 ?? null }, null, 2));

function replaceBlock(sourceText: string, replacement: string): string {
  const pattern = /<!-- lab3-watchdog-status:start -->[\s\S]*?<!-- lab3-watchdog-status:end -->/;
  if (!pattern.test(sourceText)) throw new Error("Lab 3 watchdog status markers are missing");
  return sourceText.replace(pattern, replacement);
}

function command(executable: string, args: string[], cwd: string): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(executable, args, { cwd, encoding: "utf8", maxBuffer: 4 * 1024 * 1024 }, (error, stdout, stderr) => {
      if (error !== null) reject(new Error(`${executable} failed: ${stderr.trim() || error.message}`));
      else resolve(stdout);
    });
  });
}
