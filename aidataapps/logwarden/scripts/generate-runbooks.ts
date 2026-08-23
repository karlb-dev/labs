import { mkdir } from "node:fs/promises";
import { generatePrimaryRunbookCorpus } from "../src/runbooks.js";
import { atomicWrite } from "../src/run.js";

await mkdir("config/runbooks", { recursive: true });
const corpus = generatePrimaryRunbookCorpus();
await atomicWrite("config/runbooks/primary-v1.json", `${JSON.stringify(corpus, null, 2)}\n`);
console.log(JSON.stringify({ corpusId: corpus.corpusId, runbooks: corpus.runbooks.length }, null, 2));
