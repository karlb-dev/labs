import { generatePrimaryRunbookCorpus, headingAwareChunks, runbookCorpusManifest } from "../src/runbooks.js";
import { hashJson } from "../src/hash.js";

describe("primary runbook corpus", () => {
  const corpus = generatePrimaryRunbookCorpus();

  it("contains six distinct guides for each governed incident class", () => {
    expect(corpus.runbooks).toHaveLength(60);
    const counts = new Map<string, number>();
    for (const runbook of corpus.runbooks) counts.set(runbook.incidentClass, (counts.get(runbook.incidentClass) ?? 0) + 1);
    expect([...counts.values()].sort()).toEqual(Array(10).fill(6));
    expect(new Set(corpus.runbooks.map((runbook) => runbook.runbookId)).size).toBe(60);
  });

  it("produces unique heading-aware chunks with stable hashes", () => {
    const chunks = corpus.runbooks.flatMap(headingAwareChunks);
    expect(chunks).toHaveLength(480);
    expect(new Set(chunks.map((chunk) => chunk.contentSha256)).size).toBe(chunks.length);
    expect(chunks.every((chunk) => chunk.tokenCount > 0)).toBe(true);
    expect(hashJson(runbookCorpusManifest(corpus))).toMatch(/^[0-9a-f]{64}$/);
  });

  it("contains no lab identifiers or reserved synthetic error numbers", () => {
    const text = corpus.runbooks.map((runbook) => runbook.bodyMarkdown).join("\n");
    expect(text).not.toMatch(/\bLW_[A-Za-z0-9_]+\b/);
    expect(text).not.toMatch(/\b51\d{3}\b/);
    expect(text).not.toMatch(/smoke-|LW_EVT_/i);
  });
});
