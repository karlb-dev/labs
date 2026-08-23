// normalize-v1 (frozen, campaign identity): candidate/gold comparison
// normalization for normalized_exact. Deliberately case-SENSITIVE — the
// dataset ships case-sensitive-collation edge cases (dbo.CaseSensitive
// Code vs code_shadow) that a case-folding compare would erase.
//   1. Unicode NFC
//   2. trim leading/trailing whitespace
//   3. collapse internal whitespace runs (space/tab/newline) to one space
export function normalizeV1(text: string): string {
  return text.normalize("NFC").trim().replace(/\s+/g, " ");
}

export interface EvalContract {
  expect_empty: boolean;
  forbid_markdown: boolean;
  must_contain: string[];
  must_contain_any: string[];
  must_not_contain: string[];
  required_objects: string[];
  scoring: string[];
}

export interface RowScore {
  outcome: string;
  normalizedExact: boolean | null;
  constraintPass: boolean | null;
  groundingPass: boolean | null;
  emptyPolicyPass: boolean | null;
  noMarkdownPass: boolean | null;
  insertionIntegrityPass: boolean | null;
  parsePass: boolean | null;
  detail: Record<string, unknown>;
}

const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u;

// Deterministic scoring shared by baseline arms and the model replay
// pipeline. `parseIntroduced` is the candidate's structural introduced-error
// count from the ScriptDom service (null = parse oracle not run/ineligible).
export function scoreCandidate(args: {
  candidate: string;
  contract: EvalContract;
  acceptedInsertions: string[];
  docPrefix: string;
  docSuffix: string;
  parseIntroduced: number | null;
}): RowScore {
  const { candidate, contract, acceptedInsertions, docPrefix, docSuffix, parseIntroduced } = args;
  const normalized = normalizeV1(candidate);
  const isEmpty = normalized === "";

  const emptyPolicyPass = contract.expect_empty ? isEmpty : !isEmpty;
  const normalizedExact = contract.expect_empty
    ? isEmpty
    : acceptedInsertions.some((gold) => normalizeV1(gold) === normalized);

  let constraintPass: boolean | null = null;
  if (contract.must_contain.length + contract.must_contain_any.length + contract.must_not_contain.length > 0) {
    const containsAll = contract.must_contain.every((needle) => candidate.includes(needle));
    const containsAny = contract.must_contain_any.length === 0 || contract.must_contain_any.some((needle) => candidate.includes(needle));
    const containsNone = contract.must_not_contain.every((needle) => !candidate.includes(needle));
    constraintPass = containsAll && containsAny && containsNone;
  }

  let groundingPass: boolean | null = null;
  if (contract.required_objects.length > 0) {
    const recomposedLower = (docPrefix + candidate + docSuffix).toLowerCase();
    groundingPass = contract.required_objects.every((objectName) =>
      recomposedLower.includes(objectName.toLowerCase().replace(/[\[\]]/g, "")));
  }

  const noMarkdownPass = contract.forbid_markdown
    ? !candidate.includes("```") && !/^\s*(here is|here's|this |the |sure|certainly)/i.test(candidate)
    : null;

  const insertionIntegrityPass = !LONE_SURROGATE.test(docPrefix + candidate + docSuffix);
  const parsePass = parseIntroduced === null ? null : parseIntroduced === 0;

  const gatePassed = [emptyPolicyPass, constraintPass, groundingPass, noMarkdownPass, insertionIntegrityPass, parsePass]
    .every((gate) => gate !== false);
  const outcome = contract.expect_empty
    ? (isEmpty ? "abstain_correct" : "fail")
    : isEmpty ? "abstain_wrong"
    : noMarkdownPass === false ? "format_fail"
    : normalizedExact && gatePassed ? "success"
    : gatePassed && (constraintPass !== false && groundingPass !== false) ? "partial"
    : "fail";

  return {
    outcome, normalizedExact, constraintPass, groundingPass, emptyPolicyPass,
    noMarkdownPass, insertionIntegrityPass, parsePass,
    detail: { normalizeRule: "normalize-v1", candidateLength: candidate.length },
  };
}
