// candidate-extract-v1 (frozen, addendum A-1): raw model text -> candidate.
// Allowed transformations ONLY:
//   1. strip one leading and one trailing whitespace run (recorded)
//   2. suffix-overlap trim: if the candidate tail equals a prefix of the
//      request suffix (FIM duplication), trim it; suffix_overlap_trimmed set
// Fences/prose/mode leakage are DETECTED and flagged as MODEL_FORMAT
// failures — never repaired. Empty extracted text is an abstention.

export interface Extraction {
  extracted: string;
  extractionVersion: "candidate-extract-v1";
  strippedLeading: number;
  strippedTrailing: number;
  suffixOverlapTrimmed: number;
  formatFailure: "markdown_fence" | "prose" | "mode_leak" | null;
  isAbstention: boolean;
}

const PROSE_LEAD = /^(here('|’)?s|here is|sure|certainly|okay|of course|the (completed|following)|this (query|completes)|i (would|will|can)|note:|explanation)/i;

export function extractCandidate(raw: string, docSuffix: string): Extraction {
  const leading = raw.length - raw.replace(/^\s+/, "").length;
  const trailing = raw.length - raw.replace(/\s+$/, "").length;
  let text = raw.trim();

  let formatFailure: Extraction["formatFailure"] = null;
  if (text.includes("```")) formatFailure = "markdown_fence";
  else if (text.includes("<mode>") || text.includes("<schema_context>") || text.includes("<current_statement_prefix>")) formatFailure = "mode_leak";
  else if (PROSE_LEAD.test(text)) formatFailure = "prose";

  let suffixOverlapTrimmed = 0;
  if (text !== "" && docSuffix !== "") {
    const maxOverlap = Math.min(text.length, docSuffix.length);
    for (let overlap = maxOverlap; overlap > 0; overlap -= 1) {
      if (docSuffix.startsWith(text.slice(text.length - overlap))) {
        suffixOverlapTrimmed = overlap;
        text = text.slice(0, text.length - overlap).replace(/\s+$/, "");
        break;
      }
    }
  }

  return {
    extracted: text,
    extractionVersion: "candidate-extract-v1",
    strippedLeading: leading,
    strippedTrailing: trailing,
    suffixOverlapTrimmed,
    formatFailure,
    isAbstention: text === "",
  };
}
