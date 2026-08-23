// recompose-v1 (frozen): document recomposition for parse/binding oracles.
// Plain concatenation is wrong when the cursor sits at the end of an open
// `--` line comment (intent-mode rows: the prefix is a comment like
// "-- blocking chain ..." with no trailing newline and the completion is a
// full query). Lexically the insertion would be swallowed by the comment; an
// editor inserts intent completions on the next line. So: if the cursor line
// of the prefix is a line comment (and the candidate does not itself start
// with a newline), join prefix and candidate with "\n".
// The same rule is applied to gold at GT-3 and to model candidates at
// scoring time — the recomposition function is part of campaign identity.

export interface Recomposition {
  text: string;
  joiner: "" | "\n";
  rule: "recompose-v1";
}

export function recomposeForParse(prefix: string, candidate: string, suffix: string): Recomposition {
  const lastLineStart = prefix.lastIndexOf("\n") + 1;
  const cursorLine = prefix.slice(lastLineStart);
  const openLineComment = /^\s*--/.test(cursorLine);
  const joiner: "" | "\n" = openLineComment && candidate !== "" && !candidate.startsWith("\n") ? "\n" : "";
  return { text: prefix + joiner + candidate + suffix, joiner, rule: "recompose-v1" };
}
