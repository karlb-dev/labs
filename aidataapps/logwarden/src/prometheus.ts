export interface PrometheusSample {
  name: string;
  labels: Record<string, string>;
  value: number;
  type: "counter" | "gauge" | "histogram_count" | "histogram_sum" | "histogram_bucket";
}

export function parsePrometheusExposition(source: string): { samples: PrometheusSample[]; metricNames: string[] } {
  const declaredTypes = new Map<string, string>();
  const samples: PrometheusSample[] = [];
  for (const rawLine of source.replaceAll("\r\n", "\n").split("\n")) {
    const line = rawLine.trim();
    if (line === "") continue;
    const type = /^#\s+TYPE\s+([^\s]+)\s+([^\s]+)$/.exec(line);
    if (type !== null) {
      declaredTypes.set(type[1]!, type[2]!);
      continue;
    }
    if (line.startsWith("#")) continue;
    const match = /^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\d+)?$/.exec(line);
    if (match === null) continue;
    const value = parsePrometheusNumber(match[3]!);
    const name = match[1]!;
    samples.push({ name, labels: parseLabels(match[2]), value, type: classify(name, declaredTypes) });
  }
  return { samples, metricNames: [...new Set(samples.map((sample) => sample.name))].sort() };
}

export function sumPrometheusMetric(samples: PrometheusSample[], candidates: string[]): number | null {
  const matching = samples.filter((sample) => candidates.includes(sample.name));
  if (matching.length === 0) return null;
  const finite = matching.map((sample) => sample.value).filter(Number.isFinite);
  return finite.length === 0 ? null : finite.reduce((sum, value) => sum + value, 0);
}

function classify(name: string, declared: Map<string, string>): PrometheusSample["type"] {
  if (name.endsWith("_bucket")) return "histogram_bucket";
  if (name.endsWith("_count")) return "histogram_count";
  if (name.endsWith("_sum")) return "histogram_sum";
  const root = name.replace(/_(bucket|count|sum)$/, "");
  return declared.get(name) === "counter" || declared.get(root) === "counter" ? "counter" : "gauge";
}

function parsePrometheusNumber(value: string): number {
  if (value === "+Inf" || value === "Inf") return Number.POSITIVE_INFINITY;
  if (value === "-Inf") return Number.NEGATIVE_INFINITY;
  if (value === "NaN") return Number.NaN;
  return Number(value);
}

function parseLabels(source: string | undefined): Record<string, string> {
  if (source === undefined || source === "") return {};
  const labels: Record<string, string> = {};
  const pattern = /([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"(?:,|$)/g;
  for (const match of source.matchAll(pattern)) {
    labels[match[1]!] = match[2]!
      .replaceAll("\\n", "\n")
      .replaceAll('\\"', '"')
      .replaceAll("\\\\", "\\");
  }
  return labels;
}
