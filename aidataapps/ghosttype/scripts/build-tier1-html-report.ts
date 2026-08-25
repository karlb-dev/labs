import { readFileSync, writeFileSync } from "node:fs";

// Renders reports/tier1-report-data.json into the frozen template.
// Emits two files:
//   reports/ghosttype-tier1-report.html          full standalone document
//   reports/ghosttype-tier1-report.fragment.html head/body-less fragment for
//                                                the Artifact publisher
// The Colab campaign regenerates the identical report from its own data by
// running export-report-data.ts there and re-running this script.

const dataPath = process.argv[2] ?? "reports/tier1-report-data.json";
const template = readFileSync("templates/tier1-report.template.html", "utf8");
const data = readFileSync(dataPath, "utf8");

const PLACEHOLDER = '"__GHOSTTYPE_DATA__"';
if (!template.includes(PLACEHOLDER)) throw new Error("template placeholder missing");
// replacement callback: a plain string here would have $-patterns in the
// JSON (e.g. SQL text containing $' or $&) expanded by String.replace
const payload = data.trim().replace(/<\/script/gi, "<\\/script");
const fragment = template.replace(PLACEHOLDER, () => payload);

writeFileSync("reports/ghosttype-tier1-report.fragment.html", fragment);
writeFileSync("reports/ghosttype-tier1-report.html",
  `<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n${fragment}\n</body>\n</html>\n`);
console.log(JSON.stringify({
  fragmentBytes: fragment.length,
  out: ["reports/ghosttype-tier1-report.html", "reports/ghosttype-tier1-report.fragment.html"],
}));
