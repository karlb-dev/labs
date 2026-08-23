import { mkdir } from "node:fs/promises";
import { atomicWrite } from "../src/run.js";
import { generateMultiInjectorGateCatalog } from "../src/standard-catalog.js";

await mkdir("config/scenarios", { recursive: true });
const catalog = generateMultiInjectorGateCatalog();
const variants = catalog.scenarios.reduce((total, scenario) => total + (scenario.variants?.length ?? 0), 0);
await atomicWrite("config/scenarios/injector-multi-gate-v2.json", `${JSON.stringify(catalog, null, 2)}\n`);
console.log(JSON.stringify({ catalogId: catalog.catalogId, templates: catalog.scenarios.length, variants }, null, 2));
