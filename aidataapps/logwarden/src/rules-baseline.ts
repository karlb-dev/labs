import { readFileSync } from "node:fs";
import { z } from "zod";
import { actions, incidentClasses, severities } from "./contracts.js";
import { hashJson } from "./hash.js";
import { LAB_ROOT } from "./run.js";

const predictionSchema = z.object({
  incidentClass: z.enum(incidentClasses),
  severity: z.enum(severities),
  action: z.enum(actions),
  confidence: z.number().min(0).max(1),
  abstain: z.boolean(),
}).strict();

const conditionSchema = z.object({
  eventNamesAny: z.array(z.string().min(1).max(120)).min(1).optional(),
  errorNumbersAny: z.array(z.number().int()).min(1).optional(),
  requireNoErrorNumber: z.boolean().optional(),
  minimumSeverity: z.number().int().min(0).max(25).optional(),
  maximumSeverity: z.number().int().min(0).max(25).optional(),
  statesAny: z.array(z.number().int().min(0).max(255)).min(1).optional(),
  messageRegexAny: z.array(z.string().min(1).max(300)).min(1).optional(),
  minimumSameFingerprint5m: z.number().int().min(0).optional(),
  minimumSameClass1h: z.number().int().min(0).optional(),
  minimumOpenRelatedIncidents: z.number().int().min(0).optional(),
}).strict().refine((value) => Object.keys(value).length > 0, "condition must contain at least one predicate")
  .refine((value) => value.minimumSeverity === undefined || value.maximumSeverity === undefined || value.minimumSeverity <= value.maximumSeverity,
    "minimumSeverity must not exceed maximumSeverity");

export const rulesBaselineSchema = z.object({
  schemaVersion: z.literal(1),
  baselineId: z.string().regex(/^B1-[a-z0-9-]+$/),
  authoredAtUtc: z.iso.datetime(),
  authoringProvenance: z.object({
    sources: z.array(z.string().min(1)).min(1),
    packetInspection: z.string().min(1),
    temporalDeviation: z.string().min(1),
    impact: z.string().min(1),
  }).strict(),
  rules: z.array(z.object({
    ruleId: z.string().regex(/^[a-z0-9-]+$/),
    priority: z.number().int().positive(),
    condition: conditionSchema,
    prediction: predictionSchema,
    sourceReferences: z.array(z.string().min(1)).min(1),
  }).strict()).min(1),
  defaultPrediction: predictionSchema,
}).strict();

export type RulesBaseline = z.infer<typeof rulesBaselineSchema>;
export type RulesPrediction = z.infer<typeof predictionSchema>;

export interface RulesBaselineResult {
  baselineId: string;
  rulesetSha256: string;
  resolved: boolean;
  matchedRuleId: string | null;
  matchedEventOrdinals: number[];
  prediction: RulesPrediction;
}

export function loadRulesBaseline(path = `${LAB_ROOT}/config/baselines/rules-v1.json`): RulesBaseline {
  const parsed = rulesBaselineSchema.parse(JSON.parse(readFileSync(path, "utf8")));
  const ruleIds = new Set<string>();
  const priorities = new Set<number>();
  for (const rule of parsed.rules) {
    if (ruleIds.has(rule.ruleId)) throw new Error(`Duplicate rules baseline ID ${rule.ruleId}`);
    if (priorities.has(rule.priority)) throw new Error(`Duplicate rules baseline priority ${rule.priority}`);
    ruleIds.add(rule.ruleId);
    priorities.add(rule.priority);
    for (const pattern of rule.condition.messageRegexAny ?? []) compilePattern(pattern);
  }
  return parsed;
}

export function applyRulesBaseline(packet: unknown, ruleset: RulesBaseline): RulesBaselineResult {
  const normalized = packetSchema.parse(packet);
  const rulesetSha256 = hashJson(ruleset);
  for (const rule of [...ruleset.rules].sort((left, right) => left.priority - right.priority)) {
    if (!recurrenceMatches(normalized.recentHistory, rule.condition)) continue;
    const hasEventPredicate = [
      rule.condition.eventNamesAny, rule.condition.errorNumbersAny, rule.condition.requireNoErrorNumber,
      rule.condition.minimumSeverity, rule.condition.maximumSeverity, rule.condition.statesAny,
      rule.condition.messageRegexAny,
    ].some((value) => value !== undefined);
    const matchedEventOrdinals = hasEventPredicate
      ? normalized.sourceEvents.flatMap((event, index) => eventMatches(event, rule.condition) ? [index] : [])
      : [];
    if (hasEventPredicate && matchedEventOrdinals.length === 0) continue;
    return {
      baselineId: ruleset.baselineId, rulesetSha256, resolved: true,
      matchedRuleId: rule.ruleId, matchedEventOrdinals, prediction: rule.prediction,
    };
  }
  return {
    baselineId: ruleset.baselineId, rulesetSha256, resolved: false,
    matchedRuleId: null, matchedEventOrdinals: [], prediction: ruleset.defaultPrediction,
  };
}

const packetSchema = z.object({
  sourceEvents: z.array(z.object({
    eventName: z.string(),
    errorNumber: z.number().int().nullable(),
    severity: z.number().int().nullable(),
    state: z.number().int().nullable(),
    message: z.string().nullable(),
  }).passthrough()).min(1),
  recentHistory: z.object({
    sameFingerprint5m: z.number().int().min(0),
    sameClass1h: z.number().int().min(0),
    openRelatedIncidents: z.number().int().min(0),
  }).strict(),
}).passthrough();

type Condition = z.infer<typeof conditionSchema>;
type Packet = z.infer<typeof packetSchema>;
type Event = Packet["sourceEvents"][number];

function eventMatches(event: Event, condition: Condition): boolean {
  if (condition.eventNamesAny !== undefined && !condition.eventNamesAny.some((name) => name.toLowerCase() === event.eventName.toLowerCase())) return false;
  if (condition.errorNumbersAny !== undefined && (event.errorNumber === null || !condition.errorNumbersAny.includes(event.errorNumber))) return false;
  if (condition.requireNoErrorNumber === true && event.errorNumber !== null) return false;
  if (condition.minimumSeverity !== undefined && (event.severity === null || event.severity < condition.minimumSeverity)) return false;
  if (condition.maximumSeverity !== undefined && (event.severity === null || event.severity > condition.maximumSeverity)) return false;
  if (condition.statesAny !== undefined && (event.state === null || !condition.statesAny.includes(event.state))) return false;
  if (condition.messageRegexAny !== undefined && (event.message === null || !condition.messageRegexAny.some((pattern) => compilePattern(pattern).test(event.message!)))) return false;
  return true;
}

function recurrenceMatches(history: Packet["recentHistory"], condition: Condition): boolean {
  return (condition.minimumSameFingerprint5m === undefined || history.sameFingerprint5m >= condition.minimumSameFingerprint5m)
    && (condition.minimumSameClass1h === undefined || history.sameClass1h >= condition.minimumSameClass1h)
    && (condition.minimumOpenRelatedIncidents === undefined || history.openRelatedIncidents >= condition.minimumOpenRelatedIncidents);
}

function compilePattern(source: string): RegExp {
  try {
    return new RegExp(source, "iu");
  } catch (error) {
    throw new Error(`Invalid rules baseline regex ${JSON.stringify(source)}`, { cause: error });
  }
}
