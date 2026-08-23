import sql from "mssql";
import type { AgentSpanLink } from "./agent-loop.js";

export const DEFAULT_TELEMETRY_LINK_BATCH_SIZE = 1_000;

export interface TelemetryLinkResult {
  requested: number;
  updated: number;
  batches: number;
}

export function batchTelemetryLinks(
  links: AgentSpanLink[],
  batchSize = DEFAULT_TELEMETRY_LINK_BATCH_SIZE,
): AgentSpanLink[][] {
  if (!Number.isSafeInteger(batchSize) || batchSize < 1 || batchSize > 10_000) {
    throw new Error(`Invalid telemetry link batch size: ${batchSize}`);
  }
  const keys = new Set<string>();
  for (const link of links) {
    const key = `${link.traceId}:${link.spanId}`;
    if (keys.has(key)) throw new Error(`Duplicate telemetry span link: ${key}`);
    keys.add(key);
  }
  const batches: AgentSpanLink[][] = [];
  for (let offset = 0; offset < links.length; offset += batchSize) {
    batches.push(links.slice(offset, offset + batchSize));
  }
  return batches;
}

export async function linkTelemetrySpans(
  pool: sql.ConnectionPool,
  links: AgentSpanLink[],
  batchSize = DEFAULT_TELEMETRY_LINK_BATCH_SIZE,
): Promise<TelemetryLinkResult> {
  const batches = batchTelemetryLinks(links, batchSize);
  let updated = 0;
  for (const batch of batches) {
    const result = await pool.request()
      .input("links", sql.NVarChar(sql.MAX), JSON.stringify(batch))
      .query<{ updated_count: number }>(`
        WITH source AS
        (
          SELECT trace_id,span_id,turn_id,model_request_id,tool_invocation_id
          FROM OPENJSON(@links) WITH
          (
            trace_id char(32) '$.traceId',
            span_id char(16) '$.spanId',
            turn_id bigint '$.turnId',
            model_request_id bigint '$.modelRequestId',
            tool_invocation_id bigint '$.toolInvocationId'
          )
        )
        UPDATE span SET turn_id=COALESCE(source.turn_id,span.turn_id),
          model_request_id=COALESCE(source.model_request_id,span.model_request_id),
          tool_invocation_id=COALESCE(source.tool_invocation_id,span.tool_invocation_id)
        FROM telemetry.spans span
        INNER JOIN source ON source.trace_id=span.trace_id AND source.span_id=span.span_id;
        SELECT @@ROWCOUNT AS updated_count;
      `);
    const batchUpdated = Number(result.recordset[0]?.updated_count ?? 0);
    if (batchUpdated !== batch.length) {
      throw new Error(`Telemetry link batch updated ${batchUpdated}/${batch.length} spans`);
    }
    updated += batchUpdated;
  }
  return { requested: links.length, updated, batches: batches.length };
}
