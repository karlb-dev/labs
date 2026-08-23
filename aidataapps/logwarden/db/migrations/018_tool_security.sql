SET XACT_ABORT ON;

-- Runbooks are intentionally reachable through the bounded retrieval module,
-- not by arbitrary table scans from the runtime identity. Same-owner module
-- execution preserves the positive path while this deny closes the bypass.
DENY SELECT ON SCHEMA::kb TO [lw_agent_role];
