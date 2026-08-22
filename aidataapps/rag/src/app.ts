import Fastify, { type FastifyInstance } from "fastify";
import { ZodError } from "zod";
import { queryRequestSchema, type QueryAgent } from "./types.js";

export function buildApp(options: {
  agent: QueryAgent;
  logger?: boolean | { level: string };
}): FastifyInstance {
  const app = Fastify({ logger: options.logger ?? false });

  app.get("/health", async () => ({ status: "ok" }));
  app.get("/ready", async (_request, reply) => {
    try {
      await options.agent.ready();
      return { status: "ready" };
    } catch (error) {
      reply.code(503);
      return {
        status: "not_ready",
        error: error instanceof Error ? error.message : String(error),
      };
    }
  });

  app.post("/api/query", async (request, reply) => {
    const parsed = queryRequestSchema.safeParse(request.body);
    if (!parsed.success) {
      reply.code(400);
      return { error: "invalid_request", issues: parsed.error.issues };
    }
    return options.agent.query(parsed.data);
  });

  app.setErrorHandler((error, _request, reply) => {
    if (error instanceof ZodError) {
      reply.code(502).send({ error: "invalid_model_output", issues: error.issues });
      return;
    }
    app.log.error(error);
    reply.code(500).send({
      error: "query_failed",
      message: error instanceof Error ? error.message : "Unknown error",
    });
  });

  return app;
}
