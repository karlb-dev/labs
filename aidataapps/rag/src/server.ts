import { RagAgent } from "./agent.js";
import { buildApp } from "./app.js";
import { loadConfig } from "./config.js";
import { VllmGateway } from "./inference.js";
import { SqlServerRepository } from "./repository.js";

const config = loadConfig();
const repository = await SqlServerRepository.connect(config.database);
const inference = new VllmGateway(config.inference);
const agent = new RagAgent(repository, inference, {
  topK: config.topK,
  modelProfile: config.inference.model.key,
});
const app = buildApp({ agent, logger: { level: config.logLevel } });

app.addHook("onClose", async () => repository.close());

const stop = async (signal: string): Promise<void> => {
  app.log.info({ signal }, "shutting down");
  await app.close();
  process.exit(0);
};
process.on("SIGINT", () => void stop("SIGINT"));
process.on("SIGTERM", () => void stop("SIGTERM"));

await app.listen({ host: config.host, port: config.port });
app.log.info(
  {
    modelProfile: config.inference.model.key,
    modelId: config.inference.model.modelId,
    revision: config.inference.model.revision,
  },
  "RAG lab API started",
);
