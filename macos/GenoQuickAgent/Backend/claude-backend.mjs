import { fileURLToPath } from "node:url";

const DEFAULT_MODEL = "sonnet";
const SYSTEM_PROMPT =
  "Answer the user's one-off question directly and concisely. Prefer one short paragraph; use bullets only when they materially improve clarity. Do not add conversational filler.";

export class BackendError extends Error {
  constructor(code, message) {
    super(message);
    this.name = "BackendError";
    this.code = code;
  }
}

function requiredString(value, field) {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new BackendError("invalid_request", `Missing ${field}.`);
  }
  return value.trim();
}

function resultFailureMessage(message) {
  if (Array.isArray(message?.errors) && message.errors.length > 0) {
    return message.errors.map(String).join("; ").slice(0, 600);
  }
  if (typeof message?.result === "string" && message.result.trim()) {
    return message.result.trim().slice(0, 600);
  }
  return "Claude could not complete the request.";
}

export async function answerRequest(request, queryFn) {
  const id = requiredString(request?.id, "request id");
  const question = requiredString(request?.question, "question");
  const model =
    typeof request?.model === "string" && request.model.trim()
      ? request.model.trim()
      : DEFAULT_MODEL;

  const queryInput = {
    prompt: question,
    options: {
      model,
      maxTurns: 1,
      allowedTools: [],
      permissionMode: "dontAsk",
      systemPrompt: SYSTEM_PROMPT,
      cwd: process.env.GENO_CLAUDE_CWD || process.env.HOME || process.cwd(),
      env: process.env,
    },
  };

  for await (const message of queryFn(queryInput)) {
    if (message?.type !== "result") continue;
    if (message.subtype !== "success") {
      throw new BackendError("sdk_failed", resultFailureMessage(message));
    }
    const answer = typeof message.result === "string" ? message.result.trim() : "";
    if (!answer) {
      throw new BackendError("empty_answer", "Claude returned an empty answer.");
    }
    return { id, answer };
  }

  throw new BackendError("empty_answer", "Claude returned no final answer.");
}

export function serializeError(id, error) {
  const safeID = typeof id === "string" && id.trim() ? id.trim() : "unknown";
  const code = error instanceof BackendError ? error.code : "sdk_failed";
  const rawMessage = error instanceof Error ? error.message : String(error);
  const message = (rawMessage.trim() || "Claude backend failed.").slice(0, 600);
  return { id: safeID, error: { code, message } };
}

async function readRequest() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  const input = Buffer.concat(chunks).toString("utf8").trim();
  if (!input) throw new BackendError("invalid_request", "No request was provided.");
  try {
    return JSON.parse(input.split(/\r?\n/, 1)[0]);
  } catch {
    throw new BackendError("invalid_request", "The request was not valid JSON.");
  }
}

async function main() {
  let request;
  try {
    request = await readRequest();
    const { query } = await import("@anthropic-ai/claude-agent-sdk");
    const response = await answerRequest(request, query);
    process.stdout.write(`${JSON.stringify(response)}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify(serializeError(request?.id, error))}\n`);
    process.exitCode = 1;
  }
}

if (fileURLToPath(import.meta.url) === process.argv[1]) {
  await main();
}
