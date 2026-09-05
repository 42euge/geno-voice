import assert from "node:assert/strict";
import test from "node:test";

import {
  BackendError,
  answerRequest,
  serializeError,
} from "./claude-backend.mjs";

test("runs one Sonnet turn with no tools and returns the final answer", async () => {
  let captured;
  async function* fakeQuery(input) {
    captured = input;
    yield { type: "assistant", message: { content: [{ type: "text", text: "draft" }] } };
    yield { type: "result", subtype: "success", result: "  The answer.  " };
  }

  const response = await answerRequest(
    { id: "request-1", question: "What is a closure?", model: "sonnet" },
    fakeQuery,
  );

  assert.deepEqual(response, { id: "request-1", answer: "The answer." });
  assert.equal(captured.prompt, "What is a closure?");
  assert.equal(captured.options.model, "sonnet");
  assert.equal(captured.options.maxTurns, 1);
  assert.deepEqual(captured.options.allowedTools, []);
  assert.equal(captured.options.permissionMode, "dontAsk");
  assert.match(captured.options.systemPrompt, /directly and concisely/i);
});

test("rejects an empty question before invoking Claude", async () => {
  let called = false;
  async function* fakeQuery() {
    called = true;
    yield { type: "result", subtype: "success", result: "unused" };
  }

  await assert.rejects(
    answerRequest({ id: "request-2", question: "   ", model: "sonnet" }, fakeQuery),
    (error) => error instanceof BackendError && error.code === "invalid_request",
  );
  assert.equal(called, false);
});

test("reports a missing final answer as a typed backend error", async () => {
  async function* fakeQuery() {
    yield { type: "result", subtype: "success", result: "" };
  }

  await assert.rejects(
    answerRequest({ id: "request-3", question: "Hello", model: "sonnet" }, fakeQuery),
    (error) => error instanceof BackendError && error.code === "empty_answer",
  );
});

test("serializes errors without leaking stack traces", () => {
  const response = serializeError("request-4", new BackendError("sdk_failed", "Gateway unavailable"));
  assert.deepEqual(response, {
    id: "request-4",
    error: { code: "sdk_failed", message: "Gateway unavailable" },
  });
  assert.equal(JSON.stringify(response).includes("stack"), false);
});
