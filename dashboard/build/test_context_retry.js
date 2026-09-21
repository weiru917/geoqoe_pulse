// Retry policy: transient errors are retried with backoff, permanent ones are
// not. Retrying a 413 or a bad key wastes the demo's time and hides a config
// error behind what looks like a slow network.
const https = require("https");
const { EventEmitter } = require("events");

let attempts = 0;
let script = [];

https.request = function (opts, cb) {
  const req = new EventEmitter();
  req.end = () => {
    const step = script[Math.min(attempts, script.length - 1)];
    attempts++;
    const res = new EventEmitter();
    res.statusCode = step.status;
    res.headers = step.retryAfter ? { "retry-after": String(step.retryAfter) } : {};
    cb(res);
    const payload = step.status >= 400
      ? JSON.stringify({ error: { message: step.msg || "boom" } })
      : JSON.stringify({ choices: [{ message: { content: '```json\n{"no_context_found":true,"context_items":[],"summary":"none","selected_action_ids":["routine_quarterly_review"]}\n```', executed_tools: [] } }] });
    res.emit("data", Buffer.from(payload));
    res.emit("end");
  };
  req.destroy = () => {};
  return req;
};

// Isolate from any real .env: this suite tests provider-native mode.
for (const k of ["SEARCH_PROVIDER", "EXA_API_KEY", "SERPER_API_KEY", "TAVILY_API_KEY", "CONTEXT_MODEL"]) process.env[k] = "";
process.env.CONTEXT_PROVIDER = "groq";
process.env.GROQ_API_KEY = "test-key";
process.env.CONTEXT_TIMEOUT_MS = "5000";
const { buildContextBrief } = require("./context_brief.js");

const failures = [];
const ok = (cond, msg) => { console.log(`  ${cond ? "PASS" : "FAIL"}  ${msg}`); if (!cond) failures.push(msg); };

async function scenario(name, steps, expectedAttempts) {
  attempts = 0; script = steps;
  const h3 = "8865050001ffff" + (Math.floor(Math.random() * 9));
  const t0 = Date.now();
  await buildContextBrief({ h3, cellName: "X", lat: 3, lon: 101, forecastQuarter: "2026Q2", cell: {} }, { refresh: true });
  try { require("fs").unlinkSync(`./data/context/${h3}.json`); } catch {}
  console.log(`\n${name}  (${attempts} attempt(s), ${((Date.now() - t0) / 1000).toFixed(1)}s)`);
  ok(attempts === expectedAttempts, `made ${expectedAttempts} attempt(s), got ${attempts}`);
}

(async () => {
  await scenario("503 then 503 then 200 — transient, should recover",
    [{ status: 503 }, { status: 503 }, { status: 200 }], 3);

  await scenario("413 Request Entity Too Large — permanent, must NOT retry",
    [{ status: 413, msg: "Request Entity Too Large" }], 1);

  await scenario("401 bad key — permanent, must NOT retry",
    [{ status: 401, msg: "invalid key" }], 1);

  await scenario("429 with no retry-after (zero quota) — stops after 2",
    [{ status: 429, msg: "quota" }], 2);

  await scenario("429 with retry-after: 1 — honours header, full 3 attempts",
    [{ status: 429, retryAfter: 1 }, { status: 429, retryAfter: 1 }, { status: 429, retryAfter: 1 }], 3);

  console.log(`\n${failures.length ? failures.length + " FAILURE(S)" : "ALL PASS"}`);
  process.exit(failures.length ? 1 : 0);
})();
