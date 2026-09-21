// Stubs https.request so the full live path runs against a synthetic API
// response, for each supported provider. Checks the request shape and, above
// all, that ungrounded items — ones citing a page the search never returned —
// are discarded rather than displayed.
//
//   node test_context_brief.js

const https = require("https");
const fs = require("fs");
const { EventEmitter } = require("events");

const realRequest = https.request;
let captured = null;

function stubResponse(json) {
  https.request = function (opts, cb) {
    captured = { opts, body: "" };
    const req = new EventEmitter();
    req.end = body => {
      captured.body = body;
      const res = new EventEmitter();
      res.statusCode = 200;
      cb(res);
      res.emit("data", Buffer.from(JSON.stringify(json)));
      res.emit("end");
    };
    req.destroy = () => {};
    return req;
  };
}

// One grounded item, one citing a page the search never returned, one missing a
// date; plus an action id that is not in the catalogue.
const MODEL_JSON = {
  no_context_found: false,
  context_items: [
    { name: "Grounded Concert", date: "2026-05-14", venue: "Stadium",
      est_attendance: 40000, distance_km: 1.2, category: "event",
      source_name: "Stadium", source_url: "https://example-stadium.com.my/events/2026",
      confidence: "high" },
    { name: "HALLUCINATED Festival", date: "2026-06-01", venue: "Nowhere",
      category: "event", source_name: "Invented",
      source_url: "https://totally-made-up-source.example/none", confidence: "high" },
    { name: "No date item", venue: "X", source_url: "https://www.mcmc.gov.my/en/media/announcements/x" },
  ],
  summary: "One corroborated event falls inside the quarter.",
  selected_action_ids: ["targeted_measurement", "fabricate_new_action", "operator_information_request"],
};

const MODEL_TEXT = "Here is what I found.\n\n```json\n" + JSON.stringify(MODEL_JSON) + "\n```";
const SEARCHED = [
  { url: "https://www.mcmc.gov.my/en/media/announcements/x", title: "MCMC" },
  { url: "https://example-stadium.com.my/events/2026", title: "Stadium events" },
];

const CASES = {
  anthropic: {
    env: { CONTEXT_PROVIDER: "anthropic", ANTHROPIC_API_KEY: "test-key-not-real" },
    response: {
      stop_reason: "end_turn",
      content: [
        { type: "web_search_tool_result", content: SEARCHED },
        // An error-shaped result block must not crash the URL collector.
        { type: "web_search_tool_result", content: { error_code: "max_uses_exceeded" } },
        { type: "text", text: MODEL_TEXT },
      ],
    },
    expect: {
      hostname: "api.anthropic.com",
      model: "claude-opus-5",
      header: ["x-api-key", "test-key-not-real"],
      prompt: body => body.messages[0].content,
    },
  },
  groq: {
    env: { CONTEXT_PROVIDER: "groq", GROQ_API_KEY: "test-key-not-real" },
    response: {
      choices: [{
        message: {
          content: MODEL_TEXT,
          executed_tools: [
            { type: "search", search_results: { results: SEARCHED } },
            { type: "search", search_results: [] },
          ],
        },
      }],
    },
    expect: {
      hostname: "api.groq.com",
      model: "groq/compound",
      header: ["authorization", "Bearer test-key-not-real"],
      prompt: body => body.messages[0].content,
    },
  },
  perplexity: {
    env: { CONTEXT_PROVIDER: "perplexity", PERPLEXITY_API_KEY: "test-key-not-real" },
    // Sources are top-level, and `citations` is a bare string array the generic
    // key-walker cannot reach — if that concatenation regresses, grounding
    // silently falls back to search_results alone.
    response: {
      choices: [{ message: { content: MODEL_TEXT } }],
      citations: ["https://example-stadium.com.my/events/2026"],
      search_results: [
        { title: "MCMC", url: "https://www.mcmc.gov.my/en/media/announcements/x", snippet: "…" },
      ],
      usage: { prompt_tokens: 1500, completion_tokens: 700 },
    },
    expect: {
      hostname: "api.perplexity.ai",
      model: "sonar",
      header: ["authorization", "Bearer test-key-not-real"],
      prompt: body => body.messages[0].content,
    },
  },
  gemini: {
    env: { CONTEXT_PROVIDER: "gemini", GEMINI_API_KEY: "test-key-not-real" },
    // Gemini reports an opaque Google redirect as `uri` and the real publisher
    // host separately as `domain`. The model cites the publisher URL, so the
    // grounding check only works if `domain` is collected too — if this case
    // regresses, every correctly sourced item silently disappears.
    response: {
      candidates: [{
        content: { parts: [{ text: MODEL_TEXT }] },
        groundingMetadata: {
          groundingChunks: [
            { web: { uri: "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC123",
                     title: "Stadium events", domain: "example-stadium.com.my" } },
            { web: { uri: "https://vertexaisearch.cloud.google.com/grounding-api-redirect/XyZ789",
                     title: "MCMC", domain: "mcmc.gov.my" } },
          ],
        },
      }],
      usageMetadata: { promptTokenCount: 1200, candidatesTokenCount: 900 },
    },
    expect: {
      hostname: "generativelanguage.googleapis.com",
      model: null,  // Gemini puts the model in the URL path, not the body
      header: ["x-goog-api-key", "test-key-not-real"],
      path: "/v1beta/models/gemini-3.5-flash:generateContent",
      prompt: body => body.contents[0].parts[0].text,
    },
  },
  openrouter: {
    env: { CONTEXT_PROVIDER: "openrouter", OPENROUTER_API_KEY: "test-key-not-real" },
    response: {
      choices: [{
        message: {
          content: MODEL_TEXT,
          annotations: [
            { type: "url_citation", url_citation: { url: "https://example-stadium.com.my/events/2026", title: "Stadium" } },
            { type: "url_citation", url_citation: { url: "https://www.mcmc.gov.my/en/media/announcements/x", title: "MCMC" } },
          ],
        },
      }],
      usage: { prompt_tokens: 1400, completion_tokens: 800 },
    },
    expect: {
      hostname: "openrouter.ai",
      model: "google/gemini-3.5-flash",
      header: ["authorization", "Bearer test-key-not-real"],
      prompt: body => body.messages[0].content,
    },
  },
};

const PROVIDER_ENV_KEYS = ["CONTEXT_PROVIDER", "CONTEXT_MODEL",
  "ANTHROPIC_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY",
  "PERPLEXITY_API_KEY",
  // Cleared so a real .env on disk cannot silently switch these cases into
  // separate-search mode and make them assert against the wrong request shape.
  "SEARCH_PROVIDER", "EXA_API_KEY", "SERPER_API_KEY", "TAVILY_API_KEY"];

const failures = [];

async function run(name) {
  const testCase = CASES[name];
  // Provider is resolved at module load, so reload per provider. Explicitly set
  // env wins over any real .env on disk, which loadEnvFile() never overrides.
  for (const key of PROVIDER_ENV_KEYS) process.env[key] = "";
  Object.assign(process.env, testCase.env);
  delete require.cache[require.resolve("./context_brief.js")];
  const { buildContextBrief } = require("./context_brief.js");

  stubResponse(testCase.response);
  const h3 = "8865050001ffffe";  // not a real cell — cleaned up below
  const brief = await buildContextBrief({
    h3, cellName: "Bandar Baru Bangi", lat: 2.95, lon: 101.78, forecastQuarter: "2026Q2",
    cell: { riskBand: "High emerging risk", riskRank: 7, downloadMbps: 18.4, measurementSupport: "typical" },
  }, { refresh: true });
  try { fs.unlinkSync(`./data/context/${h3}.json`); } catch {}

  const sent = JSON.parse(captured.body);
  const prompt = testCase.expect.prompt(sent);
  const ok = (cond, msg) => {
    console.log(`  ${cond ? "PASS" : "FAIL"}  ${msg}`);
    if (!cond) failures.push(`[${name}] ${msg}`);
  };

  console.log(`\n=== ${name} ===`);
  console.log(" request shape");
  ok(captured.opts.hostname === testCase.expect.hostname, `posts to ${testCase.expect.hostname}`);
  ok(captured.opts.headers[testCase.expect.header[0]] === testCase.expect.header[1], `sends ${testCase.expect.header[0]} header`);
  if (testCase.expect.model) {
    ok(sent.model === testCase.expect.model, `model is ${testCase.expect.model} (got ${sent.model})`);
  }
  if (testCase.expect.path) {
    ok(captured.opts.path === testCase.expect.path, `model in URL path (got ${captured.opts.path})`);
  }
  ok(/2026-04-01 to 2026-06-30/.test(prompt), "prompt carries the resolved quarter window");
  ok(/High emerging risk/.test(prompt), "prompt carries the cell evidence");
  ok(/source_url/.test(prompt) && /Never infer/.test(prompt), "prompt carries the grounding rules");

  console.log(" grounding");
  const names = brief.contextItems.map(i => i.name);
  ok(brief.status === "context_found", "status is context_found");
  ok(names.includes("Grounded Concert"), "keeps the item whose source the search returned");
  ok(!names.includes("HALLUCINATED Festival"), "DROPS the item with an unsearched source URL");
  ok(!names.includes("No date item"), "drops the item missing a date");
  ok(brief.droppedUnsourced === 2, `reports 2 discarded items (got ${brief.droppedUnsourced})`);

  console.log(" actions");
  const ids = brief.actions.map(a => a.id);
  ok(!ids.includes("fabricate_new_action"), "DROPS the invented action id");
  ok(ids.length <= 3, `at most 3 actions (got ${ids.length})`);
  ok(brief.actions.every(a => a.ituReference), "every action carries an ITU reference");
  ok(JSON.stringify(ids) === JSON.stringify(["targeted_measurement", "operator_information_request"]),
    `actions in model order, invented one removed (got ${JSON.stringify(ids)})`);
}

// A provider that returns no retrievable sources must display nothing at all,
// even when the model confidently reports events. This is the failure mode of
// pointing CONTEXT_MODEL at a search-less model.
async function runNoSearchResults() {
  for (const key of PROVIDER_ENV_KEYS) process.env[key] = "";
  Object.assign(process.env, { CONTEXT_PROVIDER: "groq", GROQ_API_KEY: "test-key-not-real" });
  delete require.cache[require.resolve("./context_brief.js")];
  const { buildContextBrief } = require("./context_brief.js");

  stubResponse({ choices: [{ message: { content: MODEL_TEXT, executed_tools: [] } }] });
  const h3 = "8865050001ffffd";
  const brief = await buildContextBrief({ h3, cellName: "X", lat: 3, lon: 101, forecastQuarter: "2026Q2", cell: {} }, { refresh: true });
  try { fs.unlinkSync(`./data/context/${h3}.json`); } catch {}

  const ok = (cond, msg) => { console.log(`  ${cond ? "PASS" : "FAIL"}  ${msg}`); if (!cond) failures.push(`[no-search] ${msg}`); };
  console.log("\n=== model with no search results ===");
  ok(brief.contextItems.length === 0, "displays no context items at all");
  ok(brief.status === "no_context", "reports no_context");
  ok(brief.droppedUnsourced === 3, `discards all 3 claimed items (got ${brief.droppedUnsourced})`);
}

(async () => {
  await run("anthropic");
  await run("groq");
  await run("gemini");
  await run("openrouter");
  await run("perplexity");
  await runNoSearchResults();
  https.request = realRequest;
  console.log(`\n${failures.length ? failures.length + " FAILURE(S):\n" + failures.join("\n") : "ALL PASS"}`);
  process.exit(failures.length ? 1 : 0);
})();
