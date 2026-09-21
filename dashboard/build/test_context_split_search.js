// Separate-search mode: SEARCH_PROVIDER runs the queries, the LLM only reads
// the results. Verifies the three properties that make this path worth having:
// the model is never asked to search, the request stays small, and the
// grounding allowlist is our own retrieved URL set.
//
//   node test_context_split_search.js

const https = require("https");
const fs = require("fs");
const { EventEmitter } = require("events");

let calls = [];

function stub(routes) {
  https.request = function (opts, cb) {
    const req = new EventEmitter();
    req.end = body => {
      const rec = { host: opts.hostname, path: opts.path, bytes: Buffer.byteLength(body || ""), body: JSON.parse(body) };
      calls.push(rec);
      const payload = routes[opts.hostname];
      const res = new EventEmitter();
      res.statusCode = 200;
      res.headers = {};
      cb(res);
      res.emit("data", Buffer.from(JSON.stringify(typeof payload === "function" ? payload(rec) : payload)));
      res.emit("end");
    };
    req.destroy = () => {};
    return req;
  };
}

// The model cites one URL Exa returned and one it did not.
const MODEL_JSON = {
  no_context_found: false,
  context_items: [
    { name: "Grounded Concert", date: "2026-05-14", venue: "Stadium", distance_km: 1.2,
      category: "event", source_name: "Stadium",
      source_url: "https://example-stadium.com.my/events/2026", confidence: "high" },
    { name: "HALLUCINATED Festival", date: "2026-06-01", category: "event",
      source_name: "Invented", source_url: "https://not-in-our-results.example/x", confidence: "high" },
  ],
  summary: "One corroborated event falls inside the quarter.",
  selected_action_ids: ["targeted_measurement"],
};
const MODEL_TEXT = "```json\n" + JSON.stringify(MODEL_JSON) + "\n```";

stub({
  "api.exa.ai": {
    results: [
      { url: "https://example-stadium.com.my/events/2026", title: "Stadium events 2026",
        text: "A concert is scheduled at the stadium on 14 May 2026. ".repeat(40) },
      { url: "https://www.mcmc.gov.my/en/media/announcements/x", title: "MCMC announcement",
        text: "Public notice. " },
    ],
  },
  "api.groq.com": { choices: [{ message: { content: MODEL_TEXT } }], usage: { prompt_tokens: 2100, completion_tokens: 400 } },
});

process.env.SEARCH_PROVIDER = "exa";
process.env.EXA_API_KEY = "exa-test-key";
process.env.CONTEXT_PROVIDER = "groq";
process.env.GROQ_API_KEY = "groq-test-key";
delete process.env.CONTEXT_MODEL;

const { buildContextBrief } = require("./context_brief.js");

const failures = [];
const ok = (cond, msg) => { console.log(`  ${cond ? "PASS" : "FAIL"}  ${msg}`); if (!cond) failures.push(msg); };

(async () => {
  const h3 = "8865050001ffffe";
  const brief = await buildContextBrief({
    h3, cellName: "Bandar Baru Bangi (SE, 4.5km)", lat: 2.95, lon: 101.78,
    forecastQuarter: "2026Q2", cell: { riskBand: "High emerging risk", riskRank: 1 },
  }, { refresh: true });
  try { fs.unlinkSync(`./data/context/${h3}.json`); } catch {}

  const searchCalls = calls.filter(c => c.host === "api.exa.ai");
  const llmCalls = calls.filter(c => c.host === "api.groq.com");
  const llm = llmCalls[0];
  const prompt = llm.body.messages[0].content;

  console.log("\n=== search step ===");
  ok(searchCalls.length === 3, `ran 3 search queries, got ${searchCalls.length}`);
  ok(searchCalls.every(c => c.path === "/search"), "posts to Exa /search");
  const queries = searchCalls.map(c => c.body.query);
  ok(queries.every(q => !/\(SE, 4\.5km\)/.test(q)),
    "strips the map bearing suffix from the place name");
  ok(queries.some(q => /Bandar Baru Bangi/.test(q)), "queries name the place");
  ok(queries.some(q => /holiday/i.test(q)), "one query covers public/school holidays");
  ok(queries.some(q => /April|June/.test(q)), "queries carry the quarter's months");

  console.log("\n=== LLM step ===");
  ok(llmCalls.length === 1, `exactly 1 LLM call, got ${llmCalls.length}`);
  ok(llm.body.model === "openai/gpt-oss-120b",
    `uses the plain model, not compound (got ${llm.body.model})`);
  ok(!llm.body.tools && !llm.body.plugins,
    "does NOT ask the model to search — no tools/plugins on the request");
  ok(/SEARCH RESULTS ALREADY RETRIEVED/.test(prompt), "prompt embeds the retrieved results");
  ok(/example-stadium\.com\.my/.test(prompt), "prompt contains the result URLs");
  ok(/no web access of your own/.test(prompt), "prompt tells the model it cannot search");

  console.log("\n=== payload is bounded here, not by the provider ===");
  const approxTokens = Math.round(llm.bytes / 3.8);
  ok(llm.bytes < 20000, `request is ${llm.bytes} bytes (~${approxTokens} tok), well under the ~8.4k free-tier ceiling`);
  // The stub supplies the phrase 40 times (~2,080 chars). Asserting only that
  // it was cut — not by how much — so tuning the cap does not break this.
  const SUPPLIED = 40;
  const repeats = (prompt.match(/A concert is scheduled at the stadium on 14 May 2026\./g) || []).length;
  ok(repeats > 0 && repeats < SUPPLIED,
    `long snippet truncated: ${repeats} of ${SUPPLIED} repeats survived`);

  console.log("\n=== grounding uses OUR retrieved URLs ===");
  const names = brief.contextItems.map(i => i.name);
  ok(names.includes("Grounded Concert"), "keeps the item citing a URL Exa returned");
  ok(!names.includes("HALLUCINATED Festival"), "DROPS the item citing a URL Exa never returned");
  ok(brief.droppedUnsourced === 1, `reports 1 discarded item (got ${brief.droppedUnsourced})`);
  ok(brief.searchProvider === "exa", `records the search provider (got ${brief.searchProvider})`);
  ok(brief.provider === "groq", `records the LLM provider (got ${brief.provider})`);

  console.log(`\n${failures.length ? failures.length + " FAILURE(S):\n" + failures.join("\n") : "ALL PASS"}`);
  process.exit(failures.length ? 1 : 0);
})();
