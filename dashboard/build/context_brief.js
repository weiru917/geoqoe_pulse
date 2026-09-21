// context_brief.js — retrieves public scheduled-event context for one H3 cell and
// selects regulator next-actions from a closed, ITU-referenced catalogue.
//
// Design constraints this file exists to enforce:
//   1. The API key stays server-side. The browser never sees it.
//   2. Every returned context item must carry a source URL that the provider's
//      own web search actually returned. Anything else is dropped, not shown.
//   3. Suggested actions are SELECTED from regulator_actions.json. The model may
//      not invent an action type or an ITU citation.
//   4. Failure and "nothing found" produce the same neutral output, so a network
//      problem at demo time is indistinguishable from a legitimate empty result.
//
// Configuration lives in a .env file next to this one (see .env.example).
// No dependencies beyond Node's standard library.

const https = require("https");
const fs = require("fs");
const path = require("path");

// ---------------------------------------------------------------------------
// .env loading
//
// Minimal KEY=VALUE parser — enough for an API key and a couple of overrides,
// without pulling in dotenv and breaking the zero-dependency property. A real
// environment variable always wins over the file, so CI and shell exports keep
// working unchanged.
// ---------------------------------------------------------------------------

function loadEnvFile() {
  const candidates = [
    path.join(__dirname, ".env"),
    path.join(__dirname, "..", "..", ".env"),  // repository root
  ];
  for (const file of candidates) {
    let raw;
    try { raw = fs.readFileSync(file, "utf8"); } catch { continue; }
    for (const line of raw.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const eq = trimmed.indexOf("=");
      if (eq < 1) continue;
      const key = trimmed.slice(0, eq).trim();
      let value = trimmed.slice(eq + 1).trim();
      if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }
      if (process.env[key] === undefined) process.env[key] = value;
    }
  }
}
loadEnvFile();

// ---------------------------------------------------------------------------
// Providers
//
// Both supported providers run web search server-side and report which pages
// they actually retrieved. That report is what the grounding check in
// groundItems() verifies every returned item against, so swapping providers
// does not weaken the guarantee — a provider WITHOUT retrievable source URLs
// would, which is why there is no generic "any OpenAI-compatible endpoint"
// option here.
// ---------------------------------------------------------------------------

// Recursively collect every source locator a provider reports. Providers differ
// in both nesting and key name, and the exact shape is not worth hard-coding:
//   url    — Anthropic, Groq, OpenRouter
//   uri    — Gemini (often a Google redirect, not the publisher's own address)
//   domain — Gemini, the real publisher host behind that redirect. Collecting it
//            is essential: the model cites the publisher URL, so matching only
//            on redirect links would discard every correctly sourced item.
const SOURCE_KEYS = new Set(["url", "uri", "domain"]);

function collectUrls(node, out, depth = 0) {
  if (!node || depth > 8) return out;
  if (Array.isArray(node)) {
    for (const entry of node) collectUrls(entry, out, depth + 1);
    return out;
  }
  if (typeof node === "object") {
    for (const [key, value] of Object.entries(node)) {
      if (SOURCE_KEYS.has(key) && typeof value === "string") out.push(value);
      else collectUrls(value, out, depth + 1);
    }
  }
  return out;
}

const PROVIDERS = {
  anthropic: {
    label: "Anthropic",
    keyVar: "ANTHROPIC_API_KEY",
    defaultModel: "claude-opus-5",
    plainModel: "claude-opus-5",
    hostname: "api.anthropic.com",
    apiPath: () => "/v1/messages",
    headers: key => ({ "x-api-key": key, "anthropic-version": "2023-06-01" }),
    body: (model, messages, nativeSearch = true) => ({
      model,
      max_tokens: 16000,
      messages,
      ...(nativeSearch ? { tools: [{ type: "web_search_20260209", name: "web_search", max_uses: 6 }] } : {}),
    }),
    // Server tools can return stop_reason "pause_turn"; the turn is continued
    // by echoing the assistant content back.
    paused: json => json?.stop_reason === "pause_turn",
    assistantTurn: json => ({ role: "assistant", content: json.content }),
    refused: json => json?.stop_reason === "refusal",
    text: json => (json?.content || []).filter(b => b.type === "text").map(b => b.text).join("\n"),
    usage: json => ({
      input: (json?.usage?.input_tokens || 0)
        + (json?.usage?.cache_read_input_tokens || 0)
        + (json?.usage?.cache_creation_input_tokens || 0),
      output: json?.usage?.output_tokens || 0,
    }),
    searchUrls: json => {
      const urls = [];
      for (const block of json?.content || []) {
        // A success `content` is a list of results; an error `content` is a
        // single object — branch before walking it.
        if (block.type === "web_search_tool_result" && Array.isArray(block.content)) {
          collectUrls(block.content, urls);
        }
      }
      return urls;
    },
  },

  groq: {
    label: "Groq",
    keyVar: "GROQ_API_KEY",
    // Compound systems run web search server-side and report the pages used in
    // executed_tools[].search_results. A plain Llama model on Groq has no search
    // and would produce ungrounded output, so it is not a valid choice here.
    defaultModel: "groq/compound",
    // With SEARCH_PROVIDER set, compound is the wrong choice: its own search is
    // what 413s. A plain model reads the supplied results instead, and works on
    // the free tier.
    // Verified present on a free Groq account 2026-09-21; the Llama chat models
    // are not. List yours with: GET https://api.groq.com/openai/v1/models
    plainModel: "openai/gpt-oss-120b",
    hostname: "api.groq.com",
    apiPath: () => "/openai/v1/chat/completions",
    headers: key => ({ authorization: `Bearer ${key}` }),
    body: (model, messages) => ({ model, messages, max_completion_tokens: 8000 }),
    // Compound runs its whole agentic loop server-side — there is no pause.
    paused: () => false,
    assistantTurn: json => ({ role: "assistant", content: json.choices?.[0]?.message?.content ?? "" }),
    refused: () => false,
    text: json => json?.choices?.[0]?.message?.content ?? "",
    usage: json => ({
      input: json?.usage?.prompt_tokens || 0,
      output: json?.usage?.completion_tokens || 0,
    }),
    searchUrls: json => collectUrls(json?.choices?.[0]?.message?.executed_tools || [], []),
  },

  perplexity: {
    label: "Perplexity",
    keyVar: "PERPLEXITY_API_KEY",
    // Sonar searches natively and — unlike the other search-capable providers
    // here — filters and compresses the results before they reach the model,
    // so the context it assembles stays bounded. That is what avoids the
    // 413 Request Entity Too Large that Groq's compound systems hit.
    defaultModel: "sonar",
    hostname: "api.perplexity.ai",
    apiPath: () => "/chat/completions",
    headers: key => ({ authorization: `Bearer ${key}` }),
    body: (model, messages) => ({ model, messages, max_tokens: 4000 }),
    paused: () => false,
    assistantTurn: json => ({ role: "assistant", content: json.choices?.[0]?.message?.content ?? "" }),
    refused: () => false,
    text: json => json?.choices?.[0]?.message?.content ?? "",
    usage: json => ({
      input: json?.usage?.prompt_tokens || 0,
      output: json?.usage?.completion_tokens || 0,
    }),
    // Sources sit at the TOP level of the response, not inside choices[].
    // `citations` is a bare array of URL strings, which the key-walker cannot
    // see, so it is concatenated directly; `search_results` holds objects with
    // a `url` field and is walked as usual.
    searchUrls: json => [
      ...(Array.isArray(json?.citations) ? json.citations.filter(u => typeof u === "string") : []),
      ...collectUrls(json?.search_results || [], []),
    ],
  },

  gemini: {
    label: "Google AI Studio",
    keyVar: "GEMINI_API_KEY",
    defaultModel: "gemini-3.5-flash",
    plainModel: "gemini-3.5-flash",  // same model; the google_search tool is simply omitted
    hostname: "generativelanguage.googleapis.com",
    apiPath: model => `/v1beta/models/${encodeURIComponent(model)}:generateContent`,
    headers: key => ({ "x-goog-api-key": key }),
    body: (model, messages, nativeSearch = true) => ({
      contents: messages.map(m => ({
        role: m.role === "assistant" ? "model" : "user",
        parts: [{ text: typeof m.content === "string" ? m.content : JSON.stringify(m.content) }],
      })),
      ...(nativeSearch ? { tools: [{ google_search: {} }] } : {}),
      generationConfig: { maxOutputTokens: 8192 },
    }),
    paused: () => false,
    assistantTurn: json => ({ role: "assistant", content: PROVIDERS.gemini.text(json) }),
    refused: json => json?.candidates?.[0]?.finishReason === "SAFETY",
    text: json => (json?.candidates?.[0]?.content?.parts || []).filter(part => part.text).map(part => part.text).join("\n"),
    usage: json => ({
      input: json?.usageMetadata?.promptTokenCount || 0,
      output: json?.usageMetadata?.candidatesTokenCount || 0,
    }),
    searchUrls: json => collectUrls(json?.candidates?.[0]?.groundingMetadata?.groundingChunks || [], []),
  },

  openrouter: {
    label: "OpenRouter",
    keyVar: "OPENROUTER_API_KEY",
    defaultModel: "google/gemini-3.5-flash",
    hostname: "openrouter.ai",
    apiPath: () => "/api/v1/chat/completions",
    headers: key => ({ authorization: `Bearer ${key}` }),
    // The `web` plugin adds search to any model and returns standard OpenAI
    // url_citation annotations. OpenRouter also exposes a newer
    // `openrouter:web_search` server tool; if this form is ever withdrawn the
    // request fails loudly with a 400 rather than silently returning no sources.
    body: (model, messages, nativeSearch = true) => ({
      model,
      messages,
      max_tokens: 8000,
      ...(nativeSearch ? { plugins: [{ id: "web", max_results: 5 }] } : {}),
    }),
    paused: () => false,
    assistantTurn: json => ({ role: "assistant", content: json.choices?.[0]?.message?.content ?? "" }),
    refused: () => false,
    text: json => json?.choices?.[0]?.message?.content ?? "",
    usage: json => ({
      input: json?.usage?.prompt_tokens || 0,
      output: json?.usage?.completion_tokens || 0,
    }),
    searchUrls: json => collectUrls(json?.choices?.[0]?.message?.annotations || [], []),
  },
};

// ---------------------------------------------------------------------------
// Separate search step (optional)
//
// Set SEARCH_PROVIDER and the model no longer searches for itself: this file
// runs the queries, truncates the snippets, and hands the model a bounded list
// of results to read. Three things follow from that, and they are the whole
// reason this path exists:
//
//   1. It is free. Bundled server-side search is the part every LLM free tier
//      withholds (Groq compound 413s; Gemini grounding is zero-quota), while
//      dedicated search APIs and plain chat models both have real free tiers.
//   2. The payload is bounded HERE, not by the provider. Snippets are capped
//      and the result list is capped, so the request cannot balloon past a
//      per-request token limit the way compound's internal context does.
//   3. Grounding gets stronger. The URL allowlist is the exact set this file
//      received, rather than whatever the provider chose to report afterwards.
// ---------------------------------------------------------------------------

const SEARCH_PROVIDERS = {
  exa: {
    label: "Exa",
    keyVar: "EXA_API_KEY",
    hostname: "api.exa.ai",
    apiPath: "/search",
    headers: key => ({ "x-api-key": key }),
    body: (query, n, chars) => ({
      query,
      numResults: n,
      type: "auto",
      contents: {
        // `text` alone returns the TOP of the page, which on event and ticket
        // sites is navigation chrome and template placeholders — the dates sit
        // far below it and get cut off. `highlights` returns the sentences that
        // actually match the query, which is what the model needs to read.
        highlights: { numSentences: 4, highlightsPerUrl: 3, query },
        text: { maxCharacters: chars },
      },
    }),
    results: json => (json?.results || []).map(r => ({
      url: r.url,
      title: r.title,
      // Highlights first; page text is only a fallback when there are none.
      snippet: (r.highlights || []).join(" … ") || r.text || r.summary || "",
    })),
  },

  serper: {
    label: "Serper",
    keyVar: "SERPER_API_KEY",
    hostname: "google.serper.dev",
    apiPath: "/search",
    headers: key => ({ "x-api-key": key }),
    body: (query, n) => ({ q: query, gl: "my", hl: "en", num: n }),
    results: json => (json?.organic || []).map(r => ({
      url: r.link, title: r.title, snippet: r.snippet || "",
    })),
  },

  tavily: {
    label: "Tavily",
    keyVar: "TAVILY_API_KEY",
    hostname: "api.tavily.com",
    apiPath: "/search",
    headers: key => ({ authorization: `Bearer ${key}` }),
    body: (query, n) => ({ query, max_results: n, search_depth: "basic" }),
    results: json => (json?.results || []).map(r => ({
      url: r.url, title: r.title, snippet: r.content || "",
    })),
  },
};

const SEARCH_NAME = (process.env.SEARCH_PROVIDER || "").trim().toLowerCase();
const SEARCH = SEARCH_PROVIDERS[SEARCH_NAME] || null;

// Caps that keep one brief's request small enough for any free tier. Twelve
// results at 350 characters is roughly 1k tokens of evidence on top of a ~1k
// token prompt — against the ~8.4k that a free Groq account accepts.
// Sized against the ~8.4k-token request ceiling measured on a free Groq
// account: 10 results x 900 characters is roughly 2.4k tokens of evidence on
// top of a ~1k token prompt. 350 characters was too mean — page chrome ate the
// whole budget before reaching any dates.
const SEARCH_RESULTS_PER_QUERY = 6;
const SEARCH_MAX_RESULTS = 10;
const SEARCH_SNIPPET_CHARS = 900;

function postSearch(query, apiKey) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify(SEARCH.body(query, SEARCH_RESULTS_PER_QUERY, SEARCH_SNIPPET_CHARS));
    const req = https.request({
      hostname: SEARCH.hostname,
      path: SEARCH.apiPath,
      method: "POST",
      headers: {
        "content-type": "application/json",
        "content-length": Buffer.byteLength(payload),
        ...SEARCH.headers(apiKey),
      },
      timeout: 30000,
    }, res => {
      const chunks = [];
      res.on("data", c => chunks.push(c));
      res.on("end", () => {
        const text = Buffer.concat(chunks).toString("utf8");
        let parsed;
        try { parsed = JSON.parse(text); }
        catch { reject(new Error(`Search: non-JSON response (HTTP ${res.statusCode})`)); return; }
        if (res.statusCode >= 400) {
          const err = new Error(`Search HTTP ${res.statusCode}: ${parsed?.error?.message || parsed?.message || text.slice(0, 150)}`);
          err.status = res.statusCode;
          reject(err); return;
        }
        resolve(SEARCH.results(parsed));
      });
    });
    req.on("timeout", () => req.destroy(new Error("Search timed out after 30000 ms")));
    req.on("error", reject);
    req.end(payload);
  });
}

// Place names carry a bearing/distance suffix for the map ("Bangi (SE, 4.5km)").
// That is meaningless to a search engine and actively harms recall.
function cleanPlaceName(name) {
  return String(name || "").replace(/\s*\([^)]*\)\s*$/, "").trim();
}

const MONTHS = ["January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December"];

function searchQueries(input) {
  const place = cleanPlaceName(input.cellName);
  const win = quarterWindow(input.forecastQuarter);
  const year = win ? win.start.slice(0, 4) : "";
  const months = win
    ? `${MONTHS[Number(win.start.slice(5, 7)) - 1]} ${MONTHS[Number(win.end.slice(5, 7)) - 1]}`
    : "";
  return [
    `events concerts festivals ${place} Selangor Malaysia ${months} ${year}`,
    `${place} stadium arena convention centre exhibition schedule ${year}`,
    `Malaysia public holidays and Selangor school holidays ${months} ${year}`,
  ].map(q => q.replace(/\s+/g, " ").trim());
}

// Sequential, never concurrent: one query at a time keeps this inside
// per-minute free-tier limits and makes a partial failure survivable — some
// results are still better than none.
async function runSearch(input, apiKey, h3) {
  const byUrl = new Map();
  for (const query of searchQueries(input)) {
    try {
      for (const r of await postSearch(query, apiKey)) {
        if (!r?.url || byUrl.has(r.url)) continue;
        byUrl.set(r.url, {
          url: r.url,
          title: String(r.title || "").slice(0, 200),
          snippet: String(r.snippet || "").replace(/\s+/g, " ").slice(0, SEARCH_SNIPPET_CHARS),
        });
      }
    } catch (err) {
      console.warn(`[context-brief] ${h3} search query failed: ${err.message}`);
    }
  }
  const results = [...byUrl.values()].slice(0, SEARCH_MAX_RESULTS);
  console.log(`[context-brief] ${h3} · ${SEARCH.label} · ${results.length} unique result(s) from ${searchQueries(input).length} queries`);
  return results;
}

const PROVIDER_NAME = (process.env.CONTEXT_PROVIDER || "anthropic").trim().toLowerCase();
const PROVIDER = PROVIDERS[PROVIDER_NAME];
const MODEL = process.env.CONTEXT_MODEL
  || (SEARCH ? (PROVIDER?.plainModel || PROVIDER?.defaultModel) : PROVIDER?.defaultModel);
const TIMEOUT_MS = Number(process.env.CONTEXT_TIMEOUT_MS || 60000);
const MAX_PAUSE_TURNS = 4;

const CACHE_DIR = path.join(__dirname, "data", "context");
const CATALOGUE_PATH = path.join(__dirname, "regulator_actions.json");

// Reported by server.js at startup so the operator knows what is configured.
function providerStatus() {
  if (!PROVIDER) {
    return { ok: false, detail: `CONTEXT_PROVIDER="${PROVIDER_NAME}" is not recognised (expected: ${Object.keys(PROVIDERS).join(", ")})` };
  }
  if (!process.env[PROVIDER.keyVar]) {
    return { ok: false, detail: `${PROVIDER.label} selected but ${PROVIDER.keyVar} is not set` };
  }
  if (SEARCH_NAME && !SEARCH) {
    return { ok: false, detail: `SEARCH_PROVIDER="${SEARCH_NAME}" is not recognised (expected: ${Object.keys(SEARCH_PROVIDERS).join(", ")})` };
  }
  if (SEARCH && !process.env[SEARCH.keyVar]) {
    return { ok: false, detail: `${SEARCH.label} search selected but ${SEARCH.keyVar} is not set` };
  }
  return {
    ok: true,
    detail: SEARCH
      ? `${SEARCH.label} search + ${PROVIDER.label} ${MODEL}`
      : `${PROVIDER.label} · ${MODEL} (provider-native search)`,
  };
}

// ---------------------------------------------------------------------------
// Action catalogue
// ---------------------------------------------------------------------------

function loadCatalogue() {
  const raw = JSON.parse(fs.readFileSync(CATALOGUE_PATH, "utf8"));
  return {
    maxActions: Number(raw.max_actions) || 3,
    actions: Array.isArray(raw.actions) ? raw.actions : [],
  };
}

// ---------------------------------------------------------------------------
// Quarter helpers
// ---------------------------------------------------------------------------

// "2026Q2" -> { start: "2026-04-01", end: "2026-06-30" }
function quarterWindow(quarter) {
  const match = /^(\d{4})Q([1-4])$/.exec(String(quarter || "").trim());
  if (!match) return null;
  const year = Number(match[1]);
  const q = Number(match[2]);
  const startMonth = (q - 1) * 3 + 1;
  const endMonth = startMonth + 2;
  const endDay = new Date(Date.UTC(year, endMonth, 0)).getUTCDate();
  const pad = n => String(n).padStart(2, "0");
  return { start: `${year}-${pad(startMonth)}-01`, end: `${year}-${pad(endMonth)}-${pad(endDay)}` };
}

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------

function describeCell(cell) {
  const line = (label, value) => value == null || value === "" ? null : `- ${label}: ${value}`;
  const num = (value, digits = 1) => value == null ? null : Number(value).toFixed(digits);
  return [
    line("Observed status in the predictor quarter", cell.observedStatus),
    line("AI emerging-risk band", cell.riskBand),
    line("AI rank", cell.riskRank == null ? null : `#${cell.riskRank} of ${cell.eligibleCells ?? "the evaluable cells"}`),
    line("Recommended triage pathway", cell.triageLabel),
    line("Quarterly download", num(cell.downloadMbps) && `${num(cell.downloadMbps)} Mbps`),
    line("Quarterly upload", num(cell.uploadMbps) && `${num(cell.uploadMbps)} Mbps`),
    line("Quarterly latency", num(cell.latencyMs) && `${num(cell.latencyMs)} ms`),
    line("Measurement support", cell.measurementSupport),
    line("Day population in cell", cell.populationDay == null ? null : Math.round(cell.populationDay).toLocaleString("en-GB")),
    line("Population density", num(cell.populationDensityPerKm2) && `${num(cell.populationDensityPerKm2)} people/km²`),
    line("Tower density", num(cell.towerDensityPerKm2, 2) && `${num(cell.towerDensityPerKm2, 2)} sites/km²`),
    line("Built-up ratio", cell.builtUpRatio == null ? null : `${(cell.builtUpRatio * 100).toFixed(1)}%`),
    line("Top model contributor", cell.topContributor),
  ].filter(Boolean).join("\n");
}

function buildPrompt(input) {
  const window = quarterWindow(input.forecastQuarter) || { start: "the forecast quarter start", end: "the forecast quarter end" };
  return `You are retrieving public scheduled-event context for a telecommunications regulator in Malaysia.

LOCATION
Area name: ${input.cellName}
Coordinates: ${input.lat}, ${input.lon}
Region: Klang Valley, Malaysia
H3 cell: ${input.h3}

FORECAST PERIOD
${input.forecastQuarter} — ${window.start} to ${window.end}

WHAT THE REGULATOR'S MODEL ALREADY SAYS ABOUT THIS AREA
${describeCell(input.cell) || "- No additional evidence was supplied."}

${input.searchResults ? `SEARCH RESULTS ALREADY RETRIEVED FOR THIS AREA
These are the only sources available to you. You have no web access of your own.

${input.searchResults.map((r, i) => `[${i + 1}] ${r.title}
    URL: ${r.url}
    ${r.snippet}`).join("\n\n")}

TASK
Read the search results above and extract scheduled, publicly announced events` : `TASK
Search the web for scheduled, publicly announced events`} within roughly 3 km of the
coordinates that fall inside the forecast period and could concentrate mobile data
demand in this area:
- stadium, arena, convention centre or exhibition events
- festivals, concerts, public gatherings, markets
- Malaysian public holidays, state holidays for Selangor and Kuala Lumpur, and school holidays
- major transport, sporting or tourism events

RULES — these are absolute
${input.searchResults ? `- Every item you return MUST have a source_url copied EXACTLY from the numbered
  results above. Do not shorten, rewrite, or construct a URL.
- Use ONLY those results. You have no other sources. If they do not mention an
  event, it does not exist for this task.` : `- Every item you return MUST have a source_url that appeared in your web search results.
- If you cannot verify an item from a search result you actually retrieved, omit it.`}
- Prefer official venue, organiser, or government sources.
- Never infer, estimate, or reconstruct an event you did not find in a search result.
- Do not state or imply that any event caused, or will cause, the model's risk ranking.
  The ranking was produced before this search and is independent of it.
- If nothing verifiable is found, that is a correct and acceptable answer.

OUTPUT
Reply with a single JSON object in a \`\`\`json fenced code block and nothing else after it:

{
  "no_context_found": false,
  "context_items": [
    {
      "name": "",
      "date": "",
      "venue": "",
      "est_attendance": null,
      "distance_km": null,
      "category": "event | holiday | transport | other",
      "source_name": "",
      "source_url": "",
      "confidence": "high | medium"
    }
  ],
  "summary": "One or two neutral sentences describing what was found. No causal language.",
  "selected_action_ids": []
}

"selected_action_ids" must contain between 1 and ${input.maxActions} ids chosen ONLY from
the catalogue below, ordered most relevant first. You may not invent an action.

ACTION CATALOGUE
${input.catalogue.map(a => `${a.id} — [${a.type}] ${a.title}\n  When it applies: ${a.applies_when}`).join("\n")}

If "no_context_found" is true, return an empty "context_items" array and select only
"routine_quarterly_review".`;
}

// ---------------------------------------------------------------------------
// API transport
// ---------------------------------------------------------------------------

function postCompletion(body, apiKey) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify(body);
    const req = https.request({
      hostname: PROVIDER.hostname,
      path: PROVIDER.apiPath(MODEL),
      method: "POST",
      headers: {
        "content-type": "application/json",
        "content-length": Buffer.byteLength(payload),
        ...PROVIDER.headers(apiKey),
      },
      timeout: TIMEOUT_MS,
    }, res => {
      const chunks = [];
      res.on("data", chunk => chunks.push(chunk));
      res.on("end", () => {
        const text = Buffer.concat(chunks).toString("utf8");
        let parsed;
        try { parsed = JSON.parse(text); }
        catch { reject(new Error(`Non-JSON response (HTTP ${res.statusCode})`)); return; }
        if (res.statusCode >= 400) {
          const err = new Error(`HTTP ${res.statusCode}: ${parsed?.error?.message || text.slice(0, 200)}`);
          err.status = res.statusCode;
          const after = Number(res.headers["retry-after"]);
          err.retryAfterMs = Number.isFinite(after) && after > 0 ? after * 1000 : null;
          reject(err);
          return;
        }
        resolve(parsed);
      });
    });
    req.on("timeout", () => { req.destroy(new Error(`Timed out after ${TIMEOUT_MS} ms`)); });
    req.on("error", reject);
    req.end(payload);
  });
}

// Retrying a permanent error wastes time and money and, worse, makes a hard
// configuration failure look like a slow network. Only genuinely transient
// conditions are retried; 400/401/403/404/413 fail immediately.
const RETRYABLE = new Set([429, 500, 502, 503, 504]);
const MAX_ATTEMPTS = 3;

function isRetryable(err) {
  if (err.status == null) return /Timed out|ECONNRESET|ETIMEDOUT|ENOTFOUND|socket hang up/i.test(err.message);
  return RETRYABLE.has(err.status);
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function postWithRetry(body, apiKey, h3) {
  let lastError;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      return await postCompletion(body, apiKey);
    } catch (err) {
      lastError = err;
      if (!isRetryable(err)) throw err;
      // A 429 carrying no retry-after generally means the quota is zero rather
      // than momentarily spent — it will not recover on this timescale, so stop
      // after one confirming retry instead of burning the full backoff.
      const quotaExhausted = err.status === 429 && !err.retryAfterMs;
      const cap = quotaExhausted ? 2 : MAX_ATTEMPTS;
      if (attempt >= cap) break;
      const waitMs = err.retryAfterMs ?? 2000 * Math.pow(2, attempt - 1);
      console.warn(`[context-brief] ${h3} attempt ${attempt}/${cap} failed (${err.status ?? "network"}); retrying in ${waitMs} ms`);
      await sleep(waitMs);
    }
  }
  throw lastError;
}

// Some providers pause a long server-tool turn and expect it to be continued by
// echoing the assistant content back; others run the whole loop internally.
async function runWithSearch(prompt, apiKey, h3) {
  const messages = [{ role: "user", content: prompt }];
  const spend = { turns: 0, inputTokens: 0, outputTokens: 0 };
  let response = null;
  for (let turn = 0; turn < MAX_PAUSE_TURNS; turn++) {
    response = await postWithRetry(PROVIDER.body(MODEL, messages, !SEARCH), apiKey, h3);
    spend.turns++;
    // A paused turn is resent with the whole conversation so far, search results
    // included, so input tokens compound across continuations. Accumulate every
    // turn rather than reading only the final response.
    const usage = PROVIDER.usage(response);
    spend.inputTokens += usage.input;
    spend.outputTokens += usage.output;
    if (!PROVIDER.paused(response)) break;
    messages.push(PROVIDER.assistantTurn(response));
  }
  console.log(`[context-brief] ${h3} · ${PROVIDER.label} ${MODEL} · ${spend.turns} turn(s) · `
    + `${spend.inputTokens.toLocaleString()} in / ${spend.outputTokens.toLocaleString()} out tokens`);
  return { response, spend };
}

// ---------------------------------------------------------------------------
// Response parsing and grounding checks
// ---------------------------------------------------------------------------

function hostOf(url) {
  try { return new URL(url).hostname.replace(/^www\./i, "").toLowerCase(); }
  catch { return null; }
}

// Every URL the provider's web search actually retrieved. This set is the whole
// basis of the grounding check — if it comes back empty, nothing is displayed.
function searchedSources(json) {
  const urls = new Set();
  const hosts = new Set();
  const locators = json?.__ourResults
    ? json.__ourResults.map(r => r.url)
    : PROVIDER.searchUrls(json) || [];
  for (const entry of locators) {
    if (typeof entry !== "string" || !entry.trim()) continue;
    urls.add(entry);
    const host = hostOf(entry);
    if (host) { hosts.add(host); continue; }
    // Not a full URL — a bare publisher host, which Gemini reports alongside
    // its redirect links. Normalise and keep it as a matchable host.
    const bare = entry.trim().toLowerCase().replace(/^www\./, "");
    if (/^[a-z0-9.-]+\.[a-z]{2,}$/.test(bare)) hosts.add(bare);
  }
  return { urls, hosts };
}

function extractJson(json) {
  const text = PROVIDER.text(json) || "";
  const fenced = /```json\s*([\s\S]*?)```/i.exec(text) || /```\s*([\s\S]*?)```/.exec(text);
  const candidate = fenced ? fenced[1] : text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1);
  try { return JSON.parse(candidate); } catch { return null; }
}

// Drop any item that is not anchored to a host the search actually returned.
function groundItems(items, sources) {
  if (!Array.isArray(items)) return { kept: [], dropped: 0 };
  let dropped = 0;
  const kept = [];
  for (const item of items) {
    const url = typeof item?.source_url === "string" ? item.source_url.trim() : "";
    const host = hostOf(url);
    const grounded = url && host && (sources.urls.has(url) || sources.hosts.has(host));
    if (!grounded || !item.name || !item.date) { dropped++; continue; }
    kept.push({
      name: String(item.name),
      date: String(item.date),
      venue: item.venue ? String(item.venue) : "",
      estAttendance: Number.isFinite(Number(item.est_attendance)) ? Number(item.est_attendance) : null,
      distanceKm: Number.isFinite(Number(item.distance_km)) ? Number(item.distance_km) : null,
      category: ["event", "holiday", "transport", "other"].includes(item.category) ? item.category : "other",
      sourceName: item.source_name ? String(item.source_name) : host,
      sourceUrl: url,
      confidence: item.confidence === "high" ? "high" : "medium",
    });
  }
  return { kept, dropped };
}

function selectActions(ids, catalogue, maxActions, hasContext) {
  const byId = new Map(catalogue.map(a => [a.id, a]));
  const chosen = [];
  for (const id of Array.isArray(ids) ? ids : []) {
    const action = byId.get(id);
    if (action && !chosen.some(a => a.id === action.id)) chosen.push(action);
    if (chosen.length >= maxActions) break;
  }
  if (!chosen.length) {
    const fallbackId = hasContext ? "targeted_measurement" : "routine_quarterly_review";
    const fallback = byId.get(fallbackId) || catalogue[catalogue.length - 1];
    if (fallback) chosen.push(fallback);
  }
  return chosen.map(a => ({
    id: a.id, type: a.type, title: a.title, detail: a.detail, ituReference: a.itu_reference,
  }));
}

// ---------------------------------------------------------------------------
// Disk cache
// ---------------------------------------------------------------------------

function cachePath(h3) {
  if (!/^[0-9a-f]{15,16}$/i.test(String(h3))) return null;  // also blocks path traversal
  return path.join(CACHE_DIR, `${h3}.json`);
}

function readCache(h3) {
  const file = cachePath(h3);
  if (!file) return null;
  try { return JSON.parse(fs.readFileSync(file, "utf8")); } catch { return null; }
}

function writeCache(h3, record) {
  const file = cachePath(h3);
  if (!file) return;
  try {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
    fs.writeFileSync(file, JSON.stringify(record, null, 2), "utf8");
  } catch (err) {
    console.warn(`[context-brief] could not cache ${h3}: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

// Turns a provider error into an operator-actionable hint. Console only — none
// of this reaches the browser.
function diagnoseFailure(message) {
  const text = String(message || "");
  if (/HTTP 413/.test(text)) {
    return "HINT: the provider rejected the request as too large. On Groq this is an account "
      + "per-request token cap — compound's web-search results exceed it before the reply is "
      + "returned, so it fails even for a one-line prompt. Raise the account's limits or switch "
      + "CONTEXT_PROVIDER in .env. It is not caused by the prompt.";
  }
  if (/HTTP 429/.test(text)) return "HINT: rate limited. Wait and retry, or raise the account's limits.";
  if (/HTTP 503|over capacity/i.test(text)) return "HINT: the provider is over capacity. Transient — retry shortly.";
  if (/HTTP 401|HTTP 403/.test(text)) return `HINT: the ${PROVIDER.label} key was rejected. Check ${PROVIDER.keyVar} in .env.`;
  if (/HTTP 404/.test(text) && /model/i.test(text)) {
    return `HINT: "${MODEL}" is not available on this account. Set CONTEXT_MODEL in .env to one that is `
      + `(Groq: GET https://api.groq.com/openai/v1/models — model availability differs per account).`;
  }
  if (/HTTP 404/.test(text)) return "HINT: endpoint not found — check the provider's base path.";
  if (/Timed out/.test(text)) return `HINT: exceeded CONTEXT_TIMEOUT_MS (${TIMEOUT_MS} ms). Search plus reasoning can take 30-60s.`;
  return "HINT: check network reachability to the provider.";
}

// Shared shape for "nothing corroborated", used for empty results AND for every
// failure path, so a demo-time network problem looks identical to a real null.
function emptyBrief(h3, forecastQuarter, catalogue, maxActions, note) {
  return {
    h3,
    forecastQuarter,
    status: "no_context",
    retrievedAt: new Date().toISOString(),
    source: "none",
    contextItems: [],
    droppedUnsourced: 0,
    summary: "No corroborated upcoming context found for this area and period.",
    actions: selectActions(["routine_quarterly_review"], catalogue, maxActions, false),
    note: note || null,
  };
}

async function buildContextBrief(input, options = {}) {
  const { maxActions, actions: catalogue } = loadCatalogue();
  const h3 = String(input.h3 || "");
  const forecastQuarter = input.forecastQuarter || "";

  if (!options.refresh) {
    const cached = readCache(h3);
    if (cached) return { ...cached, source: "cache" };
  }

  const status = providerStatus();
  if (!status.ok) {
    return emptyBrief(h3, forecastQuarter, catalogue, maxActions,
      `Live retrieval is not configured on this server (${status.detail}).`);
  }
  const apiKey = process.env[PROVIDER.keyVar];

  // With a separate search provider, the evidence is gathered here and handed
  // to the model. The URLs collected now — not anything the model reports back
  // — become the allowlist that groundItems() checks every item against.
  let searchResults = null;
  if (SEARCH) {
    try {
      searchResults = await runSearch(input, process.env[SEARCH.keyVar], h3);
    } catch (err) {
      console.warn(`[context-brief] search failed for ${h3}: ${err.message}`);
      console.warn(`               ${diagnoseFailure(err.message)}`);
      return emptyBrief(h3, forecastQuarter, catalogue, maxActions, "Context retrieval was unavailable.");
    }
    if (!searchResults.length) {
      // No sources means nothing could be corroborated even if the model
      // produced a confident list, so do not spend a call finding that out.
      const brief = emptyBrief(h3, forecastQuarter, catalogue, maxActions, null);
      brief.source = "live";
      writeCache(h3, brief);
      return brief;
    }
  }

  let response, spend;
  try {
    ({ response, spend } = await runWithSearch(
      buildPrompt({ ...input, catalogue, maxActions, searchResults }), apiKey, h3));
  } catch (err) {
    // The brief shown to the user stays neutral whatever went wrong, but the
    // operator needs to tell a provider limit apart from a network blip —
    // otherwise an always-empty panel looks like "there are genuinely no
    // events here" and nobody investigates.
    console.warn(`[context-brief] retrieval failed for ${h3}: ${err.message}`);
    console.warn(`               ${diagnoseFailure(err.message)}`);
    return emptyBrief(h3, forecastQuarter, catalogue, maxActions, "Context retrieval was unavailable.");
  }

  if (PROVIDER.refused(response)) {
    return emptyBrief(h3, forecastQuarter, catalogue, maxActions, "Context retrieval returned no usable result.");
  }

  const parsed = extractJson(response);
  if (!parsed) {
    return emptyBrief(h3, forecastQuarter, catalogue, maxActions, "Context retrieval returned no usable result.");
  }

  // When we ran the search, the allowlist is exactly what we retrieved. That is
  // strictly stronger than trusting the provider to report its own sources.
  const sources = searchResults
    ? searchedSources({ __ourResults: searchResults })
    : searchedSources(response);
  const { kept, dropped } = groundItems(parsed.context_items, sources);
  const hasContext = kept.length > 0 && parsed.no_context_found !== true;

  if (!hasContext) {
    const brief = emptyBrief(h3, forecastQuarter, catalogue, maxActions,
      dropped ? `${dropped} candidate item(s) were discarded because no retrieved source corroborated them.` : null);
    brief.source = "live";
    brief.droppedUnsourced = dropped;
    writeCache(h3, brief);
    return brief;
  }

  const brief = {
    h3,
    forecastQuarter,
    status: "context_found",
    retrievedAt: new Date().toISOString(),
    source: "live",
    provider: PROVIDER_NAME,
    searchProvider: SEARCH ? SEARCH_NAME : null,
    model: MODEL,
    spend,
    contextItems: kept,
    droppedUnsourced: dropped,
    summary: typeof parsed.summary === "string" ? parsed.summary : "",
    actions: selectActions(parsed.selected_action_ids, catalogue, maxActions, true),
    note: null,
  };
  writeCache(h3, brief);
  return brief;
}

module.exports = { buildContextBrief, quarterWindow, providerStatus };
