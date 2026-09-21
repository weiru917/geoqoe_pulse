# Retrieved context cache

One JSON file per H3 cell (`<h3_index>.json`), written by `context_brief.js` after a
successful retrieval. The dashboard serves a cached brief instantly and only calls the
API when a cell has no file here.

## Why this directory exists

Live retrieval takes 30–60 seconds and needs a working network connection. Pre-warming
the cells you intend to show means the demo never waits and never depends on venue wifi.

## Pre-warming before a demo

```bash
cp .env.example .env     # then fill in one API key
node server.js           # the banner confirms which provider is live
# then, for each cell you plan to show:
curl -X POST 'http://localhost:3000/api/context-brief' \
  -H 'Content-Type: application/json' \
  -d '{"h3":"<h3_index>","cellName":"<place name>","lat":<lat>,"lon":<lon>,"forecastQuarter":"2026Q2","cell":{}}'
```

Add `?refresh=1` to the URL to re-search a cell that is already cached.

## Verify before you present

The code guarantees that every retained item carries a source URL from a page the search
actually returned. It does **not** guarantee the source says what the model claims it says.
Open each `source_url` and confirm the event, its date and its location before presenting
these briefs as verified. Delete the file to discard a brief and search again.

Committing the verified files is intentional: it makes the demo reproducible and lets a
reviewer check the same sources you did.
