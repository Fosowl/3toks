# Local SearXNG for the web vertical

`SearxngProvider` (in `threetoks/web/search.py`) scrapes a local SearXNG
instance's HTML results page. This directory runs one via Docker. It is
adapted from `agenticSeek/searxng/`: the upstream `searxng/searxng:latest`
image, with only two settings overrides — `formats: [html]` (so the results
page is scrapable) and `limiter: false` (so local automated queries are not
rate-limited).

## Start

```sh
./start_search_engine.sh
```

From the repo root. It generates a `SEARXNG_SECRET` on first run (saved to
`infra/searxng/.env`, reused on later runs so sessions/cache aren't rotated
every restart), brings the stack up via `docker compose` (Docker daemon must
be running), and polls until the search endpoint responds.

To do it by hand instead:

```sh
# 1. Generate a secret (searxng refuses to start with the placeholder).
export SEARXNG_SECRET="$(openssl rand -hex 32)"

# 2. Bring it up (Docker daemon must be running).
cd infra/searxng
docker compose up -d

# 3. Verify.
curl -s -X POST http://localhost:8080/search \
  -d 'q=hello&categories=general&format=' >/dev/null && echo "searxng up"
```

`auto_provider()` then detects it automatically. Override the base URL with
`THREETOKS_SEARXNG_URL` (defaults to `http://localhost:8080`).

## Stop

```sh
cd infra/searxng
docker compose down
```

## Notes

- `settings.yml` is the full upstream file (~66 KB); the only meaningful
  edits versus stock are `formats: [html]` and `limiter: false`. The
  `secret_key` placeholder is overwritten at runtime by `SEARXNG_SECRET`.
- The `valkey` service is a cache for faster repeat queries; the compose
  file lists it as a dependency of `searxng`, so `docker compose up -d`
  starts both. SearXNG itself works fine without it if you run it
  standalone outside this compose file.
- No API key is needed. If the daemon is not running, the harness falls
  back to `DdgHtmlProvider` with no configuration.
