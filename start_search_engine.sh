#!/usr/bin/env bash
# Bring up the local SearXNG instance used by the ThreeToks web vertical.
# See infra/searxng/README.md for details.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARXNG_DIR="$ROOT_DIR/infra/searxng"
ENV_FILE="$SEARXNG_DIR/.env"

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker is not installed or not on PATH" >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "error: Docker daemon is not running" >&2
    exit 1
fi

# Reuse a previously generated secret so restarts don't rotate sessions/cache.
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

if [ -z "${SEARXNG_SECRET:-}" ]; then
    SEARXNG_SECRET="$(openssl rand -hex 32)"
    echo "SEARXNG_SECRET=$SEARXNG_SECRET" > "$ENV_FILE"
    echo "generated new SEARXNG_SECRET, saved to $ENV_FILE"
fi

export SEARXNG_SECRET

(cd "$SEARXNG_DIR" && docker compose up -d)

echo "waiting for searxng to come up..."
for _ in $(seq 1 15); do
    if curl -s -X POST http://localhost:8080/search \
        -d 'q=hello&categories=general&format=' >/dev/null 2>&1; then
        echo "searxng up"
        exit 0
    fi
    sleep 1
done

echo "error: searxng did not respond in time" >&2
exit 1
