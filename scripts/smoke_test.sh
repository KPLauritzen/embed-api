#!/usr/bin/env bash
# Start the image, wait for readiness, embed one text. Used by CI.
#   scripts/smoke_test.sh <image>
set -euo pipefail

image=$1
name=embed-smoke
port=8000

cleanup() {
  docker logs "$name" 2>&1 | tail -n 30 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "$name" -p "$port:8000" --read-only --tmpfs /tmp "$image" >/dev/null

for _ in $(seq 120); do
  status=$(curl -s -o /dev/null -w '%{http_code}' "localhost:$port/health/ready" || true)
  [ "$status" = 200 ] && break
  sleep 2
done
[ "$status" = 200 ] || { echo "not ready (last status: $status)"; exit 1; }

dimension=$(curl -sf "localhost:$port/v1/embed" -H 'content-type: application/json' \
  -d '{"input": "Hej verden", "input_type": "query"}' |
  python3 -c 'import json, sys; print(len(json.load(sys.stdin)["embeddings"][0]["embedding"]))')
[ "$dimension" = 1024 ] || { echo "unexpected dimension: $dimension"; exit 1; }

echo "smoke test passed"
