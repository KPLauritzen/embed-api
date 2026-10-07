"""Truncation and the token budget, against a running embed-api.

    uv run --no-dev python examples/limits.py [http://localhost:8000]

Sends one over-long text (embedded, flagged `truncated`) and one request over
the per-request token budget (rejected with 422 before any inference).
"""

import json
import sys
import urllib.error
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def post(payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{BASE_URL}/v1/embed",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def main() -> None:
    long_text = "Dette er en lang tekst. " * 300  # ~1800 tokens, over the 512 limit
    status, body = post({"input": [long_text, "En kort tekst."], "input_type": "passage"})
    print(f"over-long input -> {status}")
    for item in body["embeddings"]:
        print(f"  [{item['index']}] tokens={item['tokens']} truncated={item['truncated']}")

    status, body = post({"input": [long_text] * 20, "input_type": "passage"})
    print(f"\n20 x 512 tokens -> {status}")
    print(f"  {body['error']['code']}: {body['error']['message']}")


if __name__ == "__main__":
    main()
