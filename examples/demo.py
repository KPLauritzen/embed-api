"""Cross-lingual similarity demo against a running embed-api.

    uv run python examples/demo.py [http://localhost:8000]

Embeds a few Danish and English sentences and prints their cosine similarity
matrix: translations land close together, unrelated topics do not.
"""

import json
import sys
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

SENTENCES = [
    ("da", "Katten sover på sofaen."),
    ("en", "The cat is sleeping on the couch."),
    ("da", "Renten steg med et halvt procentpoint."),
    ("en", "Interest rates rose by half a percentage point."),
    ("de", "Die Katze schläft auf dem Sofa."),
]


def embed(texts: list[str]) -> tuple[list[list[float]], str]:
    request = urllib.request.Request(
        f"{BASE_URL}/v1/embed",
        data=json.dumps({"input": texts, "input_type": "query"}).encode(),
        headers={"content-type": "application/json", "X-Request-ID": "demo-similarity"},
    )
    with urllib.request.urlopen(request) as response:
        body = json.load(response)
        return [e["embedding"] for e in body["embeddings"]], response.headers["X-Request-ID"]


def main() -> None:
    vectors, request_id = embed([text for _, text in SENTENCES])
    print(f"request id: {request_id}\n")
    for i, (lang, text) in enumerate(SENTENCES):
        print(f"  [{i}] {lang}  {text}")
    print("\n      " + "".join(f"  [{j}]  " for j in range(len(SENTENCES))))
    for i, a in enumerate(vectors):
        # Vectors are L2-normalised, so the dot product is the cosine similarity.
        row = "".join(f"  {sum(x * y for x, y in zip(a, b, strict=True)):.2f} " for b in vectors)
        print(f"  [{i}] {row}")


if __name__ == "__main__":
    main()
