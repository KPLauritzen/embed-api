"""Retrieval quality of the served int8 model against the fp32 original, on MTEB tasks.

    just eval     # = uv run --extra torch --with datasets python scripts/eval_retrieval.py

Embeds each task's corpus ("passage: ") and queries ("query: ") with both
backends, through the same embedder classes the API uses, ranks the corpus by
cosine similarity, and reports nDCG@10 (MTEB's main retrieval metric) and
Recall@10. Also reports how often the two backends agree on the top-1 document.

nDCG@10 is averaged over the queries that have a relevant document. MTEB
averages over every query that has a judgement, counting one whose judgements
are all "not relevant" as 0; DanFEVER has 3,271 such queries, which halves its
MTEB number. Both are reported; the second reproduces the official `mteb`
package. The two
Danish tasks are small enough for a CPU (~10-15 minutes for both backends).
"""

import argparse
import math
from collections import defaultdict

import numpy as np
from datasets import load_dataset

from embed_api.config import Backend, Settings
from embed_api.embedder import Embedder, load_embedder

TASKS = ["mteb/DanFeverRetrieval", "mteb/TwitterHjerneRetrieval"]
K = 10


def load_task(repo: str) -> tuple[list[str], list[str], list[str], list[str], dict, int]:
    corpus = load_dataset(repo, "corpus", split="train")
    queries = load_dataset(repo, "queries", split="train")
    relevant: dict[str, set[str]] = defaultdict(set)
    judged = set()
    for row in load_dataset(repo, "qrels", split="train"):
        judged.add(str(row["query-id"]))
        if row["score"] > 0:
            relevant[str(row["query-id"])].add(str(row["corpus-id"]))
    # MTEB joins title and text for retrieval corpora.
    docs = [
        f"{title} {text}".strip()
        for title, text in zip(corpus["title"], corpus["text"], strict=True)
    ]
    doc_ids = [str(i) for i in corpus["_id"]]
    scored = [
        (str(i), t)
        for i, t in zip(queries["_id"], queries["text"], strict=True)
        if str(i) in relevant
    ]
    query_ids, query_texts = [i for i, _ in scored], [t for _, t in scored]
    return doc_ids, docs, query_ids, query_texts, relevant, len(judged)


def embed(model: Embedder, texts: list[str], prefix: str) -> np.ndarray:
    return np.vstack(
        [
            model.embed([f"{prefix}: {t}" for t in texts[i : i + 256]])
            for i in range(0, len(texts), 256)
        ]
    )


def evaluate(top: np.ndarray, doc_ids: list[str], query_ids: list[str], relevant: dict) -> dict:
    ndcg, recall = [], []
    for qid, ranked in zip(query_ids, top, strict=True):
        rel = relevant[qid]
        gains = [1.0 if doc_ids[d] in rel else 0.0 for d in ranked]
        dcg = sum(g / math.log2(rank + 2) for rank, g in enumerate(gains))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(rel), K)))
        ndcg.append(dcg / ideal)
        recall.append(sum(gains) / len(rel))
    return {"ndcg@10": float(np.mean(ndcg)), "recall@10": float(np.mean(recall))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    models = {
        backend: load_embedder(Settings(backend=backend, num_threads=args.threads))
        for backend in (Backend.TORCH, Backend.ONNX)
    }

    print(
        "| Task | Backend | nDCG@10 | nDCG@10, MTEB convention | Recall@10 | Same top-1 |\n"
        "|---|---|---|---|---|---|",
        flush=True,
    )
    for repo in TASKS:
        doc_ids, docs, query_ids, queries, relevant, n_judged = load_task(repo)
        tops = {}
        for backend, model in models.items():
            corpus, q = embed(model, docs, "passage"), embed(model, queries, "query")
            scores = q @ corpus.T
            top = np.argpartition(-scores, K, axis=1)[:, :K]
            order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
            tops[backend] = np.take_along_axis(top, order, axis=1)
            m = evaluate(tops[backend], doc_ids, query_ids, relevant)
            mteb = m["ndcg@10"] * len(query_ids) / n_judged  # all-irrelevant queries count 0
            agree = np.mean(tops[backend][:, 0] == tops[Backend.TORCH][:, 0])
            print(
                f"| {repo.split('/')[1]} ({len(query_ids)} queries) | {model.backend} "
                f"| {m['ndcg@10']:.4f} | {mteb:.4f} | {m['recall@10']:.4f} | {agree:.1%} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
