"""Retrieval quality of the int8 model and the fp32 model on Danish MTEB tasks.

    just eval     # = uv run --extra torch --with datasets python scripts/eval_retrieval.py

For each task, embeds the corpus with the "passage: " prefix and the queries with
"query: ", using both backends through the API's own embedder classes, ranks the
corpus by cosine similarity, and reports nDCG@10, Recall@10, and how often the two
backends rank the same document first.

The first nDCG@10 column averages over queries that have a relevant document.
MTEB averages over every query in the relevance judgements, and a query whose
judgements are all "not relevant" scores 0. DanFEVER has 3,271 such queries, so
its MTEB score is about half. The second column uses MTEB's convention and gives
the same fp32 scores as the official mteb package. Both tasks together take
10-15 minutes on a CPU.
"""

import argparse
import math
import time
from collections import defaultdict

import numpy as np
from datasets import load_dataset

from embed_api.config import Backend, Settings
from embed_api.embedder import Embedder, InputType, load_embedder, read_prefixes

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
        [model.embed([prefix + t for t in texts[i : i + 256]]) for i in range(0, len(texts), 256)]
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
    parser.add_argument(
        "--model",
        action="append",
        metavar="BACKEND:PATH",
        help="model to evaluate, repeatable (default: torch:models/e5 and onnx:models/e5-int8); "
        "'Same top-1' compares each model with the first",
    )
    args = parser.parse_args()
    models = []
    for spec in args.model or ["torch:models/e5", "onnx:models/e5-int8"]:
        backend, path = spec.split(":", 1)
        settings = Settings(backend=Backend(backend), model_path=path, num_threads=args.threads)
        models.append((load_embedder(settings), read_prefixes(settings.model_dir)))

    print(
        "| Task | Model | nDCG@10 | nDCG@10, MTEB convention | Recall@10 | Same top-1 | Time |\n"
        "|---|---|---|---|---|---|---|",
        flush=True,
    )
    for repo in TASKS:
        doc_ids, docs, query_ids, queries, relevant, n_judged = load_task(repo)
        first_top1 = None
        for model, prefixes in models:
            start = time.perf_counter()
            corpus = embed(model, docs, prefixes[InputType.PASSAGE])
            q = embed(model, queries, prefixes[InputType.QUERY])
            seconds = time.perf_counter() - start
            scores = q @ corpus.T
            top = np.argpartition(-scores, K, axis=1)[:, :K]
            order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
            top = np.take_along_axis(top, order, axis=1)
            if first_top1 is None:
                first_top1 = top[:, 0]
            m = evaluate(top, doc_ids, query_ids, relevant)
            # MTEB's convention: queries without a relevant document count as 0.
            mteb = m["ndcg@10"] * len(query_ids) / n_judged
            print(
                f"| {repo.split('/')[1]} ({len(query_ids)} queries) "
                f"| {model.model_name.split('/')[-1]} {model.backend} "
                f"| {m['ndcg@10']:.4f} | {mteb:.4f} | {m['recall@10']:.4f} "
                f"| {np.mean(top[:, 0] == first_top1):.1%} | {seconds:.0f} s |",
                flush=True,
            )


if __name__ == "__main__":
    main()
