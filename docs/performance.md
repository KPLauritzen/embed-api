# Performance

`scripts/benchmark.py` (`just bench`) times each backend through the API's own embedder
classes, with 4 threads and texts of about 30 tokens, and reports medians of 10 runs on an
AMD Ryzen 9 5950X. It also compares each backend's embeddings with PyTorch fp32 on 12
Danish and English query/passage pairs.

| Backend | Weights | 1 text | 8 texts | 32 texts | Texts/s | Cosine vs fp32 (mean / min) | Same top-1 |
|---|---|---|---|---|---|---|---|
| PyTorch fp32 | 2,240 MB | 115 ms | 485 ms | 1,829 ms | 17 | 1.0000 / 1.0000 | 100% |
| ONNX Runtime fp32 | 2,236 MB | 58 ms | 458 ms | 1,816 ms | 18 | 1.0000 / 1.0000 | 100% |
| ONNX Runtime int8 (served) | 562 MB | 33 ms | 262 ms | 1,037 ms | 31 | 0.9951 / 0.9929 | 100% |

For a single text, switching to ONNX Runtime halves the latency and int8 removes another 40%.
For 32 texts, ONNX fp32 is no faster than PyTorch because it also runs one text at a time, so
the 1.8x throughput at that size comes from int8. Twelve pairs are only a sanity check; the
next section measures quality properly.

## Retrieval quality

`scripts/eval_retrieval.py` (`just eval`) runs two Danish retrieval tasks from
[MTEB](https://github.com/embeddings-benchmark/mteb) with both backends. It embeds the corpus
and the queries with the e5 prefixes, ranks the corpus by cosine similarity, and computes
nDCG@10 (MTEB's main retrieval metric) and Recall@10.

| Task (queries / corpus) | Backend | nDCG@10 | nDCG@10, MTEB convention | Recall@10 | Same top-1 as fp32 |
|---|---|---|---|---|---|
| [DanFEVER](https://huggingface.co/datasets/mteb/DanFeverRetrieval) (3,102 / 2,524) | PyTorch fp32 | 0.8396 | 0.4087 | 0.9869 | |
| | ONNX int8 (served) | 0.8276 (-1.4%) | 0.4028 | 0.9815 | 91.5% |
| [TwitterHjerne](https://huggingface.co/datasets/mteb/TwitterHjerneRetrieval) (77 / 262) | PyTorch fp32 | 0.7539 | 0.7539 | 0.8071 | |
| | ONNX int8 (served) | 0.7297 (-3.2%) | 0.7297 | 0.7693 | 83.1% |

int8 loses 1.4% nDCG@10 on DanFEVER and 3.2% on TwitterHjerne. DanFEVER is the better
estimate; TwitterHjerne has only 77 queries. If an application needs that last bit of
quality, it can run the fp32 backend with `EMBED_BACKEND=torch`.

The two nDCG columns average over different queries. The first uses only queries that have a
relevant document. MTEB averages over all queries in the relevance judgements, and 3,271
DanFEVER queries have no relevant document in the corpus, so they score 0 and halve the
result. With MTEB's convention, the fp32 scores are identical to those from the official
`mteb` package, which confirms the evaluation code.
