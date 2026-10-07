# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sentence-transformers[onnx]", "torch"]
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Compare PyTorch fp32, ONNX fp32 and ONNX int8 for speed and fidelity.

    uv run scripts/benchmark.py --model models/e5 --threads 4

Runs as a standalone script with its own dependencies: the ONNX tooling
(optimum) pins an older transformers, which should not leak into the API's
lockfile. The int8 model is quantised for AVX2, the instruction set of the
deployment node; the int8 file shipped on the Hub targets AVX-512 VNNI.

Fidelity is measured against PyTorch fp32 on a small Danish/English retrieval
set: per-text cosine similarity, and whether each query's top-ranked passage
is unchanged.
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, export_dynamic_quantized_onnx_model

# (query, relevant passage) pairs; every other passage is a distractor.
PAIRS = [
    ("hvad er hovedstaden i Danmark?", "København er Danmarks hovedstad og største by."),
    ("how tall is the Round Tower?", "Rundetårn i København er 34,8 meter højt."),
    ("hvornår blev Tivoli åbnet?", "Tivoli Gardens opened to the public in August 1843."),
    ("what do pandas eat?", "Kæmpepandaen lever næsten udelukkende af bambus."),
    ("hvordan fungerer en varmepumpe?", "A heat pump moves heat from a cold space to a warm one."),
    ("best time to visit Skagen", "Skagen er mest besøgt om sommeren på grund af lyset."),
    ("hvad koster et månedskort i Aarhus?", "A monthly bus pass in Aarhus costs about 400 DKK."),
    ("symptoms of vitamin D deficiency", "Mangel på D-vitamin kan give træthed og muskelsmerter."),
    ("hvem skrev Den grimme ælling?", "The Ugly Duckling was written by Hans Christian Andersen."),
    ("how to reset a router", "Hold reset-knappen på routeren inde i 10 sekunder."),
    ("hvorfor er himlen blå?", "Sunlight is scattered by air molecules, blue light the most."),
    ("rules of handball", "I håndbold må en spiller højst tage tre skridt med bolden."),
]

SENTENCE = (
    "Dette er en typisk sætning på omkring tyve ord, som bruges til at måle, "
    "hvor hurtigt modellen svarer på en almindelig CPU."
)


def encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)


def latency(model: SentenceTransformer, batch: int, repeats: int) -> tuple[float, float]:
    texts = [f"passage: {SENTENCE}"] * batch
    encode(model, texts)  # warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        encode(model, texts)
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return statistics.median(times), times[int(0.95 * (len(times) - 1))]


def fidelity(reference: SentenceTransformer, candidate: SentenceTransformer) -> tuple[float, float]:
    queries = [f"query: {q}" for q, _ in PAIRS]
    passages = [f"passage: {p}" for _, p in PAIRS]
    ref_q, ref_p = encode(reference, queries), encode(reference, passages)
    cand_q, cand_p = encode(candidate, queries), encode(candidate, passages)
    cosine = float(np.mean(np.sum(np.vstack([ref_q, ref_p]) * np.vstack([cand_q, cand_p]), 1)))
    same_top1 = float(np.mean((ref_q @ ref_p.T).argmax(1) == (cand_q @ cand_p.T).argmax(1)))
    return cosine, same_top1


def recall_at_1(model: SentenceTransformer) -> float:
    q = encode(model, [f"query: {q}" for q, _ in PAIRS])
    p = encode(model, [f"passage: {p}" for _, p in PAIRS])
    return float(np.mean((q @ p.T).argmax(1) == np.arange(len(PAIRS))))


def size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.parent.glob(path.name + "*")) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/e5")
    parser.add_argument("--onnx-dir", default="models/e5-onnx")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    onnx_dir = Path(args.onnx_dir)
    onnx_kwargs = {"provider": "CPUExecutionProvider"}

    if not (onnx_dir / "onnx" / "model_quint8_avx2.onnx").exists():
        print("exporting ONNX (one-off, a few minutes)...", flush=True)
        exported = SentenceTransformer(args.model, backend="onnx", model_kwargs=onnx_kwargs)
        exported.save_pretrained(str(onnx_dir))
        export_dynamic_quantized_onnx_model(exported, "avx2", str(onnx_dir))
        del exported

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    ort_kwargs = {**onnx_kwargs, "session_options": options}

    backends = {
        "torch fp32": (
            SentenceTransformer(args.model, device="cpu"),
            Path(args.model) / "model.safetensors",
        ),
        "onnx fp32": (
            SentenceTransformer(str(onnx_dir), backend="onnx", model_kwargs=ort_kwargs),
            onnx_dir / "onnx" / "model.onnx",
        ),
        "onnx int8 (avx2)": (
            SentenceTransformer(
                str(onnx_dir),
                backend="onnx",
                model_kwargs={**ort_kwargs, "file_name": "onnx/model_quint8_avx2.onnx"},
            ),
            onnx_dir / "onnx" / "model_quint8_avx2.onnx",
        ),
    }
    reference = backends["torch fp32"][0]

    print(f"\n{args.threads} threads, ~25-token inputs, {args.repeats} repeats\n")
    print(
        "| Backend | Size | p50 b=1 | p50 b=8 | p50 b=32 | p95 b=32 | Texts/s (b=32) "
        "| Cosine vs fp32 | Same top-1 | Recall@1 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, (model, weights) in backends.items():
        p50 = {b: latency(model, b, args.repeats) for b in (1, 8, 32)}
        cosine, same = fidelity(reference, model)
        cells = [
            name,
            f"{size_mb(weights):,.0f} MB",
            *(f"{p50[b][0]:.0f} ms" for b in (1, 8, 32)),
            f"{p50[32][1]:.0f} ms",
            f"{32 / p50[32][0] * 1000:.0f}",
            f"{cosine:.4f}",
            f"{same:.0%}",
            f"{recall_at_1(model):.0%}",
        ]
        print("| " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
