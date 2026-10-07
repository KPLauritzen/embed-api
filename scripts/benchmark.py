"""Speed and fidelity of the API's backends, measured through the API's own code.

    uv run scripts/download_model.py && uv run scripts/export_onnx.py   # once
    uv run python scripts/benchmark.py --threads 4

Three rows, each timed through the embedder class the server uses:

- torch fp32: the reference (sentence-transformers, batches of 16)
- ONNX Runtime fp32: the Hub's fp32 ONNX export through OnnxEmbedder, which
  separates the runtime's contribution from quantisation's (downloaded once
  to models/e5-onnx-fp32)
- ONNX Runtime int8: what the API serves (models/e5-int8)

Fidelity is measured against torch fp32 on a small Danish/English retrieval
set: cosine similarity between the embeddings (mean and minimum), and whether
each query's top-ranked passage is unchanged. Twelve pairs make this a sanity
check, not an evaluation.
"""

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from embed_api.config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, Backend, Settings
from embed_api.embedder import Embedder, load_embedder

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
QUERIES = [f"query: {q}" for q, _ in PAIRS]
PASSAGES = [f"passage: {p}" for _, p in PAIRS]

SENTENCE = (
    "passage: Dette er en typisk sætning på omkring tyve ord, som bruges til at måle, "
    "hvor hurtigt modellen svarer på en almindelig CPU."
)


def fp32_onnx_dir(int8_dir: Path, dest: Path) -> Path:
    """The Hub's fp32 ONNX export, laid out like the int8 directory."""
    if not (dest / "model.onnx").exists():
        snapshot_download(
            repo_id=DEFAULT_MODEL_ID,
            revision=DEFAULT_MODEL_REVISION,
            allow_patterns=["onnx/model.onnx", "onnx/model.onnx_data"],
            local_dir=dest,
        )
        for name in ("model.onnx", "model.onnx_data"):
            shutil.move(dest / "onnx" / name, dest / name)
        for name in ("tokenizer.json", "sentence_bert_config.json"):
            shutil.copy(int8_dir / name, dest / name)
        source = {"model_id": DEFAULT_MODEL_ID, "revision": DEFAULT_MODEL_REVISION}
        (dest / "source.json").write_text(json.dumps(source))
    return dest


def p50_ms(model: Embedder, batch: int, repeats: int) -> float:
    texts = [SENTENCE] * batch
    model.embed(texts)  # warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.embed(texts)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def weights_mb(backend: Backend, path: Path) -> float:
    if backend is Backend.TORCH:
        return (path / "model.safetensors").stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.iterdir() if f.name.startswith("model.")) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-path", default="models/e5")
    parser.add_argument("--int8-path", default="models/e5-int8")
    parser.add_argument("--fp32-onnx-path", default="models/e5-onnx-fp32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    int8 = Path(args.int8_path)
    paths = {
        "torch fp32": (Backend.TORCH, Path(args.torch_path)),
        "ONNX Runtime fp32": (Backend.ONNX, fp32_onnx_dir(int8, Path(args.fp32_onnx_path))),
        "ONNX Runtime int8 (served)": (Backend.ONNX, int8),
    }
    models = {
        name: load_embedder(Settings(backend=b, model_path=str(p), num_threads=args.threads))
        for name, (b, p) in paths.items()
    }
    ref_q, ref_p = (models["torch fp32"].embed(t) for t in (QUERIES, PASSAGES))
    ref_top1 = (ref_q @ ref_p.T).argmax(1)

    print(f"\n{args.threads} threads, ~30-token inputs, median of {args.repeats} runs\n")
    print(
        "| Backend | Weights | 1 text | 8 texts | 32 texts | Texts/s "
        "| Cosine vs fp32 (mean / min) | Same top-1 |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for name, model in models.items():
        q, p = model.embed(QUERIES), model.embed(PASSAGES)
        cosines = np.concatenate([np.sum(q * ref_q, 1), np.sum(p * ref_p, 1)])
        same_top1 = np.mean((q @ p.T).argmax(1) == ref_top1)
        ms = {b: p50_ms(model, b, args.repeats) for b in (1, 8, 32)}
        cells = [
            name,
            f"{weights_mb(*paths[name]):,.0f} MB",
            *(f"{ms[b]:,.0f} ms" for b in (1, 8, 32)),
            f"{32 / ms[32] * 1000:.0f}",
            f"{cosines.mean():.4f} / {cosines.min():.4f}",
            f"{same_top1:.0%}",
        ]
        print("| " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
