"""Speed and memory of embedding models, measured through the API's own embedder classes.

    just bench    # needs models/e5 and models/e5-int8 (just model model-torch)
    uv run --extra torch python scripts/benchmark.py --model torch:models/e5 \
        --model torch:models/embeddinggemma-2

Without --model it measures e5 three ways: PyTorch fp32, ONNX Runtime fp32 (the
model repository's ONNX export, downloaded once to models/e5-onnx-fp32) and the
served ONNX int8 model. The middle one shows how much of the speedup comes from
ONNX Runtime and how much from quantisation.

Each model runs in its own process, so its peak memory is its own. Texts get the
model's own query and document prefixes. The report also has a quick quality check
on 12 Danish and English query/passage pairs: whether each query ranks its own
passage first, and, for the same model, the cosine similarity to the first model's
embeddings. See eval_retrieval.py for a proper quality measurement.
"""

import argparse
import json
import multiprocessing
import resource
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from embed_api.config import Backend, Settings
from embed_api.embedder import Embedder, InputType, load_embedder, read_prefixes

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
LONG = " ".join([SENTENCE] * 14)  # about 420 e5 tokens: long, but under every model's limit


def fp32_onnx_dir(int8_dir: Path, dest: Path) -> Path:
    """The repository's fp32 ONNX export, in a directory laid out like the int8 one."""
    if not (dest / "model.onnx").exists():
        source = json.loads((int8_dir / "source.json").read_text())
        snapshot_download(
            repo_id=source["model_id"],
            revision=source["revision"],
            allow_patterns=["onnx/model.onnx", "onnx/model.onnx_data"],
            local_dir=dest,
        )
        for name in ("model.onnx", "model.onnx_data"):
            shutil.move(dest / "onnx" / name, dest / name)
        for name in ("tokenizer.json", "sentence_bert_config.json", "source.json"):
            shutil.copy(int8_dir / name, dest / name)
    return dest


def p50_ms(model: Embedder, texts: list[str], repeats: int) -> float:
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


def measure(backend: Backend, path: str, threads: int, repeats: int) -> dict:
    """Runs in a fresh process, so the peak memory belongs to this model alone."""
    settings = Settings(backend=backend, model_path=path, num_threads=threads)
    prefixes = read_prefixes(settings.model_dir)
    start = time.perf_counter()
    model = load_embedder(settings)
    model.embed([prefixes[InputType.QUERY] + "warm-up"])
    load_s = time.perf_counter() - start
    passage = prefixes[InputType.PASSAGE] + SENTENCE
    long_texts = [prefixes[InputType.PASSAGE] + LONG] * 16
    return {
        "name": f"{Path(path).name} ({model.backend.split('-')[0]})",
        "model": model.model_name,
        "weights_mb": weights_mb(backend, Path(path)),
        "dimension": model.dimension,
        "load_s": load_s,
        "ms": {b: p50_ms(model, [passage] * b, repeats) for b in (1, 8, 32)},
        "long_s": p50_ms(model, long_texts, max(3, repeats // 3)) / 1000,
        "long_tokens": sum(n for n, _ in model.count_tokens(long_texts)),
        "queries": model.embed([prefixes[InputType.QUERY] + q for q, _ in PAIRS]),
        "passages": model.embed([prefixes[InputType.PASSAGE] + p for _, p in PAIRS]),
        "peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--model",
        action="append",
        metavar="BACKEND:PATH",
        help="model to measure, repeatable (default: e5 as torch fp32, ONNX fp32 and ONNX int8); "
        "cosine is reported against the first model when it is the same model",
    )
    args = parser.parse_args()
    if args.model:
        specs = [spec.split(":", 1) for spec in args.model]
    else:
        fp32_onnx = fp32_onnx_dir(Path("models/e5-int8"), Path("models/e5-onnx-fp32"))
        specs = [["torch", "models/e5"], ["onnx", str(fp32_onnx)], ["onnx", "models/e5-int8"]]

    spawn = multiprocessing.get_context("spawn")
    results = []
    for backend, path in specs:
        with spawn.Pool(1) as pool:
            results.append(
                pool.apply(measure, (Backend(backend), path, args.threads, args.repeats))
            )

    ref = results[0]
    print(f"\n{args.threads} threads, medians of {args.repeats} runs\n")
    print(
        "| Model | Weights | Dim | Load | 1 text | 8 texts | 32 texts | Texts/s "
        "| 16 long texts | Peak memory | Cosine vs first (min) | Recall@1, 12 pairs |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        q, p = r["queries"], r["passages"]
        if r["model"] == ref["model"]:
            cosines = np.concatenate(
                [np.sum(q * ref["queries"], 1), np.sum(p * ref["passages"], 1)]
            )
            cosine = f"{cosines.min():.4f}"
        else:
            cosine = "-"  # a different model's vectors are not comparable
        cells = [
            r["name"],
            f"{r['weights_mb']:,.0f} MB",
            str(r["dimension"]),
            f"{r['load_s']:.1f} s",
            *(f"{r['ms'][b]:,.0f} ms" for b in (1, 8, 32)),
            f"{32 / r['ms'][32] * 1000:.0f}",
            f"{r['long_s']:.1f} s ({r['long_tokens']:,} tokens)",
            f"{r['peak_mb']:,.0f} MB",
            cosine,
            f"{np.mean((q @ p.T).argmax(1) == np.arange(len(PAIRS))):.0%}",
        ]
        print("| " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
