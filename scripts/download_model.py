# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub==1.33.0"]
# ///
"""Download the pinned fp32 model for the torch backend to models/e5 (2.2 GB).

The model id and revision are pinned here, and export_onnx.py imports them.
Only the files sentence-transformers loads are downloaded; the full repository,
with a .bin copy and ONNX and OpenVINO exports, is about 9.5 GB.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "intfloat/multilingual-e5-large"
REVISION = "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"
ALLOW_PATTERNS = [
    "config.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/*",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]

# Not pinned to a commit yet: replace "main" with one once the model has been tried.
GEMMA_ID = "google/embeddinggemma-2"
GEMMA_REVISION = "main"
GEMMA_ALLOW_PATTERNS = [
    "*.json",  # includes config_sentence_transformers.json, which holds the prompts
    "model*.safetensors",
    "tokenizer.model",
    "*_Pooling/*",
    "*_Dense/*",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["e5", "embeddinggemma-2"], default="e5")
    parser.add_argument("--dest")
    args = parser.parse_args()
    if args.model == "e5":
        model_id, revision, patterns = MODEL_ID, REVISION, ALLOW_PATTERNS
    else:
        model_id, revision, patterns = GEMMA_ID, GEMMA_REVISION, GEMMA_ALLOW_PATTERNS
    dest = Path(args.dest or f"models/{args.model}")

    snapshot_download(repo_id=model_id, revision=revision, allow_patterns=patterns, local_dir=dest)
    # /v1/info reports this as the loaded model.
    source = {"model_id": model_id, "revision": revision}
    (dest / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
