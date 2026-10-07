"""Download the pinned model revision to a local directory.

Only the files the PyTorch sentence-transformers path needs are fetched; the
Hugging Face repo also carries a .bin copy, ONNX and OpenVINO exports (~9.5 GB
in total). The model is loaded from this directory by path, which works offline
without relying on the Hugging Face cache layout.
"""

import argparse
import os

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=os.environ.get("EMBED_MODEL_ID", MODEL_ID))
    parser.add_argument("--revision", default=os.environ.get("EMBED_MODEL_REVISION", REVISION))
    parser.add_argument("--dest", default="models/e5")
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        allow_patterns=ALLOW_PATTERNS,
        local_dir=args.dest,
    )
    print(path)


if __name__ == "__main__":
    main()
