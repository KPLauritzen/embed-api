# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub==1.33.0"]
# ///
"""Download the pinned fp32 model for the torch backend (models/e5, 2.2 GB).

This file is also the one place the model and its revision are pinned;
export_onnx.py imports them from here. Only the files sentence-transformers
loads are fetched: the repository also has a .bin copy and ONNX and OpenVINO
exports (~9.5 GB in all).
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="models/e5")
    dest = Path(parser.parse_args().dest)

    snapshot_download(
        repo_id=MODEL_ID, revision=REVISION, allow_patterns=ALLOW_PATTERNS, local_dir=dest
    )
    # The server reports this as the model it loaded (/v1/info).
    source = {"model_id": MODEL_ID, "revision": REVISION}
    (dest / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
