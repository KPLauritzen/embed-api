# /// script
# requires-python = ">=3.12"
# dependencies = ["onnxruntime==1.30.0", "onnx==1.23.2", "huggingface-hub==1.33.0"]
# ///
"""Build the int8 ONNX model that the API serves, in models/e5-int8.

    uv run scripts/export_onnx.py

Downloads the fp32 ONNX export from the model repository at the pinned revision
(2.2 GB) and quantises it with ONNX Runtime's dynamic quantisation. The settings
are those of sentence-transformers' `export_dynamic_quantized_onnx_model(model,
"avx2")`: uint8, per-channel, symmetric weights, full range. This needs neither
torch nor optimum, and less memory than exporting from PyTorch, which does not fit
on a CI runner. The repository's own int8 file is built for AVX-512 VNNI, which
the deployment CPU does not have.

Writes model.onnx (about 560 MB), tokenizer.json, sentence_bert_config.json, and
source.json with the model id and revision. Needs about 8.5 GB of RAM.
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, str(Path(__file__).parent))
from download_model import MODEL_ID, REVISION

FILES = [
    "onnx/model.onnx",
    "onnx/model.onnx_data",
    "tokenizer.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="models/e5-int8")
    args = parser.parse_args()
    dest = Path(args.dest)

    # Hides ONNX Runtime's advice to pre-process the model, which is for static quantisation.
    logging.getLogger().setLevel(logging.ERROR)

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        snapshot_download(repo_id=MODEL_ID, revision=REVISION, allow_patterns=FILES, local_dir=src)

        # The ONNX backend implements mean pooling only.
        pooling = json.loads((src / "1_Pooling" / "config.json").read_text())
        if not pooling.get("pooling_mode_mean_tokens"):
            raise SystemExit(f"expected mean pooling, got {pooling}")

        dest.mkdir(parents=True, exist_ok=True)
        quantize_dynamic(
            src / "onnx" / "model.onnx",
            dest / "model.onnx",
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
            extra_options={"WeightSymmetric": True, "ActivationSymmetric": False},
        )
        for name in ("tokenizer.json", "sentence_bert_config.json"):
            shutil.copy(src / name, dest / name)
    source = {"model_id": MODEL_ID, "revision": REVISION, "quantization": "dynamic int8, avx2"}
    (dest / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
