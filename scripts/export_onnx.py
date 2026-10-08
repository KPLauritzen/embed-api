# /// script
# requires-python = ">=3.12"
# dependencies = ["onnxruntime==1.30.0", "onnx==1.23.2", "huggingface-hub==1.33.0"]
# ///
"""Build the int8 ONNX model the default backend serves.

    uv run scripts/export_onnx.py --dest models/e5-int8

Downloads the fp32 ONNX export that the Hugging Face repo ships at the pinned
revision (2.2 GB) and quantises it with ONNX Runtime's dynamic quantiser, using
the settings of sentence-transformers' `export_dynamic_quantized_onnx_model(
model, "avx2")` (optimum's AVX2 config: uint8, per-channel, symmetric weights,
full range). No torch or optimum needed; exporting from PyTorch needs more
memory than a CI runner has. The Hub's own int8 file targets AVX-512 VNNI,
which loses accuracy on CPUs without it; the deployment CPU has AVX2.

Writes only what the ONNX backend loads: `model.onnx` (~560 MB),
`tokenizer.json`, `sentence_bert_config.json`, and `source.json` recording the
model id and revision the weights came from. Peak memory is ~8.5 GB.
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
from download_model import MODEL_ID, REVISION  # one place pins the revision

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

    # ORT suggests shape-inference pre-processing for static quantisation; it
    # does not apply to dynamic quantisation of a transformer.
    logging.getLogger().setLevel(logging.ERROR)

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        snapshot_download(repo_id=MODEL_ID, revision=REVISION, allow_patterns=FILES, local_dir=src)

        # The ONNX backend computes mean pooling itself; refuse anything else.
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
