# /// script
# requires-python = ">=3.12"
# dependencies = ["onnxruntime==1.30.0", "huggingface-hub==1.33.0"]
# ///
"""Build the int8 ONNX model the default backend serves.

    uv run scripts/download_model.py --dest models/e5
    uv run scripts/export_onnx.py --src models/e5 --dest models/e5-int8

Quantises the fp32 ONNX export that the Hugging Face repo ships at the pinned
revision, with ONNX Runtime's dynamic quantiser. The settings match
sentence-transformers' `export_dynamic_quantized_onnx_model(model, "avx2")`
(optimum's AVX2 config: uint8, per-channel, symmetric weights, full range),
without needing torch or optimum: exporting from PyTorch needs more memory than
a CI runner has. The Hub's own int8 file targets AVX-512 VNNI, which loses
accuracy on CPUs without it; the deployment CPU has AVX2.

Writes only what the ONNX backend loads: `model.onnx` (~560 MB), and
`tokenizer.json` and `sentence_bert_config.json` copied from --src.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, str(Path(__file__).parent))
from download_model import MODEL_ID, REVISION  # one place pins the revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="models/e5", help="output of download_model.py")
    parser.add_argument("--dest", default="models/e5-int8")
    args = parser.parse_args()
    src, dest = Path(args.src), Path(args.dest)

    # The ONNX backend computes mean pooling itself; refuse anything else.
    pooling = json.loads((src / "1_Pooling" / "config.json").read_text())
    if not pooling.get("pooling_mode_mean_tokens"):
        raise SystemExit(f"expected mean pooling, got {pooling}")

    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            allow_patterns=["onnx/model.onnx", "onnx/model.onnx_data"],
            local_dir=tmp,
        )
        quantize_dynamic(
            Path(tmp) / "onnx" / "model.onnx",
            dest / "model.onnx",
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
            extra_options={"WeightSymmetric": True, "ActivationSymmetric": False},
        )
    for name in ("tokenizer.json", "sentence_bert_config.json"):
        shutil.copy(src / name, dest / name)
    print(dest)


if __name__ == "__main__":
    main()
