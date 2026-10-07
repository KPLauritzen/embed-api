# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sentence-transformers[onnx]==5.7.0", "torch"]
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Export the downloaded model to ONNX and quantise it to int8 for AVX2.

    uv run scripts/download_model.py --dest models/e5
    uv run scripts/export_onnx.py --src models/e5 --dest models/e5-int8

Writes only what the ONNX backend loads: `model.onnx` (int8, ~560 MB),
`tokenizer.json` and `sentence_bert_config.json`. Runs with its own
dependencies: the export tooling (optimum) pins older transformers and
huggingface-hub releases, and is never needed at runtime.

Dynamic quantisation for AVX2 matches the deployment CPU. The int8 file shipped
on the Hub targets AVX-512 VNNI, which loses accuracy on CPUs without it.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from sentence_transformers import SentenceTransformer, export_dynamic_quantized_onnx_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="models/e5")
    parser.add_argument("--dest", default="models/e5-int8")
    args = parser.parse_args()
    src, dest = Path(args.src), Path(args.dest)

    # The ONNX backend computes mean pooling itself; refuse anything else.
    pooling = json.loads((src / "1_Pooling" / "config.json").read_text())
    if not pooling.get("pooling_mode_mean_tokens"):
        raise SystemExit(f"expected mean pooling, got {pooling}")

    with tempfile.TemporaryDirectory() as tmp:
        model = SentenceTransformer(str(src), backend="onnx", device="cpu")
        model.save_pretrained(tmp)
        export_dynamic_quantized_onnx_model(model, "avx2", tmp)

        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(tmp) / "onnx" / "model_quint8_avx2.onnx", dest / "model.onnx")
    for name in ("tokenizer.json", "sentence_bert_config.json"):
        shutil.copy(src / name, dest / name)
    print(dest)


if __name__ == "__main__":
    main()
