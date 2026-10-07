import importlib.util
from pathlib import Path

from embeda_api.config import DEFAULT_MODEL_REVISION


def test_download_script_pins_the_same_revision() -> None:
    # The Docker model stage runs the script without installing the package,
    # so the revision lives in both places; this keeps them in step.
    path = Path(__file__).parent.parent / "scripts" / "download_model.py"
    spec = importlib.util.spec_from_file_location("download_model", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.REVISION == DEFAULT_MODEL_REVISION
