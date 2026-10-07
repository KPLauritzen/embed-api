"""Checks against the real model, for both backends. Run with `uv run pytest -m slow`.

Expects the models from the README quickstart: ./models/e5 (torch, from
scripts/download_model.py) and ./models/e5-int8 (onnx, from
scripts/export_onnx.py). A backend whose model is missing is skipped. e5
similarities cluster in 0.7-1.0, so assertions compare orderings rather than
absolute thresholds.
"""

from pathlib import Path

import numpy as np
import pytest

from embed_api.config import Backend, Settings
from embed_api.embedder import Embedder, load_embedder

pytestmark = pytest.mark.slow

MODELS = Path(__file__).parent.parent / "models"
PATHS = {Backend.TORCH: MODELS / "e5", Backend.ONNX: MODELS / "e5-int8"}

TEXTS = [
    "query: hvad er hovedstaden i Danmark?",
    "passage: København er Danmarks hovedstad og største by.",
    "passage: Bananer er en god kilde til kalium.",
    "query: The cat is sleeping on the couch.",
    "passage: " + "lang tekst " * 400,  # truncated at 512 tokens
]


def _load(backend: Backend) -> Embedder:
    path = PATHS[backend]
    if not path.exists():
        pytest.skip(f"{path} not found; see the README quickstart")
    return load_embedder(Settings(backend=backend, model_path=str(path)))


@pytest.fixture(scope="module", params=list(Backend), ids=lambda b: b.value)
def model(request: pytest.FixtureRequest) -> Embedder:
    return _load(request.param)


def test_shape_and_norm(model: Embedder) -> None:
    vectors = model.embed(["query: hej", "passage: verden"])

    assert vectors.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_query_prefers_relevant_passage(model: Embedder) -> None:
    query, relevant, irrelevant = model.embed(TEXTS[:3])

    assert query @ relevant > query @ irrelevant


def test_danish_and_english_paraphrases_are_close(model: Embedder) -> None:
    danish, english, unrelated = model.embed(
        [
            "query: Katten sover på sofaen.",
            "query: The cat is sleeping on the couch.",
            "query: Renten steg med et halvt procentpoint.",
        ]
    )

    assert danish @ english > danish @ unrelated


def test_token_count_includes_prefix_and_special_tokens(model: Embedder) -> None:
    [with_prefix] = model.count_tokens(["query: hej"])
    [bare] = model.count_tokens(["hej"])

    assert with_prefix > bare >= 3  # <s> hej </s>
    assert model.max_tokens == 512


def test_onnx_int8_matches_torch_fp32() -> None:
    torch_model, onnx_model = _load(Backend.TORCH), _load(Backend.ONNX)

    assert onnx_model.count_tokens(TEXTS) == torch_model.count_tokens(TEXTS)
    cosines = np.sum(torch_model.embed(TEXTS) * onnx_model.embed(TEXTS), axis=1)
    assert cosines.min() > 0.98, cosines


def test_embedding_does_not_depend_on_the_rest_of_the_batch(model: Embedder) -> None:
    # Dynamic int8 quantisation scales activations per input tensor, so a
    # batched run would let one text's vector depend on its neighbours.
    alone = model.embed([TEXTS[0]])[0]
    batched = model.embed([TEXTS[0], TEXTS[2], TEXTS[4]])[0]

    assert alone @ batched > 0.99999


def test_concurrent_counting_and_encoding() -> None:
    # Token counting runs outside the inference semaphore, concurrently with
    # encoding. With a shared transformers tokenizer, counting could switch off
    # truncation mid-encode and push >512 tokens into the model.
    from concurrent.futures import ThreadPoolExecutor

    model = _load(Backend.TORCH)
    long_text = [TEXTS[4]]
    with ThreadPoolExecutor(8) as pool:
        counts = [pool.submit(model.count_tokens, long_text * 4) for _ in range(200)]
        encodes = [pool.submit(model.embed, long_text) for _ in range(20)]
        for future in counts + encodes:
            future.result()
