import io
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from embed_api.config import Settings
from embed_api.main import create_app
from tests.conftest import FakeEmbedder


def embed(client: TestClient, payload: dict, **kwargs):
    return client.post("/v1/embed", json=payload, **kwargs)


class TestEmbed:
    def test_single_string(self, client: TestClient) -> None:
        r = embed(client, {"input": "hej verden", "input_type": "query"})

        assert r.status_code == 200
        body = r.json()
        assert body["model"] == "fake/e5"
        assert body["dimension"] == 8
        [item] = body["embeddings"]
        assert item["index"] == 0
        assert len(item["embedding"]) == 8
        assert np.isclose(np.linalg.norm(item["embedding"]), 1.0, atol=1e-5)
        assert item == item | {"tokens": 5, "truncated": False}  # "query: hej verden" + 2
        assert body["usage"] == {"total_tokens": 5}

    def test_batch_keeps_order(self, client: TestClient) -> None:
        texts = ["one", "two", "three"]
        r = embed(client, {"input": texts, "input_type": "passage"})

        assert r.status_code == 200
        assert [e["index"] for e in r.json()["embeddings"]] == [0, 1, 2]

    @pytest.mark.parametrize("input_type", ["query", "passage"])
    def test_server_adds_e5_prefix(
        self, client: TestClient, fake: FakeEmbedder, input_type: str
    ) -> None:
        embed(client, {"input": ["a", "b"], "input_type": input_type})

        assert fake.calls[-1] == [f"{input_type}: a", f"{input_type}: b"]

    def test_over_long_input_is_truncated_and_flagged(self, client: TestClient) -> None:
        long_text = " ".join(["word"] * 50)  # FakeEmbedder.max_tokens is 16
        r = embed(client, {"input": [long_text, "short"], "input_type": "passage"})

        assert r.status_code == 200
        long_item, short_item = r.json()["embeddings"]
        assert (long_item["tokens"], long_item["truncated"]) == (16, True)
        assert short_item["truncated"] is False
        assert r.json()["usage"]["total_tokens"] == 16 + 4


class TestValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"input": "hej"}, id="missing input_type"),
            pytest.param({"input": "hej", "input_type": "document"}, id="unknown input_type"),
            pytest.param({"input_type": "query"}, id="missing input"),
            pytest.param({"input": "", "input_type": "query"}, id="empty string"),
            pytest.param({"input": " \n\t", "input_type": "query"}, id="whitespace only"),
            pytest.param({"input": [], "input_type": "query"}, id="empty list"),
            pytest.param({"input": ["ok", ""], "input_type": "query"}, id="empty item"),
            pytest.param({"input": [1, 2], "input_type": "query"}, id="not strings"),
            pytest.param({"input": ["x"] * 65, "input_type": "query"}, id="too many inputs"),
            pytest.param({"input": "x" * 8001, "input_type": "query"}, id="input too long"),
            pytest.param({"input": "hej", "input_type": "query", "model": "x"}, id="extra field"),
        ],
    )
    def test_rejected_with_422(self, client: TestClient, payload: dict) -> None:
        assert embed(client, payload).status_code == 422

    def test_error_points_at_offending_item(self, client: TestClient) -> None:
        r = embed(client, {"input": ["ok", "  "], "input_type": "query"})

        assert r.json()["error"]["details"][0]["loc"] == ["body", "input", 1]

    def test_validation_error_uses_envelope_without_echoing_input(self, client: TestClient) -> None:
        r = embed(
            client,
            {"input": "x" * 8001, "input_type": "query"},
            headers={"X-Request-ID": "req-422"},
        )

        error = r.json()["error"]
        assert (error["code"], error["request_id"]) == ("validation_error", "req-422")
        assert error["details"][0]["loc"] == ["body", "input", 0]
        assert "xxxx" not in r.text

    def test_token_budget(self, client: TestClient) -> None:
        # 40 inputs x 16 tokens = 640 > the fixture's budget of 512
        texts = [" ".join(["word"] * 30)] * 40
        r = embed(client, {"input": texts, "input_type": "passage"})

        assert r.status_code == 422
        assert r.json()["error"]["code"] == "token_budget_exceeded"

    def test_body_too_large_by_content_length(self, client: TestClient) -> None:
        r = embed(client, {"input": ["x" * 8000, "y" * 8000, "z" * 8000], "input_type": "query"})

        assert r.status_code == 413
        assert r.json()["error"]["code"] == "request_too_large"

    def test_malformed_content_length(self, client: TestClient) -> None:
        r = client.post(
            "/v1/embed",
            content=b"{}",
            headers={"content-type": "application/json", "content-length": "abc"},
        )

        assert r.status_code == 400

    def test_body_too_large_when_chunked(self, client: TestClient) -> None:
        def chunks():
            yield b'{"input": "'
            for _ in range(30):
                yield b"x" * 1000
            yield b'", "input_type": "query"}'

        r = client.post("/v1/embed", content=chunks(), headers={"content-type": "application/json"})

        assert r.status_code == 413


class TestRequestId:
    def test_generated_when_absent(self, client: TestClient) -> None:
        r = client.get("/v1/info")

        assert len(r.headers["x-request-id"]) == 16

    def test_caller_id_is_echoed(self, client: TestClient) -> None:
        r = client.get("/v1/info", headers={"X-Request-ID": "trace-abc.123"})

        assert r.headers["x-request-id"] == "trace-abc.123"

    def test_unsafe_caller_id_is_replaced(self, client: TestClient) -> None:
        r = client.get("/v1/info", headers={"X-Request-ID": "bad id\nwith newline"})

        assert r.headers["x-request-id"] != "bad id\nwith newline"

    def test_error_body_carries_the_id(self, client: TestClient) -> None:
        r = embed(
            client,
            {"input": ["x" * 8000, "y" * 8000, "z" * 8000], "input_type": "query"},
            headers={"X-Request-ID": "req-1"},
        )

        assert r.json()["error"]["request_id"] == "req-1"


class TestLifecycle:
    def test_not_ready_while_loading(self, settings: Settings) -> None:
        release = threading.Event()

        def slow_factory(_: Settings) -> FakeEmbedder:
            release.wait(5)
            return FakeEmbedder()

        with TestClient(create_app(settings, embedder_factory=slow_factory)) as client:
            assert client.get("/health/live").status_code == 200
            ready = client.get("/health/ready")
            assert ready.status_code == 503
            assert ready.json()["error"]["code"] == "model_not_ready"
            assert embed(client, {"input": "x", "input_type": "query"}).status_code == 503
            release.set()

    def test_load_failure_is_reported(self, settings: Settings) -> None:
        def broken_factory(_: Settings) -> FakeEmbedder:
            raise OSError("no such model")

        with TestClient(create_app(settings, embedder_factory=broken_factory)) as client:
            for _ in range(200):
                r = client.get("/health/ready")
                if r.json()["error"]["code"] == "model_load_failed":
                    break
                threading.Event().wait(0.01)
            assert r.status_code == 503
            assert r.json()["error"]["code"] == "model_load_failed"
            # A failed load never recovers by itself, so liveness asks for a restart.
            assert client.get("/health/live").status_code == 503

    def test_unexpected_error_hides_internals(
        self, client: TestClient, fake: FakeEmbedder, logs: io.StringIO
    ) -> None:
        def explode(texts: list[str]) -> np.ndarray:
            raise RuntimeError(f"failed on {texts!r}")

        fake.embed = explode  # type: ignore[method-assign]
        r = embed(client, {"input": "very-private-text", "input_type": "query"})

        assert r.status_code == 500
        assert r.json()["error"]["code"] == "internal_error"
        assert r.json()["error"]["request_id"] == r.headers["x-request-id"]
        assert "private" not in r.text
        logged = logs.getvalue()
        assert '"unhandled_error"' in logged  # the failure is logged...
        assert "very-private-text" not in logged  # ...but not the message carrying input


def test_info(client: TestClient, settings: Settings) -> None:
    r = client.get("/v1/info")

    assert r.status_code == 200
    assert r.json()["limits"]["max_total_tokens"] == settings.max_total_tokens
    assert r.json()["limits"]["max_tokens_per_input"] == 16


def test_root_redirects_to_docs(client: TestClient) -> None:
    r = client.get("/", follow_redirects=False)

    assert r.status_code == 307
    assert r.headers["location"] == "/docs"


def test_input_text_is_never_logged(client: TestClient, logs: io.StringIO) -> None:
    embed(client, {"input": "very-private-text", "input_type": "query"})

    logged = logs.getvalue()
    assert '"event": "request"' in logged  # the access line was captured...
    assert "very-private-text" not in logged  # ...and carries no input
