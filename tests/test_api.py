import io
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from embed_api.config import Settings
from embed_api.main import create_app
from tests.conftest import FakeEmbedder, wait_until


def embed(client: TestClient, payload: dict, **kwargs):
    return client.post("/v1/embed", json=payload, **kwargs)


class TestEmbed:
    def test_single_string(self, client: TestClient, fake: FakeEmbedder) -> None:
        r = embed(client, {"input": "hej verden", "input_type": "query"})

        assert r.status_code == 200
        body = r.json()
        assert (body["model"], body["dimension"]) == (fake.model_name, fake.dimension)
        [item] = body["embeddings"]
        assert item["index"] == 0
        assert len(item["embedding"]) == fake.dimension
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
        r = embed(client, payload)

        assert r.status_code == 422
        assert r.json()["error"]["code"] == "validation_error"

    def test_validation_error_points_at_the_item_without_echoing_it(
        self, client: TestClient
    ) -> None:
        r = embed(
            client,
            {"input": ["ok", "x" * 8001], "input_type": "query"},
            headers={"X-Request-ID": "req-422"},
        )

        error = r.json()["error"]
        assert (error["code"], error["request_id"]) == ("validation_error", "req-422")
        assert error["details"][0]["loc"] == ["body", "input", 1]
        assert "xxxx" not in r.text

    def test_token_budget(self, client: TestClient) -> None:
        # 40 inputs x 16 tokens = 640 > the fixture's budget of 512
        texts = [" ".join(["word"] * 30)] * 40
        r = embed(client, {"input": texts, "input_type": "passage"})

        assert r.status_code == 422
        assert r.json()["error"]["code"] == "token_budget_exceeded"

    def test_body_too_large(self, client: TestClient) -> None:
        r = embed(client, {"input": ["x" * 8000, "y" * 8000, "z" * 8000], "input_type": "query"})

        assert r.status_code == 413
        assert r.json()["error"]["code"] == "request_too_large"

    def test_body_too_large_when_chunked(self, client: TestClient) -> None:
        def chunks():
            yield b'{"input": "'
            for _ in range(30):
                yield b"x" * 1000
            yield b'", "input_type": "query"}'

        r = client.post("/v1/embed", content=chunks(), headers={"content-type": "application/json"})

        assert r.status_code == 413
        assert r.json()["error"]["code"] == "request_too_large"


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
            wait_until(
                lambda: client.get("/health/ready").json()["error"]["code"] == "model_load_failed"
            )
            # A failed load never recovers by itself, so liveness asks for a restart.
            assert client.get("/health/live").status_code == 503


def test_info(client: TestClient, settings: Settings, fake: FakeEmbedder) -> None:
    r = client.get("/v1/info")

    assert r.status_code == 200
    assert r.json()["revision"] == fake.revision
    assert r.json()["limits"]["max_total_tokens"] == settings.max_total_tokens
    assert r.json()["limits"]["max_tokens_per_input"] == fake.max_tokens


@pytest.mark.parametrize(
    ("method", "path", "status", "code"),
    [("GET", "/nope", 404, "not_found"), ("GET", "/v1/embed", 405, "method_not_allowed")],
)
def test_routing_errors_use_the_envelope(
    client: TestClient, method: str, path: str, status: int, code: str
) -> None:
    r = client.request(method, path, headers={"X-Request-ID": "req-404"})

    assert r.status_code == status
    assert r.json()["error"]["code"] == code
    assert r.json()["error"]["request_id"] == "req-404"


def test_root_redirects_to_docs(client: TestClient) -> None:
    r = client.get("/", follow_redirects=False)

    assert r.status_code == 307
    assert r.headers["location"] == "/docs"


def test_input_text_is_never_logged(client: TestClient, logs: io.StringIO) -> None:
    embed(client, {"input": "very-private-text", "input_type": "query"})

    logged = logs.getvalue()
    assert '"event": "request"' in logged  # the access line was captured...
    assert "very-private-text" not in logged  # ...and carries no input
