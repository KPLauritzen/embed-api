from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    WithJsonSchema,
)

from embed_api.embedder import InputType

# Fixed limits that are part of the API contract and appear in the OpenAPI schema.
# The token budget, which is what bounds the cost of a request, is a setting.
MAX_INPUTS = 64
MAX_CHARS = 8000


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain non-whitespace characters")
    return value


def _as_list(value: Any) -> Any:
    return [value] if isinstance(value, str) else value


_TEXT_SCHEMA = {"type": "string", "minLength": 1, "maxLength": MAX_CHARS}

Text = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_CHARS),
    AfterValidator(_not_blank),
]


class EmbedRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"input": "Hvordan bliver vejret i morgen?", "input_type": "query"},
                {
                    "input": [
                        "København er Danmarks hovedstad.",
                        "Copenhagen is the capital of Denmark.",
                    ],
                    "input_type": "passage",
                },
            ]
        },
    )

    # A single string becomes a one-item list before validation, so both forms follow the
    # same rules and an error points at input[i].
    input: Annotated[
        list[Text],
        BeforeValidator(_as_list),
        Field(min_length=1, max_length=MAX_INPUTS),
        WithJsonSchema(
            {
                "anyOf": [
                    _TEXT_SCHEMA,
                    {
                        "type": "array",
                        "items": _TEXT_SCHEMA,
                        "minItems": 1,
                        "maxItems": MAX_INPUTS,
                    },
                ]
            }
        ),
    ] = Field(
        description=f"One text, or a list of up to {MAX_INPUTS} texts of at most {MAX_CHARS} "
        "characters each. Send the text without the `query: ` or `passage: ` prefix."
    )
    input_type: InputType = Field(
        description="`query` for search queries and for symmetric tasks such as similarity "
        "or clustering; `passage` for the documents being searched. The server adds the "
        "matching e5 prefix. There is no default, because the wrong prefix gives worse "
        "results without any error."
    )


class Embedding(BaseModel):
    index: int = Field(description="Position of the input this embedding belongs to.")
    embedding: list[float] = Field(
        description="L2-normalised, so cosine similarity is the dot product."
    )
    tokens: int = Field(description="Tokens embedded, including the prefix and special tokens.")
    truncated: bool = Field(
        description="True if the text was longer than the model's limit and only its "
        "beginning was embedded."
    )


class Usage(BaseModel):
    total_tokens: int


class EmbedResponse(BaseModel):
    model: str
    input_type: InputType
    dimension: int
    embeddings: list[Embedding]
    usage: Usage


class Limits(BaseModel):
    max_inputs: int
    max_chars_per_input: int
    max_tokens_per_input: int
    max_total_tokens: int
    max_body_bytes: int


class InfoResponse(BaseModel):
    model: str
    revision: str
    backend: str = Field(description="onnx-int8 (default) or torch-fp32.")
    dimension: int
    limits: Limits


class HealthResponse(BaseModel):
    status: str
