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

# Part of the API contract, so fixed here (and shown in the OpenAPI schema)
# rather than varying per deployment. The per-request token budget, which is
# what actually bounds inference cost, is configurable.
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

    # A single string is wrapped in a list before validation, so both shapes
    # share one set of rules and errors point at `input[i]`, not at a union branch.
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
        "characters each. Do not add the `query: `/`passage: ` prefix yourself."
    )
    input_type: InputType = Field(
        description="`query` for search queries and for symmetric tasks (similarity, "
        "clustering); `passage` for documents being searched. The server adds the "
        "prefix e5 was trained with. Required, because a wrong default silently "
        "lowers retrieval quality."
    )


class Embedding(BaseModel):
    index: int = Field(description="Position of the input this embedding belongs to.")
    embedding: list[float] = Field(description="L2-normalised: cosine similarity = dot product.")
    tokens: int = Field(description="Tokens embedded, including the prefix and special tokens.")
    truncated: bool = Field(
        description="True if the input exceeded the model's token limit and only its "
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
    dimension: int
    limits: Limits


class HealthResponse(BaseModel):
    status: str
