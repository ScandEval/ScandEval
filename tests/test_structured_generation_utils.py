"""Unit tests for the `structured_generation_utils` module."""

import pytest

from scandeval.structured_generation_utils import get_ner_schema


@pytest.mark.parametrize(
    argnames=["ner_tag_names", "expected"],
    argvalues=[
        (list(), dict(properties=dict(), title="AnswerFormat", type="object")),
        (
            ["person"],
            dict(
                properties=dict(
                    person=dict(
                        items=dict(type="string"),
                        maxItems=5,
                        title="Person",
                        type="array",
                    )
                ),
                required=["person"],
                title="AnswerFormat",
                type="object",
            ),
        ),
        (
            ["person", "location"],
            dict(
                properties=dict(
                    location=dict(
                        items=dict(type="string"),
                        maxItems=5,
                        title="Location",
                        type="array",
                    ),
                    person=dict(
                        items=dict(type="string"),
                        maxItems=5,
                        title="Person",
                        type="array",
                    ),
                ),
                required=["person", "location"],
                title="AnswerFormat",
                type="object",
            ),
        ),
    ],
)
def test_get_ner_schema(ner_tag_names, expected):
    """Test that the NER schema can be retrieved."""
    schema = get_ner_schema(ner_tag_names=ner_tag_names).model_json_schema()
    assert schema == expected


# TODO
def test_get_ner_logits_processors():
    """Test that the NER logits processors can be retrieved."""
    pass
