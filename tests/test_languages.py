"""Unit tests for the `languages` module."""

import pytest

from src.scandeval.languages import get_all_languages


class TestGetAllLanguages:
    @pytest.fixture(scope="class")
    def languages(self):
        yield get_all_languages()

    def test_languages_is_dict(self, languages):
        assert isinstance(languages, dict)

    def test_languages_contain_scandinavian_languages(self, languages):
        assert "sv" in languages
        assert "da" in languages
        assert "no" in languages
        assert "nb" in languages
        assert "nn" in languages
