"""Tests for the `constants` module."""

from scandeval import constants


def test_all_objects_in_constants_are_constants():
    """Test that all objects in the `constants` module are constants."""
    for name in dir(constants):
        if name.startswith("__"):
            continue
        assert name.isupper()
        assert isinstance(getattr(constants, name), (int, str, list, dict))
