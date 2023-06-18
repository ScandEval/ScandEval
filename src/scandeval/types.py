"""Types used throughout the project."""

from typing import Any, TypeGuard

SCORE_DICT = dict[str, dict[str, float] | dict[str, list[dict[str, float]]]]


def is_list_of_int(x: Any) -> TypeGuard[list[int]]:
    """Check if an object is a list of integers.

    Args:
        x (Any):
            The object to check.

    Returns:
        TypeGuard[list[int]]:
            Whether the object is a list of integers.
    """
    return isinstance(x, list) and all(isinstance(i, int) for i in x)


def is_list_of_str(x: Any) -> TypeGuard[list[str]]:
    """Check if an object is a list of integers.

    Args:
        x (Any):
            The object to check.

    Returns:
        TypeGuard[list[str]]:
            Whether the object is a list of strings.
    """
    return isinstance(x, list) and all(isinstance(i, str) for i in x)
