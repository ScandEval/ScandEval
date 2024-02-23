"""Types used throughout the project."""

from typing import Any, TypeGuard

import numpy as np

ScoreDict = dict[str, dict[str, float] | dict[str, list[dict[str, float]]]]
Predictions = np.ndarray | list[str] | list[list[str]]
Labels = np.ndarray | list[str] | list[list[str]]


def is_list_of_int(x: Any) -> TypeGuard[list[int]]:
    """Check if an object is a list of integers.

    Args:
        x:
            The object to check.

    Returns:
        Whether the object is a list of integers.
    """
    return isinstance(x, list) and all(isinstance(i, int) for i in x)


def is_list_of_list_of_int(x: Any) -> TypeGuard[list[list[int]]]:
    """Check if an object is a list of list of integers.

    Args:
        x:
            The object to check.

    Returns:
        Whether the object is a list of list of integers.
    """
    return (
        isinstance(x, list)
        and all(isinstance(i, list) for i in x)
        and all(isinstance(j, int) for i in x for j in i)
    )


def is_list_of_str(x: Any) -> TypeGuard[list[str]]:
    """Check if an object is a list of integers.

    Args:
        x:
            The object to check.

    Returns:
        Whether the object is a list of strings.
    """
    return isinstance(x, list) and all(isinstance(i, str) for i in x)
