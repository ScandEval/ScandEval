"""Types used throughout the project."""

from typing import Dict, List, Union

SCORE_DICT = Dict[str, Union[Dict[str, float], Dict[str, List[Dict[str, float]]]]]
