"""Protocols used in the project."""

from typing import Dict, Iterator, List, Protocol, Tuple, Union

from torch import FloatTensor, Tensor
from torch.nn.parameter import Parameter


class Config(Protocol):
    id2label: List[str]
    label2id: Dict[str, int]

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "Config":
        ...


class TokenizedOutputs(Protocol):
    input_ids: List[List[int]]

    def word_ids(self, *args, **kwargs) -> List[Union[int, None]]:
        ...

    def __setitem__(self, *args, **kwargs) -> None:
        ...


class Tokenizer(Protocol):
    model_max_length: int
    max_model_input_sizes: Dict[str, int]
    special_tokens_map: Dict[str, str]

    def __call__(self, *args, **kwargs) -> TokenizedOutputs:
        ...

    def convert_ids_to_tokens(self, *args, **kwargs) -> List[Union[str, None]]:
        ...


class ModelOutput(Protocol):
    logits: FloatTensor


class Model(Protocol):
    config: Config

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Union[Tuple[Tensor], ModelOutput]:
        ...

    def parameters(self, *args, **kwargs) -> Iterator[Parameter]:
        ...

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> Union[Tuple["Model", Dict], "Model"]:
        ...


class DataCollator(Protocol):
    def __call__(self, *args) -> Dict:
        ...
