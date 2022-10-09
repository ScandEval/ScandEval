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


class Tokenizer(Protocol):
    model_max_length: int
    max_model_input_sizes: Dict[str, int]

    def __call__(self, *args, **kwargs) -> Dict:
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
    def __call__(self, *args, **kwargs) -> Dict:
        ...
