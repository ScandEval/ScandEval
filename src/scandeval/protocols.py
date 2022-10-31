"""Protocols used in the project."""

from typing import Any, Dict, Iterator, List, Protocol, Tuple, Union

from torch import FloatTensor, Tensor
from torch.nn.parameter import Parameter


class Config(Protocol):
    id2label: List[str]
    label2id: Dict[str, int]
    vocab_size: int

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "Config":
        ...


class TokenizedOutputs(Protocol):
    input_ids: List[List[int]]
    offset_mapping: List[List[Tuple[int, int]]]
    start_positions: List[int]
    end_positions: List[int]
    id: List[int]

    def word_ids(self, *args, **kwargs) -> List[Union[int, None]]:
        ...

    def sequence_ids(self, *args, **kwargs) -> List[Union[int, None]]:
        ...

    def pop(self, *args, **kwargs) -> Any:
        ...

    def __setitem__(self, *args, **kwargs) -> None:
        ...


class Tokenizer(Protocol):
    model_max_length: int
    max_model_input_sizes: Dict[str, int]
    special_tokens_map: Dict[str, str]
    cls_token: str
    bos_token: str
    sep_token: str
    eos_token: str
    cls_token_id: int
    bos_token_id: int
    sep_token_id: int
    eos_token_id: int
    vocab_size: int

    def encode(self, *args, **kwargs) -> List[int]:
        ...

    def __call__(self, *args, **kwargs) -> TokenizedOutputs:
        ...

    def convert_ids_to_tokens(self, *args, **kwargs) -> List[Union[str, None]]:
        ...

    def convert_tokens_to_ids(self, *args, **kwargs) -> List[Union[int, None]]:
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
