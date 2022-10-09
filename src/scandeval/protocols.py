"""Protocols used in the project."""

from os import PathLike
from typing import Callable, Dict, Iterator, List, Protocol, Tuple, Union

from datasets.features.features import Features
from torch import FloatTensor, LongTensor, Tensor
from torch.nn.parameter import Parameter


class Config(Protocol):
    id2label: List[str]
    label2id: Dict[str, int]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, PathLike, None],
        **kwargs,
    ) -> "Config":
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
        # input_ids: Union[LongTensor, None],
        # attention_mask: Union[FloatTensor, None],
        # token_type_ids: Union[LongTensor, None],
        # position_ids: Union[LongTensor, None],
        # head_mask: Union[FloatTensor, None],
        # inputs_embeds: Union[FloatTensor, None],
        # labels: Union[LongTensor, None],
        # output_attentions: Union[bool, None],
        # output_hidden_states: Union[bool, None],
        # return_dict: Union[bool, None],
    ) -> Union[Tuple[Tensor], ModelOutput]:
        ...

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        ...

    @classmethod
    def from_pretrained(
        cls,
        *args,
        **kwargs,
        # pretrained_model_name_or_path: Union[str, PathLike, None],
        # *model_args,
        # **kwargs,
    ) -> Union[Tuple["Model", Dict], "Model"]:
        ...


class DataCollator(Protocol):
    def __call__(self, features: List[Dict]) -> Dict:
        ...
