# type: ignore
"""A module to override stupid Texar functionalities"""
from typing import TypeVar

import torch

__all__ = [
    "write_log",
]

T = TypeVar('T')

# Monkey-patch `RandomSampler` for our use case specifically.
from texar.torch.data.data.sampler import RandomSampler


def _RandomSampler_iterator_given_size(self, size: int):
    order = torch.randperm(size).tolist()
    for idx in order:
        self._data._prefetch_source(idx)
        yield idx


def _RandomSampler_iterator_unknown_size(self):
    return self._iterator_given_size(len(self._data))


RandomSampler._iterator_given_size = _RandomSampler_iterator_given_size
RandomSampler._iterator_unknown_size = _RandomSampler_iterator_unknown_size

# Prevent `_CachedDataSource` from deleting stuff.
import texar.torch.data.data.data_base


class _PatchedCachedDataSource(texar.torch.data.data.data_base._CachedDataSource[T]):
    def __init__(self, data_source, erase_after_access: bool = False):
        super().__init__(data_source, erase_after_access=False)  # False whatever


texar.torch.data.data.data_base._CachedDataSource = _PatchedCachedDataSource

# Make `Executor` a singleton class so we can have a globally available `write_log` method.
from texar.torch.run.executor import Executor

if not hasattr(Executor, "__instance__"):
    Executor.__instance__ = None
    Executor__init = Executor.__init__


    def __init__(self, *args, **kwargs):
        if getattr(Executor, "__instance__", None) is not None:
            raise ValueError("An instance of `Executor` already exists")

        Executor.__instance__ = self
        Executor__init(self, *args, **kwargs)


    Executor.__init__ = __init__


def write_log(log_str: str, *, mode: str = "info",
              skip_tty: bool = False, skip_non_tty: bool = False) -> None:
    r"""The `Executor.write_log` function, applied on the singleton instance."""
    Executor.__instance__.write_log(log_str, mode=mode, skip_tty=skip_tty, skip_non_tty=skip_non_tty)
