# type: ignore
"""A module to override stupid Texar functionalities"""

__all__ = [
    "write_log",
]

# Monkey-patch `RandomSampler` for our use case specifically.

from texar.torch.data.data.sampler import RandomSampler

RandomSampler._iterator_unknown_size = lambda self: self._iterator_given_size(None)

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
