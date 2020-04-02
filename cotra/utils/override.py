"""A module to override stupid Texar functionalities"""

__all__ = [
    "write_log",
]

# Monkey-patch `RandomSampler` for our use case specifically.
from texar.torch.data.data.sampler import RandomSampler

RandomSampler._iterator_unknown_size = lambda self: self._iterator_given_size(None)

# Make `Executor` a singleton class so we can have a globally available `write_log` method.
from texar.torch.run.executor import Executor
from texar.torch.run.condition import Event
import texar.torch.run.executor_utils as executor_utils

if not hasattr(Executor, "__instance__"):
    Executor.__instance__ = None
    Executor__init = Executor.__init__


    def __init__(self, *args, **kwargs):
        if getattr(Executor, "__instance__", None) is not None:
            raise ValueError("An instance of `Executor` already exists")

        Executor.__instance__ = self
        Executor__init(self, *args, **kwargs)


    def _validate_loop(self, iterator) -> None:
        for batch in iterator:
            self._fire_event(Event.ValidationIteration, False)
            return_dict = self._validate_step(batch)

            self._valid_tracker.add(len(batch))
            executor_utils.update_metrics(return_dict, batch, self.valid_metrics)

            self._fire_event(Event.ValidationIteration, True)


    Executor.__init__ = __init__
    Executor._validate_loop = _validate_loop


def write_log(log_str: str, *, mode: str = "info",
              skip_tty: bool = False, skip_non_tty: bool = False) -> None:
    r"""The `Executor.write_log` function, applied on the singleton instance."""
    Executor.__instance__.write_log(log_str, mode=mode, skip_tty=skip_tty, skip_non_tty=skip_non_tty)
