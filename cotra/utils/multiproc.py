import contextlib
import multiprocessing
import threading
import traceback
import types
from typing import Any, Callable, Iterator, List, Optional, TextIO, TypeVar

import psutil

__all__ = [
    "safe_pool",
    "MultiprocessingFileWriter",
    "kill_proc_tree",
]

T = TypeVar('T')
R = TypeVar('R')


class Pool:
    r"""A wrapper over ``multiprocessing.Pool`` that uses single-threaded execution when :attr:`processes` is zero.
    """

    def __new__(cls, processes: int, *args, **kwargs):
        if processes > 0:
            return multiprocessing.Pool(processes, *args, **kwargs)
        return super().__new__(cls)  # return a mock Pool instance.

    def map(self, fn: Callable[[T], R], iterator: Iterator[T], *args, **kwargs) -> Iterator[R]:
        yield from map(fn, iterator)

    imap = map
    imap_unordered = map

    @staticmethod
    def _no_op(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        return types.MethodType(Pool._no_op, self)  # no-op for everything else


@contextlib.contextmanager
def safe_pool(processes: int, *args, closing: Optional[List[Any]] = None, **kwargs) \
        -> Iterator[multiprocessing.Pool]:
    r"""A wrapper over ``multiprocessing.Pool`` that gracefully handles exceptions.

    :param processes: The number of worker processes to run. A value of 0 means single threaded execution.
    :param closing: An optional list of objects to close at exit, routines to run at exit. For each element ``obj``:

        - If it is a callable, ``obj`` is called with no arguments.
        - If it has an ``close`` method, ``obj.close()`` is invoked.
        - Otherwise, it is ignored.

    :return: A context manager that can be used in a ``with`` statement.
    """
    if closing is not None and not isinstance(closing, list):
        raise ValueError("`closing` should either be `None` or a list")

    def close_fn():
        for obj in (closing or []):
            if callable(obj):
                obj()
            elif hasattr(obj, "close") and callable(getattr(obj, "close")):
                obj.close()

    pool = Pool(processes, *args, **kwargs)
    if processes == 0:
        # Don't swallow exceptions in the single-process case.
        yield pool  # type: ignore
        close_fn()
        return

    try:
        yield pool  # type: ignore
    except KeyboardInterrupt:
        print("Press Ctrl-C again to force terminate...")
        try:
            pool.join()
        except KeyboardInterrupt:
            pass
    except Exception as e:
        print(traceback.format_exc())
    finally:
        close_fn()
        if isinstance(pool, multiprocessing.pool.Pool):
            # Only required in multiprocessing scenario
            pool.close()
            pool.terminate()


class MultiprocessingFileWriter(TextIO):
    r"""A multiprocessing file writer that allows multiple processes to write to the same file. Order is not guaranteed.

    This is very similar to :class:`~ghcc.logging.MultiprocessingFileHandler`.
    """

    def __init__(self, path: str, mode: str = "a"):
        self._file = open("path")
        self._queue: 'multiprocessing.Queue[str]' = multiprocessing.Queue(-1)

        self._thread = threading.Thread(target=self._receive)
        self._thread.daemon = True
        self._thread.start()

    def __enter__(self) -> TextIO:
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._thread.join()
        self._file.close()

    def write(self, s: str):
        self._queue.put_nowait(s)

    def _receive(self):
        while True:
            try:
                record = self._queue.get()
                self._file.write(record)
            except EOFError:
                break


def kill_proc_tree(pid: int, including_parent: bool = True) -> None:
    r"""Kill entire process tree.
    """
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    gone, still_alive = psutil.wait_procs(children, timeout=5)
    if including_parent:
        parent.kill()
        parent.wait(5)
