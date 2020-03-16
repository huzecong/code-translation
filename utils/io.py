import os
from typing import TextIO

import tqdm

__all__ = [
    "FileProgress",
]


class FileProgress:
    def __new__(cls, f: TextIO, *, verbose: bool = True, **kwargs):
        if not verbose:
            return f
        return super().__new__(cls)

    def __init__(self, f: TextIO, *, encoding: str = 'utf-8', **kwargs):
        self.f = f
        self.encoding = encoding
        self.file_size = os.fstat(f.fileno()).st_size
        self.progress_bar = tqdm.tqdm(total=self.file_size, **kwargs)
        self.size_read = 0
        self._next_tick = 1
        self._next_size = self.file_size // 100
        self._accum_size = 0

    def __iter__(self):
        return self

    def _update(self, line: str):
        size = len(line)  # `line.decode(self.encoding)` would be more precise, but who cares?
        self._accum_size += size
        if self.size_read + self._accum_size >= self._next_size:  # do a bulk update
            self.progress_bar.update(self._accum_size, )  # type: ignore
            self.size_read += self._accum_size
            self._accum_size = 0
            while self.size_read >= self._next_size:
                self._next_tick += 1
                self._next_size = self.file_size * self._next_tick // 100

    def __next__(self) -> str:
        line = next(self.f)
        self._update(line)
        return line

    def readline(self, *args) -> str:
        line = self.f.readline(*args)
        self._update(line)
        return line

    def __enter__(self):
        self.f.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress_bar.close()
        self.f.__exit__(exc_type, exc_val, exc_tb)
