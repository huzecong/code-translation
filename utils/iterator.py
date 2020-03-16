import weakref
from typing import Generic, Iterable, Iterator, List, TypeVar, Union, Optional

__all__ = [
    "LazyList", "chunk",
]

T = TypeVar('T')


class LazyList(Generic[T]):
    class LazyListIterator:
        def __init__(self, lst: 'LazyList[T]'):
            self.list = weakref.ref(lst)
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            try:
                obj = self.list()[self.index]
            except IndexError:
                raise StopIteration
            self.index += 1
            return obj

    def __init__(self, iterator: Iterable[T]):
        self.iter = iter(iterator)
        self.exhausted = False
        self.list = []

    def __iter__(self):
        if self.exhausted:
            return iter(self.list)
        return self.LazyListIterator(self)

    def _fetch_until(self, idx: Optional[int]) -> None:
        if self.exhausted:
            return
        try:
            while idx is None or len(self.list) <= idx:
                self.list.append(next(self.iter))
        except StopIteration:
            self.exhausted = True
            del self.iter

    def __getitem__(self, idx: Union[int, slice]) -> T:
        if isinstance(idx, slice):
            self._fetch_until(idx.stop)
        else:
            self._fetch_until(idx)
        return self.list[idx]

    def __len__(self):
        self._fetch_until(None)
        return len(self.list)


def chunk(iterable: Iterable[T], n: int) -> Iterator[List[T]]:
    assert n > 0
    group = []
    for x in iterable:
        group.append(x)
        if len(group) == n:
            yield group
            group = []
    if len(group) > 0:
        yield group
