from typing import List, TypeVar
import numpy as np

__all__ = [
    "lcs",
    "lcs_plan",
]

T = TypeVar('T')


def lcs(a: List[T], b: List[T]) -> int:
    return lcs_plan(a, b)[len(a), len(b)]


def lcs_plan(a: List[T], b: List[T]) -> np.ndarray:
    n, m = len(a), len(b)
    f = np.zeros((n + 1, m + 1), dtype=np.int16)
    for i in range(n):
        for j in range(m):
            f[i + 1, j + 1] = max(f[i, j + 1], f[i + 1, j])
            if a[i] == b[j]:
                f[i + 1, j + 1] = max(f[i + 1, j + 1], f[i, j] + 1)
    return f
