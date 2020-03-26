from typing import List, TypeVar, Tuple
import numpy as np

__all__ = [
    "lcs",
    "lcs_plan",
    "lcs_matrix",
]

T = TypeVar('T')


def lcs(a: List[T], b: List[T]) -> int:
    return lcs_matrix(a, b)[len(a), len(b)]


def lcs_plan(a: List[T], b: List[T], prioritize_beginning: bool = False) -> Tuple[List[bool], List[bool]]:
    r"""Compute the edit distance between two lists, and return the plan: which tokens in the two lists are kept as is.

    :param prioritize_beginning: If ``True``, when there are multiple plans, prefer the plan with more kept-as-is tokens
        at the beginning. Defaults to ``False``, which prefers more kept-as-is tokens at the end.
    :return: A pair of boolean lists.
    """
    if prioritize_beginning:
        a, b = list(reversed(a)), list(reversed(b))
    f = lcs_matrix(a, b)
    plan_a, plan_b = [False] * len(a), [False] * len(b)
    i, j = len(a), len(b)
    while i > 0 and j > 0:
        if f[i - 1, j - 1] + 1 == f[i, j] and a[i - 1] == b[j - 1]:
            i, j = i - 1, j - 1
            a[i] = True
            b[j] = True
        elif f[i - 1, j] == f[i, j]:
            i = i - 1
        elif f[i, j - 1] == f[i, j]:
            j = j - 1
        else:
            assert False
    if prioritize_beginning:
        plan_a, plan_b = list(reversed(plan_a)), list(reversed(plan_b))
    return plan_a, plan_b


def lcs_matrix(a: List[T], b: List[T]) -> np.ndarray:
    r"""Compute the edit distance between two lists.

    :return: The DP cost matrix.
    """
    n, m = len(a), len(b)
    f = np.zeros((n + 1, m + 1), dtype=np.int16)
    for i in range(n):
        for j in range(m):
            f[i + 1, j + 1] = max(f[i, j + 1], f[i + 1, j])
            if a[i] == b[j]:
                f[i + 1, j + 1] = max(f[i + 1, j + 1], f[i, j] + 1)
    return f
