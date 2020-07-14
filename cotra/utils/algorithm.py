from typing import List, Optional, Sequence, Tuple, TypeVar

import numpy as np

__all__ = [
    "lcs",
    "lcs_plan",
    "lcs_matrix",
    "edit_distance",
    "edit_distance_matrix",
]

T = TypeVar('T')


def lcs(a: Sequence[T], b: Sequence[T]) -> int:
    r"""Compute the longest common subsequence (LCS) between two lists.

    :return: Length of the LCS.
    """
    return lcs_matrix(a, b)[len(a), len(b)]


def lcs_plan(a: Sequence[T], b: Sequence[T], prioritize_beginning: bool = False) -> Tuple[List[bool], List[bool]]:
    r"""Compute the longest common subsequence between two lists, and return the plan: which tokens in the two lists are
    kept as is.

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
            plan_a[i] = True
            plan_b[j] = True
        elif f[i - 1, j] == f[i, j]:
            i = i - 1
        elif f[i, j - 1] == f[i, j]:
            j = j - 1
        else:
            assert False
    if prioritize_beginning:
        plan_a, plan_b = list(reversed(plan_a)), list(reversed(plan_b))
    return plan_a, plan_b


def lcs_matrix(a: Sequence[T], b: Sequence[T]) -> np.ndarray:
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


Cost = TypeVar('Cost', int, float)


def edit_distance_matrix(a: Sequence[T], b: Sequence[T], insert: Cost = 1, remove: Cost = 1,
                         replace: Cost = 1, swap: Optional[Cost] = None) -> np.ndarray:
    r"""Compute the edit distance between two lists. Supports settings custom costs for inserting, removing, replacing,
    and swapping.

    :param insert: Cost for inserting a token in list. Defaults to 1.
    :param remove: Cost for removing a token in list. Defaults to 1.
    :param replace: Cost for replacing a token in list. Defaults to 1.
    :param swap: Cost for swapping two adjacent tokens in list. Defaults to ``None``, which means it's forbidden.
    :return:
    """
    n, m = len(a), len(b)
    dtype = np.int32 if isinstance(insert, int) else np.float32
    f = np.empty((n + 1, m + 1), dtype=dtype)
    f[0, :] = np.arange(m + 1, dtype=dtype)
    f[:, 0] = np.arange(n + 1, dtype=dtype)
    for i in range(n):
        for j in range(m):
            f[i + 1, j + 1] = min(
                f[i, j + 1] + insert,
                f[i + 1, j] + remove,
                f[i, j] + int(a[i] != b[j]) * replace,
            )
            if swap is not None and i > 0 and j > 0 and a[i - 1] == b[j] and a[i] == b[j - 1]:
                f[i + 1, j + 1] = min(f[i + 1, j + 1], f[i - 1, j - 1] + swap)
    return f


def edit_distance(a: Sequence[T], b: Sequence[T], insert: Cost = 1, remove: Cost = 1,
                  replace: Cost = 1, swap: Optional[Cost] = None) -> int:
    return edit_distance_matrix(a, b, insert, remove, replace, swap)[len(a), len(b)]
