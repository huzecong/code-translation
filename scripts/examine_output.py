import pickle
from typing import List, Tuple, Iterator, overload, Any

import flutes
import numpy as np
import texar.torch as tx
from argtyped import *
from termcolor import colored
from tqdm import tqdm
from typing_extensions import Literal

import cotra
from cotra.utils.metric import DecodeMixin


class Args(Arguments):
    data_file: str = "data/processed/test.txt"  # original dataset
    hyp_names: str  # comma separated
    hyp_files: str  # hypotheses, comma separated
    overlap_score_files: str = "data/processed/overlap_test.txt"  # overlap scores for test set, comma separated
    output_file: str = "outputs/test.annotated"
    pickle_file: str = "outputs/test_output.pkl"


def read_lines(path: str, verbose: bool = True) -> Iterator[str]:
    with flutes.progress_open(path, verbose=verbose) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            yield line


@overload
def read_pairs(path: str, decode: bool = False,
               tuple_separator: str = "\1", token_separator: str = "\0",
               verbose: bool = True, return_additional_data: Literal[False] = False) \
        -> Tuple[List[str], List[str]]: ...


@overload
def read_pairs(path: str, decode: bool = False,
               tuple_separator: str = "\1", token_separator: str = "\0",
               verbose: bool = True, return_additional_data: Literal[True] = ...) \
        -> Tuple[List[str], List[str], List[Tuple[Any, ...]]]: ...


def read_pairs(path, decode=False, tuple_separator="\1", token_separator="\0",
               verbose=True, return_additional_data=False):
    def _filter_fn(src: str, tgt: str) -> bool:
        src_len = src.count(token_separator) + 1  # count spaces instead of actually performing splitting
        tgt_len = tgt.count(token_separator) + 1
        lower, upper = 0.5, 3.0
        return (src_len + 1 <= 512 and  # account for EOS
                tgt_len + 1 <= 512 and
                lower <= src_len / tgt_len <= upper)

    src_data, tgt_data = [], []
    additional_data = []
    for line in read_lines(path, verbose=verbose):
        example = line.split(tuple_separator)
        src, *_tgt, var_map, score, repo, sha  = example
        additional = var_map, score, repo, sha
        # DO NOT FILTER!
        # if not _filter_fn(src, tgt):
        #     continue
        tgt = tuple_separator.join(_tgt) if len(_tgt) != 1 else _tgt[0]
        if return_additional_data:
            additional_data.append(additional)
        if decode:
            src = " ".join(DecodeMixin.spm_decode(src.split(token_separator)))
            tgt = " ".join(DecodeMixin.spm_decode(tgt.split(token_separator)))
        else:
            src = " ".join(src.split(token_separator))
            tgt = " ".join(tgt.split(token_separator))
        src_data.append(src)
        tgt_data.append(tgt)
    if return_additional_data:
        return src_data, tgt_data, additional_data
    return src_data, tgt_data


def color_match(a: str, b: str) -> Tuple[str, str]:
    a, b = a.split(), b.split()
    a_set = set(a)
    plan_a, plan_b = cotra.utils.lcs_plan(a, b)
    for i in range(len(a)):
        if plan_a[i]:
            a[i] = colored(a[i], "green")
    for i in range(len(b)):
        if b[i] not in a_set:
            b[i] = colored(b[i], "red")
        elif plan_b[i]:
            b[i] = colored(b[i], "green")
    return " ".join(a), " ".join(b)


def compute_edit_score(src: str, tgt: str, hyp: str,
                       correct_copy_reward: float = 0.0, incorrect_copy_penalty: float = 0.0,
                       normalize: bool = True) -> float:
    r"""Compute the "edit score" given the source, ground-truth target, and hypothesis translation.

    A normal LCS is equivalent to setting ``correct_copy_reward`` to 1 and ``incorrect_copy_penalty`` to 0.

    :param src: The source sentence.
    :param tgt: The target sentence.
    :param hyp: The hypothesis sentence.
    :param correct_copy_reward: The reward term for each correctly copied token (i.e, when a target token is
        deemed as copied, and was correct in the hypothesis).
    :param incorrect_copy_penalty: The penalty term for each incorrectly copied token (i.e, when a target token is
        deemed as copied, but was incorrect in the hypothesis).
    :param normalize: If ``True``, the score is normalized to [0, 1].
    :return: The edit score.
    """
    if correct_copy_reward < 0.0:
        raise ValueError("`correct_copy_reward` must be non-negative")
    if incorrect_copy_penalty > 0.0:
        raise ValueError("`incorrect_copy_penalty` must be non-positive")

    src, tgt, hyp = src.split(), tgt.split(), hyp.split()
    plan_src, plan_tgt = cotra.utils.lcs_plan(src, tgt, prioritize_beginning=True)
    # Perform LCS DP on (tgt, hyp), penalizing positions with plan_tgt == True
    score = compute_edit_score_given_plan(tgt, hyp, plan_tgt, correct_copy_reward, incorrect_copy_penalty, normalize)
    return score


def compute_edit_score_given_plan(tgt: List[str], hyp: List[str], plan_tgt: List[bool],
                                  correct_copy_reward: float, incorrect_copy_penalty: float, normalize: bool) -> float:
    n, m = len(tgt), len(hyp)
    f = np.full((n + 1, m + 1), -np.inf, dtype=np.float32)
    f[0][0] = 0.0
    for i in range(n):
        for j in range(m):
            reward = correct_copy_reward if plan_tgt[i] else 1.0
            penalty = incorrect_copy_penalty if plan_tgt[i] else 0.0
            f[i + 1, j + 1] = max(
                f[i, j + 1],
                f[i + 1, j] + penalty,
                f[i, j] + (reward if tgt[i] == hyp[j] else penalty)
            )
    score = f[n, m]
    if normalize:
        n_copy = sum(plan_tgt)
        max_score = n_copy * correct_copy_reward + (n - n_copy)
        min_score = incorrect_copy_penalty * n_copy
        if max_score == min_score:
            score = 1.0  # in this case, source is equivalent to target
        else:
            score = (score - min_score) / (max_score - min_score)
    return score


def batch_compute_edit_score(src: str, tgt: str, hyp: str,
                             reward_and_penalty: List[Tuple[float, float]],
                             normalize: bool = True) -> List[float]:
    src, tgt, hyp = src.split(), tgt.split(), hyp.split()
    plan_src, plan_tgt = cotra.utils.lcs_plan(src, tgt, prioritize_beginning=True)
    scores = []
    for reward, penalty in reward_and_penalty:
        score = compute_edit_score_given_plan(tgt, hyp, plan_tgt, reward, penalty, normalize)
        scores.append(score)
    return scores


def main():
    args = Args()
    flutes.register_ipython_excepthook()
    src_data, tgt_data, additional_data = read_pairs(args.data_file, return_additional_data=True)
    names = args.hyp_names.split(",")
    hyp_paths = args.hyp_files.split(",")
    assert len(names) == len(hyp_paths)
    hyp_data = {}
    for name, hyp_path in zip(names, hyp_paths):
        hyp_data[name] = list(read_lines(hyp_path, verbose=False))
    overlap_paths = args.overlap_score_files.split(",")
    if len(overlap_paths) == 1:
        overlap_paths = overlap_paths * len(names)
    assert len(overlap_paths) == len(names)
    overlap_scores = {}
    for name, overlap_path in zip(names, overlap_paths):
        scores = [float(x) for x in read_lines(overlap_path, verbose=False)]
        overlap_scores[name] = scores
    # for idx, (tgt, ref) in enumerate(zip(tgt_data, ref_data)):
    #     if tgt != ref and "<unk>" not in ref:
    #         print(idx)
    #         print(tgt)
    #         print(ref)
    #         assert False
    # assert tgt_data == ref_data
    assert len(src_data) == len(tgt_data) == len(additional_data)
    assert len(overlap_scores) == len(names) == len(hyp_data)
    assert all(len(data) == len(src_data) == len(scores)
               for data, scores in zip(hyp_data.values(), overlap_scores.values()))

    with open(args.pickle_file, "wb") as f:
        obj = (names, src_data, tgt_data, hyp_data, overlap_scores, additional_data)
        pickle.dump(obj, f)

    return

    bleu_scores = []
    edit_scores = []
    for src, ref, hyp in zip(tqdm(src_data, desc="Computing scores"), ref_data, hyp_data):
        bleu4 = tx.evals.sentence_bleu([ref], hyp, max_order=4, smooth=True)
        bleu8 = tx.evals.sentence_bleu([ref], hyp, max_order=8, smooth=True)
        bleu_scores.append((bleu4, bleu8))
        edit_neu, edit_pos, edit_neg = batch_compute_edit_score(
            src, ref, hyp, reward_and_penalty=[(0.0, 0.0), (0.5, 0.0), (0.0, -0.5)])
        edit_scores.append((edit_neu, edit_pos, edit_neg))

    indices = sorted(range(len(src_data)), key=lambda i: bleu_scores[i][0])
    with open(args.output_file, "w") as f:
        for idx in tqdm(indices, desc="Writing output"):
            src = src_data[idx]
            ref = ref_data[idx]
            hyp = hyp_data[idx]
            overlap_score = overlap_scores[idx]
            ref, hyp = color_match(ref, hyp)
            bleu4, bleu8 = bleu_scores[idx]
            f.write(colored(f"Example {idx} (BLEU4 = {bleu4:.2f}, BLEU8 = {bleu8:.2f}), "
                            f"overlap with train = ", "yellow") +
                    colored(f"{overlap_score:.3f}", "red" if overlap_score > 0.8 else "yellow") + "\n")
            f.write(colored("Source:     ", "blue") + src + "\n")
            f.write(colored("Target:     ", "blue") + ref + "\n")
            f.write(colored("Prediction: ", "blue") + hyp + "\n")
            f.write("\n")

    bleu4 = tx.evals.corpus_bleu([[tgt] for tgt in tgt_data], hyp_data, max_order=4)
    bleu8 = tx.evals.corpus_bleu([[tgt] for tgt in tgt_data], hyp_data, max_order=8)
    print(f"BLEU4 = {bleu4:.2f}, BLEU8 = {bleu8:.2f}")


if __name__ == '__main__':
    main()
    # a, b = color_match("a a b c d e", "a b c d f e")
    # print(a)
    # print(b)
