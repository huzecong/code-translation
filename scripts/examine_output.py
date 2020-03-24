from typing import List, Tuple

import texar.torch as tx
from argtyped import *
from termcolor import colored
from tqdm import tqdm

import utils
from data import CodeDataSource
from utils.metric import DecodeMixin


class Args(Arguments):
    data_file: str = "data/processed/test.txt"  # original dataset
    ref_file: str = "outputs_no_curriculum/test.output.ref"  # reference sentences, should be the same as target
    hyp_file: str = "outputs_no_curriculum/test.output.hyp"  # hypotheses
    overlap_score_file: str = "data/processed/overlap_test.txt"  # overlap scores for test set
    output_file: str = "outputs_no_curriculum/test.annotated"


def read_lines(path: str) -> List[str]:
    with open(path) as f:
        lines = []
        for line in f:
            line = line.strip()
            if not line: continue
            lines.append(line)
    return lines


def read_pairs(path: str, decode: bool = False) -> Tuple[List[str], List[str]]:
    def _filter_fn(example) -> bool:
        src, tgt, _ = example
        src_len = src.count(" ") + 1  # count spaces instead of actually performing splitting
        tgt_len = tgt.count(" ") + 1
        lower, upper = 0.5, 3.0
        return (src_len + 1 <= 512 and  # account for EOS
                tgt_len + 1 <= 512 and
                lower <= src_len / tgt_len <= upper)

    lines = read_lines(path)
    src_data, tgt_data = [], []
    for line in lines:
        example = line.split(CodeDataSource.DELIMITER)
        if not _filter_fn(example):
            continue
        src, tgt, _ = example
        if decode:
            src = " ".join(DecodeMixin.spm_decode(src.split()))
            tgt = " ".join(DecodeMixin.spm_decode(tgt.split()))
        src_data.append(src)
        tgt_data.append(tgt)
    return src_data, tgt_data


def color_match(a: str, b: str) -> Tuple[str, str]:
    a, b = a.split(), b.split()
    a_set = set(a)
    f = utils.lcs_plan(a, b)
    i, j = len(a), len(b)
    while i > 0 and j > 0:
        if f[i - 1, j - 1] + 1 == f[i, j] and a[i - 1] == b[j - 1]:
            i, j = i - 1, j - 1
            a[i] = colored(a[i], "green")
            b[j] = colored(b[j], "green")
        elif f[i - 1, j] == f[i, j]:
            i = i - 1
        elif f[i, j - 1] == f[i, j]:
            j = j - 1
        else:
            assert False
    for i in range(len(b)):
        if b[i] not in a_set:
            b[i] = colored(b[i], "red")
    return " ".join(a), " ".join(b)


def main():
    args = Args()
    src_data, tgt_data = read_pairs(args.data_file, decode=True)
    ref_data = read_lines(args.ref_file)
    hyp_data = read_lines(args.hyp_file)
    overlap_scores = [float(x) for x in read_lines(args.overlap_score_file)]
    for idx, (tgt, ref) in enumerate(zip(tgt_data, ref_data)):
        if tgt != ref and "<unk>" not in ref:
            print(idx)
            print(tgt)
            print(ref)
            assert False
    # assert tgt_data == ref_data
    assert len(src_data) == len(tgt_data) == len(ref_data) == len(hyp_data) == len(overlap_scores)

    scores = []
    for ref, hyp in zip(ref_data, hyp_data):
        bleu4 = tx.evals.sentence_bleu([ref], hyp, max_order=4, smooth=True)
        bleu8 = tx.evals.sentence_bleu([ref], hyp, max_order=8, smooth=True)
        scores.append((bleu4, bleu8))

    indices = sorted(range(len(src_data)), key=lambda i: scores[i][0])
    with open(args.output_file, "w") as f:
        for idx in tqdm(indices):
            src = src_data[idx]
            ref = ref_data[idx]
            hyp = hyp_data[idx]
            overlap_score = overlap_scores[idx]
            ref, hyp = color_match(ref, hyp)
            bleu4, bleu8 = scores[idx]
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
