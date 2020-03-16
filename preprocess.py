import functools
import itertools
import math
import os
import random
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import tqdm
from argtyped import *

import utils


class Args(Arguments):
    src_path: str = "data/all.src"
    tgt_path: str = "data/all.tgt"
    vocab_path: str = "data/vocab.vocab"
    test_split_size: Optional[int] = 5000
    test_split_portion: Optional[float] = None
    output_path: str = "data/processed/"
    n_procs: int = 4
    block_size: int = 1000  # 1000 examples per reduce (collate) op
    random_seed: int = 19260817

    def __init__(self):
        super().__init__()
        if not ((self.test_split_size is None) ^ (self.test_split_portion is None)):
            raise ValueError("Exactly one of 'test_split_size' and 'test_split_portion' must be None")


Sentence = List[str]
InputData = Tuple[str, str]
OutputData = Tuple[str, str, float]


def read_paired_data(src_file: str, tgt_file: str) -> Iterator[InputData]:
    with utils.FileProgress(open(src_file), desc="Reading file") as f_src, open(tgt_file) as f_tgt:
        for src, tgt in itertools.zip_longest(f_src, f_tgt):
            if src is None or tgt is None:
                raise ValueError("Source and target files have different lengths")
            yield src.strip(), tgt.strip()


def count_words(sentences: List[str]) -> 'Counter[str]':
    counter = Counter()
    for sent in sentences:
        counter.update(sent.split())
    return counter


def compute_score(word_scores: Dict[str, float], example: InputData) -> float:
    src = example[0].split()
    # tgt = example[1].split()
    score = sum(word_scores.get(w, 0.0) for w in src)  # compute score for source sentence
    return score


def write_output_data(data: Iterable[Tuple[InputData, float]], path: str):
    sep = " ▁|SEP|▁ "
    with open(path, "w") as f:
        for (src, tgt), score in data:
            f.write(sep.join([src, tgt, str(score)]) + "\n")


def main():
    args = Args()
    random.seed(args.random_seed)
    os.makedirs(args.output_path, exist_ok=True)
    data = utils.LazyList(read_paired_data(args.src_path, args.tgt_path))

    word_counter = Counter()
    data_chunks = utils.chunk(map(lambda xs: xs[0], data), args.block_size)  # use only the source sentence
    with utils.safe_pool(args.n_procs) as pool:
        for counter in pool.imap_unordered(count_words, data_chunks):
            word_counter.update(counter)

    total_words = sum(word_counter.values())
    word_scores = {w: -math.log(c / total_words) for w, c in word_counter.items()}
    with utils.safe_pool(args.n_procs) as pool:
        scores = list(pool.imap(functools.partial(compute_score, word_scores),
                                tqdm.tqdm(data, desc="Computing scores"), chunksize=args.block_size))

    all_indices = list(range(len(data)))
    random.shuffle(all_indices)
    test_size = args.test_split_size or int(len(data) * args.test_split_portion)
    if len(data) <= 3 * test_size:
        raise ValueError(f"Dataset size ({len(data)}) too small to use test size {test_size}")

    test_split = all_indices[-test_size:]
    dev_split = all_indices[-(2 * test_size):-test_size]
    train_split = all_indices[:-(2 * test_size)]

    for key, indices in {"train": train_split, "dev": dev_split, "test": test_split}.items():
        write_output_data(((data[idx], scores[idx]) for idx in indices),
                          os.path.join(args.output_path, f"{key}.txt"))
        print(f"{key.capitalize()} set written")


if __name__ == '__main__':
    main()
