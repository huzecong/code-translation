import itertools
import math
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

import flutes
import sentencepiece as spm
from argtyped import *
from tqdm import tqdm


class Args(Arguments):
    data_dir: str = "mt_data/"
    test_split_size: Optional[int] = 5000
    test_split_portion: Optional[float] = None
    max_test_repos: int = 50
    output_path: str = "data/processed/"
    vocab_size: int = 32000
    n_procs: int = 4
    block_size: int = 1000  # 1000 examples per reduce (collate) op
    random_seed: int = 19260817

    def __init__(self):
        super().__init__()
        if not ((self.test_split_size is None) ^ (self.test_split_portion is None)):
            raise ValueError("Exactly one of 'test_split_size' and 'test_split_portion' must be None")


TOKEN_SEP = "\0"
TUPLE_SEP = "\1"


class InputData(NamedTuple):
    decompiled_code: str
    original_code: str
    repo: str  # "owner/name"
    sha: str


class OutputData(NamedTuple):
    decompiled_code: str
    original_code: str
    score: float
    repo: str  # "owner/name"
    sha: str


def read_data(data_dir: str) -> Iterator[InputData]:
    files = [file for file in Path(data_dir).iterdir() if file.name.startswith("data_text")]
    for file in tqdm(files, desc="Reading file"):
        with file.open("rb") as f:
            data = pickle.load(f)
        for ex in data:  # `cotra.data.PickledExample`
            yield InputData(*ex)


def tokenize(sentence: str) -> Iterator[str]:
    for tok in sentence.split(TOKEN_SEP):
        if tok.startswith('"'):
            yield '"'
            yield from tok[1:-1].split()
            yield '"'
        else:
            yield tok


def count_words(sentences: List[str]) -> 'Counter[str]':
    counter = Counter()
    for sent in sentences:
        counter.update(tokenize(sent))
    return counter


def compute_score(word_scores: Dict[str, float], example: InputData) -> float:
    # Compute score for source sentence.
    score = sum(word_scores.get(w, 0.0) for w in tokenize(example.decompiled_code))
    return score


class ComputeScore:
    @staticmethod
    def init(word_scores: Dict[str, float]) -> None:
        import inspect
        local_vars = inspect.stack()[1].frame.f_locals  # init -> worker
        # print(local_vars.keys())
        local_vars['word_scores'] = word_scores

    @staticmethod
    def compute_score(code: str) -> float:
        import inspect
        local_vars = inspect.stack()[2].frame.f_locals  # compute_score -> mapper -> worker
        # print(local_vars.keys())
        word_scores = local_vars['word_scores']
        score = sum(word_scores.get(w, 0.0) for w in tokenize(code))
        return score


class EncodeSPM:
    @staticmethod
    def init(path: str) -> None:
        import inspect
        local_vals = inspect.stack()[1].frame.f_locals  # init -> worker
        sp = spm.SentencePieceProcessor()
        sp.Load(path)
        local_vals['sp'] = sp

    @staticmethod
    def encode_sentence(sp: spm.SentencePieceProcessor, sent: str) -> str:
        return TOKEN_SEP.join(subtok for token in sent.split(TOKEN_SEP) for subtok in sp.EncodeAsPieces(token))

    @staticmethod
    def encode_spm(example: Tuple[str, str]) -> Tuple[str, str]:
        import inspect
        local_vals = inspect.stack()[2].frame.f_locals  # init -> worker
        sp: spm.SentencePieceProcessor = local_vals['sp']
        return (EncodeSPM.encode_sentence(sp, example[0]),
                EncodeSPM.encode_sentence(sp, example[1]))


def sample_sets(sets: List[Tuple[str, int]], max_size: int) -> List[str]:
    r"""A brute-force algorithm to randomly pick several sets such that the sum of their sizes does not exceed
    ``max_size``.

    Due to limit on ``max_size``, this algorithm is not uniform: The probability of a set being chosen is not
    exactly proportional to its size. The probability is

    :param sets: List of sets, described by tuple (name, size).
    :param max_size: Maximum total size of chosen sets.
    :return: The names of chosen sets.
    """
    chosen: Set[str] = set()
    remaining = max_size
    while remaining > 0:
        candidates = list(filter(lambda xs: xs[0] not in chosen and xs[1] <= remaining, sets))
        if len(candidates) == 0:
            break
        name, size = random.choices(candidates, weights=map(lambda xs: xs[1], candidates), k=1)[0]
        remaining -= size
        assert remaining >= 0
        chosen.add(name)
    return list(chosen)


def main():
    args = Args()
    random.seed(args.random_seed)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = flutes.LazyList(read_data(args.data_dir))

    @flutes.cache(output_dir / "scores.pkl", name="competency scores")
    def compute_scores():
        # Gather word counts and compute "competency" scores for each example, for use in curriculum learning.
        word_counter = Counter()
        data_chunks = flutes.chunk(map(lambda xs: xs[0], data), args.block_size)  # use only the source sentence
        with flutes.safe_pool(args.n_procs) as pool:
            for counter in pool.imap_unordered(count_words, data_chunks):
                word_counter.update(counter)
        total_words = sum(word_counter.values())
        word_scores = {w: -math.log(c / total_words) for w, c in word_counter.items()}
        with flutes.safe_pool(args.n_procs, initializer=ComputeScore.init, initargs=(word_scores,)) as pool:
            # scores = list(pool.imap(functools.partial(compute_score, word_scores),
            #                         tqdm(data, desc="Computing scores"), chunksize=args.block_size))
            scores = list(pool.imap(
                ComputeScore.compute_score,
                map(lambda ex: ex.decompiled_code, tqdm(data, desc="Computing scores")), chunksize=args.block_size))
        return scores

    scores = compute_scores()

    # Generate data splits.
    test_size = args.test_split_size or int(len(data) * args.test_split_portion)
    # Test set 1: Repositories excluded in training set.
    data_by_repo = defaultdict(list)
    for idx, ex in enumerate(data):
        data_by_repo[ex.repo].append(idx)
    repo_names = list(data_by_repo.keys())

    def create_excluded_split(target_size: int) -> Tuple[List[str], List[int]]:
        # ([name], [index])
        while True:
            repo_count = random.randint(1, args.max_test_repos)
            chosen_repos = random.choices(repo_names, k=repo_count)
            sample_size = sum(len(data_by_repo[name]) for name in chosen_repos)
            if 0.8 * target_size <= sample_size <= 1.1 * target_size:
                # Keep sampling until we get something with appropriate size.
                break
        split_indices = list(itertools.chain.from_iterable(data_by_repo[name] for name in chosen_repos))
        return chosen_repos, split_indices

    dev_repos, dev_excluded_split = create_excluded_split(test_size)
    for repo_name in dev_repos:
        del data_by_repo[repo_name]
    test_repos, test_excluded_split = create_excluded_split(test_size)
    excluded_indices = set(dev_excluded_split + test_excluded_split)

    # Test set 2: Randomly sampled functions among the rest of repositories.
    all_indices = [idx for idx in range(len(data)) if idx not in excluded_indices]
    random.shuffle(all_indices)
    if len(data) <= 3 * test_size:
        raise ValueError(f"Dataset size ({len(data)}) too small to use test size {test_size}")

    test_split = all_indices[-test_size:]
    dev_split = all_indices[-(2 * test_size):-test_size]
    train_split = all_indices[:-(2 * test_size)]
    splits = {
        "train": train_split,
        "valid": dev_split,
        "test": test_split,
        "valid_exclude": dev_excluded_split,
        "test_exclude": test_excluded_split,
    }
    with (output_dir / "split_indices.pkl").open("wb") as f:
        pickle.dump(splits, f)

    # Write out training text and train SentencePiece model.
    train_text_path = output_dir / "train.txt"
    with train_text_path.open("w") as f:
        for idx in tqdm(train_split, desc="Writing training text"):
            f.write(data[idx].decompiled_code.replace(TOKEN_SEP, " ") + "\n")
            f.write(data[idx].original_code.replace(TOKEN_SEP, " ") + "\n")
    spm_train_args = {
        "input": train_text_path,
        "model_prefix": output_dir / "vocab",
        "vocab_size": args.vocab_size,
    }
    spm.SentencePieceTrainer.Train(" ".join(f"--{name}={str(value)}" for name, value in spm_train_args.items()))

    # Encode all sentences with the trained SP model.
    with flutes.safe_pool(args.n_procs, initializer=EncodeSPM.init,
                          initargs=(str(output_dir / "vocab.model"),)) as pool:
        processed_code = list(pool.imap(
            EncodeSPM.encode_spm,
            map(lambda ex: (ex.decompiled_code, ex.original_code),
                tqdm(data, desc="Computing scores")), chunksize=args.block_size))

    for key, indices in splits.items():
        with (output_dir / f"{key}.txt").open("w") as f:
            for idx in tqdm(indices, desc=f"Writing {key} set", leave=False):
                src, tgt = processed_code[idx]
                output = TUPLE_SEP.join((src, tgt, str(scores[idx]), data[idx].repo, data[idx].sha))
                f.write(output)
                f.write("\n")
        print(f"{key.capitalize()} set written")


if __name__ == '__main__':
    main()
