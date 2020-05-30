import math
import os
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

import flutes
import sentencepiece as spm
from argtyped import *
from tqdm import tqdm

from cotra.data import InputData as OutputData


class Args(Arguments):
    data_dir: str = "mt_data_varnames/"
    test_split_size: Optional[int] = 3000
    test_split_portion: Optional[float] = None
    extra_train_portion: float = 2.0 / 3  # how much of dev/test sets should be split out into an extra fine-tuning set
    max_test_repos: int = 50
    output_path: str = "data/processed_varnames/"
    vocab_size: int = 32000
    n_procs: int = 4
    block_size: int = 1000  # 1000 examples per reduce (collate) op
    random_seed: int = 19260817
    encode_spm: Switch = False
    pdb: Switch = False

    def __init__(self):
        super().__init__()
        if not ((self.test_split_size is None) ^ (self.test_split_portion is None)):
            raise ValueError("Exactly one of 'test_split_size' and 'test_split_portion' must be None")


TOKEN_SEP = "\0"
TUPLE_SEP = "\1"


class InputData(NamedTuple):
    decompiled_code: str
    original_code: str
    var_names: Dict[str, Tuple[str, str]]
    repo: str  # "owner/name"
    sha: str




def read_data(data_dir: str) -> Iterator[InputData]:
    files = [file for file in sorted(Path(data_dir).iterdir())
             if file.name.startswith("data_") and file.suffix == ".pkl"]
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


class CountWordsState(flutes.PoolState):
    def __init__(self):
        self.counter = Counter()

    def count_words(self, sentence: str) -> None:
        self.counter.update(tokenize(sentence))


class ComputeScoreState(flutes.PoolState):
    def __init__(self, word_scores: Dict[str, float]) -> None:
        self.word_scores = word_scores

    def compute_score(self, code: str) -> float:
        word_scores = self.word_scores
        return sum(word_scores.get(w, 0.0) for w in tokenize(code))


class EncodeSPMState(flutes.PoolState):
    def __init__(self, path: Path) -> None:
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(path)

    def encode_sentence(self, sent: str) -> str:
        return TOKEN_SEP.join(subtok for token in sent.split(TOKEN_SEP)
                              for subtok in self.sp.EncodeAsPieces(token))

    def encode_spm(self, example: Tuple[str, str]) -> Tuple[str, str]:
        return (self.encode_sentence(example[0]),
                self.encode_sentence(example[1]))


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
    print(args.to_string())
    random.seed(args.random_seed)
    if args.pdb:
        flutes.register_ipython_excepthook()

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = flutes.LazyList(read_data(args.data_dir))

    @flutes.cache(output_dir / "scores.pkl", name="competency scores")
    def compute_scores():
        # Gather word counts and compute "competency" scores for each example, for use in curriculum learning.
        with flutes.safe_pool(args.n_procs, state_class=CountWordsState) as pool:
            for _ in pool.imap_unordered(
                    CountWordsState.count_words,  # use only the source sentence
                    map(lambda ex: ex.decompiled_code, tqdm(data, desc="Counting words")), chunksize=args.block_size):
                pass
            word_counter = Counter()
            for state in pool.get_states():
                word_counter.update(state.counter)
        total_words = sum(word_counter.values())
        word_scores = {w: -math.log(c / total_words) for w, c in word_counter.items()}
        with flutes.safe_pool(args.n_procs, state_class=ComputeScoreState, init_args=(word_scores,)) as pool:
            scores = list(pool.imap(
                ComputeScoreState.compute_score,
                map(lambda ex: ex.decompiled_code, tqdm(data, desc="Computing scores")), chunksize=args.block_size))
        return scores

    scores = compute_scores()

    # Generate data splits.
    test_size = args.test_split_size or int(len(data) * args.test_split_portion)
    # Dev/Test set: Repositories excluded in training set.
    data_by_repo = defaultdict(list)
    for idx, ex in enumerate(data):
        data_by_repo[ex.repo].append(idx)
    repo_names = list(data_by_repo.keys())

    def create_excluded_split(target_size: int, max_repos: int, extra_train_portion: float, min_repo_size: int = 0) \
            -> Tuple[List[str], List[int], List[int]]:
        # ([name], [index])
        filtered_repos = repo_names
        if min_repo_size > 0:
            filtered_repos = [repo for repo in filtered_repos if len(data_by_repo[repo]) >= min_repo_size]
        while True:
            repo_count = random.randint(1, min(len(filtered_repos), max_repos))
            chosen_repos = random.choices(filtered_repos, k=repo_count)
            sample_size = sum(len(data_by_repo[name]) for name in chosen_repos)
            if 0.8 * target_size <= sample_size <= 1.1 * target_size:
                # Keep sampling until we get something with appropriate size.
                break
        extra_train_indices = []
        split_indices = []
        for name in chosen_repos:
            indices = data_by_repo[name].copy()
            random.shuffle(indices)
            split_size = int(len(indices) * extra_train_portion)
            extra_train_indices += indices[:split_size]
            split_indices += indices[split_size:]
        return chosen_repos, split_indices, extra_train_indices

    dev_repos, dev_split, extra_train_dev_split = create_excluded_split(
        test_size, args.max_test_repos, args.extra_train_portion)
    for repo_name in dev_repos:
        del data_by_repo[repo_name]
    test_repos, test_split, extra_train_test_split = create_excluded_split(
        test_size, args.max_test_repos, args.extra_train_portion)
    excluded_indices = set(dev_split + extra_train_dev_split + test_split + extra_train_test_split)

    # Training set: all the remaining stuff.
    train_split = [idx for idx in range(len(data)) if idx not in excluded_indices]
    train_split.sort(key=lambda i: scores[i])  # sort training indices according to competency score
    extra_train_split = extra_train_dev_split + extra_train_test_split
    splits = {
        "train": train_split,
        "valid": dev_split,
        "test": test_split,
        "train_extra": extra_train_split,
    }
    with (output_dir / "split_indices.pkl").open("wb") as f:
        pickle.dump(splits, f)

    def write_files(folder_path: str, sentence_fn: Callable[[int], Tuple[str, str]]):
        for key, indices in splits.items():
            with open(os.path.join(folder_path, f"{key}.txt"), "w") as f:
                for idx in tqdm(indices, desc=f"Writing {key} set", leave=False):
                    src, tgt = sentence_fn(idx)
                    ex = src, tgt, data[idx].var_names, scores[idx], data[idx].repo, data[idx].sha
                    output = OutputData.encode(*ex)
                    # assert tuple(OutputData.decode(output)) == ex
                    f.write(output)
                    f.write("\n")
            print(f"{key.capitalize()} set written")

    write_files(output_dir, lambda idx: data[idx][:2])
    for key, names in [("valid", dev_repos), ("test", test_repos)]:
        with (output_dir / f"{key}_repos.txt").open("w") as f:
            f.write("\n".join(names))

    if not (output_dir / "vocab.model").exists():
        # Write out training text and train SentencePiece model.
        train_text_path = output_dir / "train_text.txt"
        with train_text_path.open("w") as f:
            for idx in tqdm(train_split, desc="Writing training text"):
                src_tokens = data[idx].decompiled_code.split(TOKEN_SEP)
                new_src_tokens = []
                for token in src_tokens:
                    if token in data[idx].var_names:
                        var1, var2 = data[idx].var_names[token]
                        new_src_tokens += [var1, var2]
                    else:
                        new_src_tokens.append(token)
                f.write(" ".join(new_src_tokens) + "\n")
                f.write(data[idx].original_code.replace(TOKEN_SEP, " ") + "\n")
        spm_train_args = {
            "input": train_text_path,
            "model_prefix": output_dir / "vocab",
            "vocab_size": args.vocab_size,
        }
        spm.SentencePieceTrainer.Train(" ".join(f"--{name}={str(value)}" for name, value in spm_train_args.items()))

    if args.encode_spm:
        # Encode all sentences with the trained SP model.
        with flutes.safe_pool(args.n_procs, state_class=EncodeSPMState,
                              init_args=(output_dir / "vocab.model",)) as pool:
            processed_code = list(pool.imap(
                EncodeSPMState.encode_spm,
                map(lambda ex: (ex.decompiled_code, ex.original_code),
                    tqdm(data, desc="Encoding with SPM")), chunksize=args.block_size))

        write_files(output_dir / "tokenized", lambda idx: processed_code[idx])


if __name__ == '__main__':
    main()
