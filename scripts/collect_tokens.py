import math
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Counter as CounterT, Iterator, List, Iterable, TypeVar

import flutes
import sentencepiece as spm
from argtyped import Arguments, Switch

from cotra.parse import Lexer

T = TypeVar('T')


class Args(Arguments):
    random_seed: int = 19260817
    data_dir: str = "../github/mt_data_varnames"
    output_dir: str = "vocab_varnames"
    n_procs: int = 0
    vocab_size: int = 10000
    sample_size: int = 10 ** 7
    pdb: Switch = False


class PoolState(flutes.PoolState):
    LITERAL_TYPES = {
        'INT_CONST_DEC', 'INT_CONST_OCT', 'INT_CONST_HEX', 'INT_CONST_BIN', 'INT_CONST_CHAR',
        'FLOAT_CONST', 'HEX_FLOAT_CONST',
        'CHAR_CONST',
        'WCHAR_CONST',
        'STRING_LITERAL',
        'WSTRING_LITERAL',
    }

    def __init__(self, bar: flutes.ProgressBarManager.Proxy):
        self.ident_tokens: CounterT[str] = Counter()
        self.lexer = Lexer()
        self.bar = bar

    def __return_state__(self):
        # Don't include `lexer` and `bar` while pickling; `lexer` includes unpickle-able regex objects.
        return self.ident_tokens

    def collect_tokens(self, path: str) -> List[str]:
        with open(path, "rb") as f:
            examples = pickle.load(f)
        str_tokens = []
        for ex in self.bar.new(examples, desc=f"Worker {flutes.get_worker_id()}", update_frequency=0.01):
            src_code, tgt_code, var_map, *_ = ex
            for code in [src_code, tgt_code]:
                for token in self.lexer.lex_tokens(code):
                    if token.type == "ID":
                        if token.value in var_map:
                            decomp_name, orig_name = var_map[token.value]
                            self.ident_tokens[decomp_name] += 1
                            self.ident_tokens[orig_name] += 1
                        else:
                            self.ident_tokens[token.value] += 1
                    elif token.type in self.LITERAL_TYPES:
                        str_tokens.append(token.value)
        return str_tokens


# def sample(k: int, iterable: Iterable[T]) -> List[T]:
#     # Performing on-the-fly sampling with the Reservoir Sampling algorithm.
#     pool = []
#     for idx, x in enumerate(iterable):
#         if idx < k:
#             pool.append(x)
#         else:
#             p = random.randint(0, idx)
#             if p < k:
#                 pool[p] = x
#     return pool


def sample(k: int, iterable: Iterable[T]) -> List[T]:
    # Performing on-the-fly sampling with the Reservoir Sampling algorithm L
    # Reference: https://en.wikipedia.org/wiki/Reservoir_sampling#An_optimal_algorithm.
    pool = []
    it = iter(iterable)
    try:
        for _ in range(k):
            pool.append(next(it))
    except StopIteration:
        return pool

    W = math.exp(math.log(random.random()) / k)
    try:
        while True:
            skip_cnt = int(math.floor(math.log(random.random()) / math.log(1 - W)))
            for _ in range(skip_cnt):
                next(it)
            pool[random.randint(0, k - 1)] = next(it)
            W *= math.exp(math.log(random.random()) / k)
    except StopIteration:
        pass
    return pool


def main():
    args = Args()
    print(args.to_string())
    if args.pdb:
        flutes.register_ipython_excepthook()
    random.seed(args.random_seed)

    files = [file for file in Path(args.data_dir).iterdir() if file.suffix == ".pkl"]
    manager = flutes.ProgressBarManager()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with flutes.safe_pool(args.n_procs, state_class=PoolState, init_args=(manager.proxy,)) as pool:
        def _wrap_iter() -> Iterator[str]:
            progress = manager.proxy.new(total=len(files))
            for tokens in pool.imap_unordered(PoolState.collect_tokens, files):
                yield from tokens
                progress.update(1)
            ident_tokens = Counter()
            for id_counts in pool.get_states():
                ident_tokens.update(id_counts)
            progress = manager.proxy.new(total=sum(ident_tokens.values()))
            for token, count in ident_tokens.items():
                yield from [token] * count
                progress.update(count)

        sampled_tokens = sample(args.sample_size, _wrap_iter())

    with (output_dir / "tokens.txt").open("w") as f:
        random.shuffle(sampled_tokens)
        for token in sampled_tokens:
            f.write(token)
            f.write("\n")

    spm_train_args = {
        "input": output_dir / "tokens.txt",
        "model_prefix": output_dir / "vocab",
        "vocab_size": args.vocab_size,
        "split_by_whitespace": 0,  # false
        "remove_extra_whitespaces": 0,  # false
        # "input_sentence_size": 10 ** 8,
    }
    spm.SentencePieceTrainer.Train(" ".join(f"--{name}={str(value)}" for name, value in spm_train_args.items()))


if __name__ == '__main__':
    main()
