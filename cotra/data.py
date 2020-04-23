import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypeVar

import flutes
import numpy as np
import texar.torch as tx
import torch
from bashplotlib.histogram import plot_hist
import sentencepiece as spm

from . import utils

__all__ = [
    "Example",
    "CodeData",
    "PairedTextTokenCountBatchingStrategy",
]

T = TypeVar('T')
RawExample = Tuple[str, str, float]


class InputData(NamedTuple):
    decompiled_code: str
    original_code: str
    score: float
    repo: str  # "owner/name"
    sha: str


class Example(NamedTuple):
    src: str
    tgt: str
    src_ids: np.ndarray
    tgt_ids: np.ndarray
    score: float


class CodeDataSource(tx.data.DataSource[RawExample]):
    def __init__(self, path: str, tuple_delimiter: str, verbose: bool = False, max_dataset_size: int = -1):
        self.path = path
        self.verbose = verbose
        self.delimiter = tuple_delimiter
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        with flutes.progress_open(self.path, "r", verbose=self.verbose, desc=f"Reading {self.path}") as f:
            count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                splits = line.split(self.delimiter)
                assert len(splits) == 5
                src, tgt, score, repo, sha = splits
                yield src, tgt, score
                count += 1
                if 0 < self.max_dataset_size <= count:
                    break


class CodeData(tx.data.DatasetBase[RawExample, Example]):
    def __init__(self, path: str, vocab: utils.Vocab,
                 hparams: Optional[Dict[str, Any]] = None, device: Optional[torch.device] = None):
        hparams = {
            **(hparams or {}),
            "lazy_strategy": "process",  # eager loading
            "cache_strategy": "processed",
        }
        self._hparams = tx.HParams(hparams, self.default_hparams())
        self.delimiter = self._hparams.token_delimiter
        self.vocab = vocab

        self.sp = None
        if self._hparams.use_alternate_vocab:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(self._hparams.use_alternate_vocab + ".model")

        file_source = CodeDataSource(path, self._hparams.tuple_delimiter,
                                     verbose=self._hparams.verbose, max_dataset_size=self._hparams.max_dataset_size)
        file_source = tx.data.FilterDataSource(file_source, self._filter_fn)
        if self._hparams.curriculum.enabled:
            data = sorted(file_source, key=lambda ex: ex[2])  # sort by increasing difficulty score
            if self._hparams.verbose:
                print("Data sorted. Difficulty histogram:")
                # scores = [ex[2] for ex in data][:int(0.999 * len(data))]  # ignore outliers
                scores = [ex[2] for ex in data]
                plot_hist(scores, height=10, pch="x", xlab=True, showSummary=True, bincount=70)
        else:
            data = list(file_source)

        source = tx.data.SequenceDataSource(data)
        super().__init__(source, hparams, device)

        self._competency = 100
        if self._hparams.curriculum.enabled:
            # Initialize current available dataset size to maximum, so we can know the actual size before training.
            self._curriculum_dataset_size = self._dataset_size
            self._hparams.shuffle_buffer_size = None

    @staticmethod
    def default_hparams():
        return {
            **tx.data.DatasetBase.default_hparams(),
            "max_src_len": 1024,
            "max_tgt_len": 1024,
            "valid_src_tgt_length_ratio": (0.5, 3.0),  # 0.5 <= len(src) / len(tgt) <= 3.0
            "lazy_strategy": "process",
            "cache_strategy": "processed",
            "curriculum": {
                "enabled": True,
                "init_competency": 0.01,
                "steps": 20000,
            },
            "shuffle_buffer_size": 4096,
            "verbose": False,
            "tuple_delimiter": " ▁|SEP|▁ ",
            "token_delimiter": " ",
            "use_alternate_vocab": None,
        }

    def update_steps(self, steps: int):
        r"""Update the current number of steps. This computes the new competency value and expands
        the available dataset.
        """
        if not self._hparams.curriculum.enabled:
            return  # do nothing
        init_comp_sqr = self._hparams.curriculum.init_competency ** 2
        anneal_steps = self._hparams.curriculum.steps
        competency = min(1.0, math.sqrt((1 - init_comp_sqr) * steps / anneal_steps + init_comp_sqr))
        # assert self._competency <= competency
        new_size = int(competency * self._dataset_size)
        self._competency = competency
        self._curriculum_dataset_size = new_size

    def __len__(self) -> int:
        if self._hparams.curriculum.enabled:
            # Return length based on competency schedule.
            return self._curriculum_dataset_size
        return super().__len__()

    def _filter_fn(self, example: RawExample) -> bool:
        src, tgt, _ = example
        src_len = src.count(self.delimiter) + 1  # count spaces instead of actually performing splitting
        tgt_len = tgt.count(self.delimiter) + 1
        lower, upper = self._hparams.valid_src_tgt_length_ratio
        return (src_len + 1 <= self._hparams.max_src_len and  # account for EOS
                tgt_len + 1 <= self._hparams.max_tgt_len and
                lower <= src_len / tgt_len <= upper)

    def _retokenize(self, tokens: List[str]) -> List[str]:
        result = []
        tokens = utils.DecodeMixin.spm_decode(tokens)
        for tok in tokens:
            result += self.sp.EncodeAsPieces(tok)
        return result

    def process(self, raw_example: RawExample) -> Example:
        # Convert to IDs and add EOS tokens.
        src, tgt, score = raw_example
        src_tokens = src.split(self.delimiter)
        tgt_tokens = tgt.split(self.delimiter)
        if self.sp is not None:
            # Truncate sentences if too long.
            src_tokens = self._retokenize(src_tokens)[:(self._hparams.max_src_len - 1)]
            tgt_tokens = self._retokenize(tgt_tokens)[:(self._hparams.max_tgt_len - 1)]
        src_seq = self.vocab.map_to_ids(src_tokens)
        src_seq.append(self.vocab.eos_id)
        tgt_seq = self.vocab.map_to_ids(tgt_tokens)
        tgt_seq.append(self.vocab.eos_id)
        assert len(src_seq) <= self._hparams.max_src_len and len(tgt_seq) <= self._hparams.max_tgt_len
        return Example(src=src, tgt=tgt, score=score,
                       src_ids=np.asarray(src_seq), tgt_ids=np.asarray(tgt_seq))

    def collate(self, examples: List[Example]) -> tx.data.Batch:
        src_seqs = [ex.src_ids for ex in examples]
        tgt_seqs = [ex.tgt_ids for ex in examples]
        # Pad sentences to equal length.
        source, src_lens = tx.data.padded_batch(src_seqs, pad_value=self.vocab.eos_id)
        target_output, tgt_lens = tx.data.padded_batch(tgt_seqs, pad_value=self.vocab.eos_id)
        # if source.shape[1] >= self._hparams.max_src_len or target_output.shape[1] >= self._hparams.max_tgt_len:
        #     breakpoint()
        # Add SOS token to the target inputs.
        target_input = np.pad(target_output[:, :-1], ((0, 0), (1, 0)),
                              mode="constant", constant_values=self.vocab.sos_id)
        source = torch.from_numpy(source)
        target_input = torch.from_numpy(target_input)
        target_output = torch.from_numpy(target_output)
        src_lens = torch.tensor(src_lens, dtype=torch.long)
        tgt_lens = torch.tensor(tgt_lens, dtype=torch.long)
        return tx.data.Batch(
            len(examples),
            source=source, source_lengths=src_lens,
            target_input=target_input, target_output=target_output, target_lengths=tgt_lens)

    @property
    def competency(self):
        return self._competency


class PairedTextTokenCountBatchingStrategy(tx.data.BatchingStrategy[Example]):
    r"""Create dynamically-sized batches for paired text data so that the total
    number of source and target tokens (including padding) inside each batch is
    constrained.

    Args:
        max_tokens (int): The maximum number of source or target tokens inside
            each batch.
    """
    max_src_len: int
    max_tgt_len: int
    cur_batch_size: int

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def reset_batch(self) -> None:
        self.max_src_len = 0
        self.max_tgt_len = 0
        self.cur_batch_size = 0

    def add_example(self, ex: Example) -> bool:
        max_src_len = max(self.max_src_len, len(ex.src_ids))
        max_tgt_len = max(self.max_tgt_len, len(ex.tgt_ids))
        if (self.cur_batch_size + 1) * max(max_src_len, max_tgt_len) > self.max_tokens:
            return False
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.cur_batch_size += 1
        return True
