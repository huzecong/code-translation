import math
from typing import List, NamedTuple, Optional, Tuple, TypeVar

import numpy as np
import texar.torch as tx
import torch

import utils

T = TypeVar('T')
RawExample = Tuple[str, str, float]


class Example(NamedTuple):
    src: str
    tgt: str
    src_ids: np.ndarray
    tgt_ids: np.ndarray
    score: float


class CodeDataSource(tx.data.DataSource[RawExample]):
    DELIMITER = " ▁|SEP|▁ "

    def __init__(self, path: str):
        self.path = path

    def __iter__(self):
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                splits = line.split(self.DELIMITER)
                assert len(splits) == 3
                src = splits[0].strip()
                tgt = splits[1].strip()
                score = float(splits[2])
                yield src, tgt, score


class CodeData(tx.data.DatasetBase[RawExample, Example]):
    def __init__(self, path: str, vocab: utils.Vocab,
                 hparams=None, device: Optional[torch.device] = None):
        source = CodeDataSource(path)
        source = tx.data.FilterDataSource(source, self._filter_fn)
        self.vocab = vocab
        hparams = (hparams or {}).update(
            lazy_strategy="process",  # eager loading
            cache_strategy="processed",
        )
        super().__init__(source, hparams, device)

        # At this point, all data is loaded from source and can be accessed through indexing `self._source`.
        # Actual data length is `self._dataset_size`.
        self._source = sorted(self._source, key=lambda ex: ex[2])  # sort by increasing difficulty score
        self._sum_scores = sum(ex[2] for ex in self._source)

        self._competency = 0
        self._curriculum_dataset_size = 0  # current available dataset size
        self._curriculum_sum_scores = 0  # sum of scores for current available data examples
        self.update_steps(0)

    def update_steps(self, steps: int):
        r"""Update the current number of steps. This computes the new competency value and expands
        the available dataset.
        """
        init_comp_sqr = self._hparams.curriculum.init_competency ** 2
        anneal_steps = self._hparams.curriculum.steps
        competency = min(1.0, math.sqrt((1 - init_comp_sqr) * steps / anneal_steps + init_comp_sqr))
        assert self._competency <= competency
        current_sum = self._curriculum_sum_scores
        current_size = self._curriculum_dataset_size
        target_sum = self._sum_scores * competency
        while current_size < self._dataset_size:
            score = self[current_size].score
            if current_sum + score > target_sum:
                break
            current_sum += score
            current_size += 1
        self._competency = competency
        self._curriculum_dataset_size = current_size
        self._curriculum_sum_scores = current_sum

    def __len__(self) -> int:
        # Return length based on competency schedule.
        return self._curriculum_dataset_size

    def _filter_fn(self, example: RawExample) -> bool:
        src, tgt, _ = example
        src_len = src.find(" ") + 1  # count spaces instead of actually performing splitting
        tgt_len = tgt.find(" ") + 1
        lower, upper = self._hparams.valid_src_tgt_length_ratio
        return (src_len <= self._hparams.max_src_len and
                tgt_len <= self._hparams.max_tgt_len and
                lower <= src_len / tgt_len <= upper)

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
                "enable": True,
                "init_competency": 0.01,
                "steps": 20000,
            }
        }

    def process(self, raw_example: RawExample) -> Example:
        # Convert to IDs and add EOS tokens.
        src, tgt, score = raw_example
        src_seq = self.vocab.map_to_ids(src.split())
        src_seq.append(self.vocab.eos_id)
        tgt_seq = self.vocab.map_to_ids(tgt.split())
        tgt_seq.append(self.vocab.eos_id)
        return Example(src=src, tgt=tgt, score=score,
                       src_ids=np.asarray(src_seq), tgt_ids=np.asarray(tgt_seq))

    def collate(self, examples: List[Example]) -> tx.data.Batch:
        src_seqs = [ex.src_ids for ex in examples]
        tgt_seqs = [ex.tgt_ids for ex in examples]
        # Pad sentences to equal length.
        source, src_lens = tx.data.padded_batch(src_seqs, pad_value=self.vocab.eos_id)
        target_output, tgt_lens = tx.data.padded_batch(tgt_seqs, pad_value=self.vocab.eos_id)
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


class CustomBatchingStrategy(tx.data.BatchingStrategy[Example]):
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
