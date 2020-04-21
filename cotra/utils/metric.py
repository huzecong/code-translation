import tempfile
from pathlib import Path
from typing import List, Optional, Union

from termcolor import colored
from texar import torch as tx
from texar.torch.run import metric

from .override import write_log
from .train import Vocab

__all__ = [
    "FileBLEU",
    "WordPieceBLEU",
]


class DecodeMixin:
    vocab: Vocab
    perform_decode: bool
    encoding: Optional[str]

    valid_encodings = ["bpe", "spm"]
    spm_bos_token = "▁"
    bpe_cont_str = "@@"

    @staticmethod
    def spm_decode(tokens: List[str]) -> List[str]:
        words = []
        pieces: List[str] = []
        for t in tokens:
            if t[0] == DecodeMixin.spm_bos_token:
                if len(pieces) > 0:
                    words.append(''.join(pieces))
                pieces = [t[1:]]
            else:
                pieces.append(t)
        if len(pieces) > 0:
            words.append(''.join(pieces))
        return words

    @staticmethod
    def bpe_decode(tokens: List[str]) -> List[str]:
        words = []
        pieces: List[str] = []
        for t in tokens:
            if t.endswith(DecodeMixin.bpe_cont_str):
                pieces.append(t[:-2])
            else:
                words.append(''.join(pieces + [t]))
                pieces = []
        if len(pieces) > 0:
            words.append(''.join(pieces))
        return words

    def _to_str(self, token_ids: List[int]) -> str:
        pos = next((idx for idx, x in enumerate(token_ids) if x == self.vocab.eos_id), -1)
        if pos != -1:
            token_ids = token_ids[:pos]

        words = self.vocab.map_to_tokens(token_ids)
        if self.encoding is not None and self.perform_decode:
            if self.encoding == "bpe":
                words = self.bpe_decode(words)
            elif self.encoding == "spm":
                words = self.spm_decode(words)
        sentence = ' '.join(words)
        return sentence

    def __getstate__(self):
        # Save anything but the vocabulary.
        state = self.__dict__.copy()
        state["vocab"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    def __getnewargs__(self):
        return (None,)


class FileBLEU(metric.SimpleMetric[List[int], float], DecodeMixin):
    def __init__(self, vocab: Vocab, file_path: Optional[Union[str, Path]] = None, encoding: Optional[str] = None):
        super().__init__(pred_name="preds", label_name="target_output", higher_is_better=True)
        self.vocab = vocab
        self.file_path = file_path
        self.perform_decode = True
        if encoding is not None and encoding not in self.valid_encodings:
            raise ValueError(f"Invalid encoding scheme {self.encoding}")
        self.encoding = encoding

    @property
    def metric_name(self) -> str:
        return "BLEU"

    def _value(self) -> float:
        if len(self.predicted) == 0:
            return 0.0
        path = self.file_path or tempfile.mktemp()
        hypotheses, references = [], []
        for hyp, ref in zip(self.predicted, self.labels):
            hypotheses.append(self._to_str(hyp))
            references.append(self._to_str(ref))
        hyp_file, ref_file = tx.utils.write_paired_text(
            hypotheses, references,
            path, mode="s", src_fname_suffix="hyp", tgt_fname_suffix="ref")
        bleu = tx.evals.file_bleu(ref_file, hyp_file, case_sensitive=True)
        return bleu


class WordPieceBLEU(metric.BLEU, DecodeMixin):
    _examples_before_sample: int
    _sample_cnt: int

    def __init__(self, vocab: Vocab, decode: bool = False, encoding: Optional[str] = None,
                 sample_output_per: Optional[int] = None):
        self._sample_output_per = sample_output_per
        super().__init__(pred_name="preds", label_name="target_output")
        self.vocab = vocab
        self.perform_decode = decode
        if encoding is not None and encoding not in self.valid_encodings:
            raise ValueError(f"Invalid encoding scheme {self.encoding}")
        self.encoding = encoding

    @property
    def metric_name(self) -> str:
        return "BLEU"

    def reset(self) -> None:
        self._examples_before_sample = 0
        self._sample_cnt = 0
        super().reset()

    def add(self, predicted, labels) -> None:
        predicted = [self._to_str(s) for s in predicted]
        labels = [self._to_str(s) for s in labels]
        super().add(predicted, labels)
        if self._sample_output_per is not None:
            self._examples_before_sample -= len(predicted)
            if self._examples_before_sample <= 0:
                self._sample_cnt += 1
                write_log(colored(f"Target {self._sample_cnt}: ", "green") + labels[0], mode="info")
                write_log(colored(f"Prediction {self._sample_cnt}: ", "red") + (predicted[0] or "<EMPTY>"), mode="info")
                self._examples_before_sample = self._sample_output_per
