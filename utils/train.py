import math
from typing import Optional, Callable, List

__all__ = [
    "get_lr_schedule",
    "Vocab",
]


def get_lr_schedule(*, static: bool, warmup_steps: Optional[int] = None) -> Callable[[int], float]:
    r"""Calculate the learning rate multiplier given current step and the number
    of warm-up steps. The learning rate schedule follows a linear warm-up and
    square-root decay.
    """
    if static:
        def lr_multiplier(step: int) -> float:
            return 1.0
    else:
        assert warmup_steps is not None

        def lr_multiplier(step: int) -> float:
            return min(1.0, step / warmup_steps) * (1 / math.sqrt(max(step, warmup_steps)))  # type: ignore
    return lr_multiplier


class Vocab:
    def __init__(self, words: List[str],
                 pad_token: str = "<pad>", unk_token: str = "<unk>", sos_token: str = "<s>", eos_token: str = "</s>",
                 special_tokens: Optional[List[str]] = None):
        special_tokens = [pad_token, unk_token, sos_token, eos_token] + (special_tokens or [])
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("All special tokens (including pad, unk, sos, eos) must be unique")
        words = words.copy()
        special_tokens = [token for token in special_tokens if token not in words]
        self._i2w = special_tokens + words
        self._w2i = {word: idx for idx, word in enumerate(self._i2w)}
        self.pad_token, self.unk_token, self.sos_token, self.eos_token = pad_token, unk_token, sos_token, eos_token
        self.unk_id = self._w2i[unk_token]
        self.pad_id, self.sos_id, self.eos_id = self.map_to_ids([pad_token, sos_token, eos_token])

    @staticmethod
    def load(path: str) -> 'Vocab':
        with open(path) as f:
            words = []
            for line in f:
                line = line.rstrip("\n")
                if line:
                    words.append(line)
        return Vocab(words)

    @property
    def token_to_id_map(self):
        return self._w2i

    @property
    def id_to_token_map(self):
        return self._i2w

    def map_to_ids(self, sentence: List[str]) -> List[int]:
        return [self._w2i.get(w, self.unk_id) for w in sentence]

    def map_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._i2w[i] for i in ids]

    @property
    def size(self):
        return len(self._i2w)
