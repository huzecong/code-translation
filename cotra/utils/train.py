import math
from typing import Any, Dict, List, Optional

import torch

__all__ = [
    "get_lr_scheduler",
    "Vocab",
]


def get_lr_scheduler(optim: torch.optim.Optimizer,
                     lr_config: Dict[str, Any]) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    r"""Calculate the learning rate multiplier given current step and the number
    of warm-up steps. The learning rate schedule follows a linear warm-up and
    square-root decay.
    """
    base_lr = lr_config["lr"]
    warmup_steps = lr_config.get("warmup_steps", 0)

    if lr_config["schedule"] == "static":
        # Constant
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0

    elif lr_config["schedule"] == "invsqrt":
        # Inverse square root
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1 / math.sqrt(max(1, step - warmup_steps))

    elif lr_config["schedule"] == "exponential":
        # Exponential (scale by constant every few iterations)
        scale = lr_config["scale"]
        per_steps = lr_config["per_steps"]

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return math.pow(scale, (step - warmup_steps) // per_steps)

    elif lr_config["schedule"] == "cosine":
        # Cosine with hard-resets
        reset_steps = lr_config.get("reset_steps", 200000)
        min_lr_multiplier = max(0.0, lr_config.get("min_lr", 0.0)) / base_lr

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, reset_steps - warmup_steps)
            return min_lr_multiplier + (1 - min_lr_multiplier) * (0.5 * (1.0 + math.cos(math.pi * (progress % 1.0))))
    else:
        raise ValueError(f"Invalid LR schedule: {lr_config['schedule']}")
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


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
                    words.append(line.split()[0])
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
