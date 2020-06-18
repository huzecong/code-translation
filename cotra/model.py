from typing import Any, Dict, Optional

import texar.torch as tx
import torch
from torch import nn

from . import utils

__all__ = [
    "Seq2seq",
]


class Seq2seq(tx.ModuleBase):
    r"""A standalone sequence-to-sequence Transformer model, from "Attention
    Is All You Need". The Transformer model consists of the word embedding
    layer, position embedding layer, an encoder and a decoder. Both encoder
    and decoder are stacks of self-attention layers followed by feed-forward
    layers. See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
    for the full description of the model.
    """

    vocab: utils.Vocab

    def __init__(self, vocab: utils.Vocab, hparams: Optional[Dict[str, Any]] = None):
        super().__init__(hparams)

        self.vocab = vocab

        hidden_dim = self._hparams.hidden_dim
        self.word_embedder = tx.modules.WordEmbedder(
            vocab_size=vocab.size, hparams={
                "dim": hidden_dim,
                "initializer": {
                    "type": "normal_",
                    "kwargs": {"mean": 0.0, "std": hidden_dim ** -0.5},
                },
            })
        self.pos_embedder = tx.modules.SinusoidsPositionEmbedder(
            position_size=self._hparams.max_sentence_length, hparams={"dim": hidden_dim})

        transformer_hparams = {
            "dim": hidden_dim,
            "num_blocks": 6,
            "multihead_attention": {
                "num_heads": 8,
                "output_dim": hidden_dim
            },
            "initializer": {
                "type": "variance_scaling_initializer",
                "kwargs": {"factor": 1.0, "mode": "FAN_AVG", "uniform": True},
            },
            "poswise_feedforward": tx.modules.default_transformer_poswise_net_hparams(
                input_dim=hidden_dim, output_dim=hidden_dim),
        }
        self.encoder = tx.modules.TransformerEncoder(hparams=transformer_hparams)
        if self._hparams.decoder == "transformer":
            self.decoder = tx.modules.TransformerDecoder(
                token_pos_embedder=self._embedding_fn, vocab_size=vocab.size,
                output_layer=self.word_embedder.embedding, hparams=transformer_hparams)
        elif self._hparams.decoder == "lstm":
            rnn_hparams = tx.modules.AttentionRNNDecoder.default_hparams()
            rnn_hparams['rnn_cell']['kwargs']['hidden_size'] = hidden_dim
            self.decoder = tx.modules.AttentionRNNDecoder(
                input_size=self.word_embedder.dim, encoder_output_size=self.encoder.output_size,
                token_embedder=self.word_embedder, vocab_size=vocab.size,
                output_layer=self.word_embedder.embedding, hparams=rnn_hparams)
        else:
            raise ValueError(f"Invalid 'decoder' hparam: {self._hparams.decoder}")

        self.smoothed_loss_func = LabelSmoothingLoss(
            label_confidence=self._hparams.loss_label_confidence,
            tgt_vocab_size=vocab.size, ignore_index=vocab.pad_id)

    @staticmethod
    def default_hparams():
        return {
            "decoder": "transformer",  # ["transformer", "lstm"]
            "hidden_dim": 512,
            "max_sentence_length": 1024,
            "loss_label_confidence": 0.9,
        }

    def _embedding_fn(self, tokens: torch.LongTensor, positions: torch.LongTensor) -> torch.Tensor:
        word_embed = self.word_embedder(tokens)
        scale = self._hparams.hidden_dim ** 0.5
        pos_embed = self.pos_embedder(positions)
        return word_embed * scale + pos_embed

    def forward(self,  # type: ignore
                encoder_input: torch.Tensor, decoder_input: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                beam_width: Optional[int] = None, length_penalty: float = 0.0):
        r"""Compute the maximum likelihood loss or perform decoding, depending
        on arguments.

        Args:
            encoder_input: the source sentence embedding, with the shape of
                `[batch_size, source_seq_length, input_dim]`.
            decoder_input: the target sentence embedding, with the shape of
                `[batch_size, target_seq_length, input_dim]`.
            labels: the target sentence labels, with the shape of
                `[batch_size, target_seq_length]`.
            beam_width: Used in beam search.

        :returns:
            - If both :attr:`decoder_input` and :attr:`labels` are both
              provided, the function enters training logic and returns the
              maximum likelihood loss.
            - Otherwise the function enters inference logic and returns the
              decoded sequence.
            - If `beam_width` > 1, beam search decoding is performed. Please
              refer to :meth:`texar.modules.TransformerDecoder.forward` for
              details on return types.
        """

        batch_size = encoder_input.size(0)
        # Text sequence length excluding padding
        encoder_input_length = (encoder_input != 0).int().sum(dim=1)
        positions = torch.arange(
            encoder_input_length.max(), dtype=torch.long,
            device=encoder_input.device).unsqueeze(0).expand(batch_size, -1)

        # Source word embedding
        src_input_embedding = self._embedding_fn(encoder_input, positions)
        encoder_output = self.encoder(inputs=src_input_embedding, sequence_length=encoder_input_length)

        if decoder_input is not None and labels is not None:
            # For training
            label_lengths = (labels != 0).long().sum(dim=1)
            outputs = self.decoder(
                decoding_strategy="train_greedy",
                inputs=decoder_input,  sequence_length=label_lengths,
                memory=encoder_output, memory_sequence_length=encoder_input_length)
            if self._hparams.decoder == "lstm":
                outputs = outputs[0]
            is_target = (labels != 0).float()
            mle_loss = self.smoothed_loss_func(outputs.logits, labels, label_lengths)
            mle_loss = (mle_loss * is_target).sum() / is_target.sum()
            return mle_loss

        else:
            start_tokens = encoder_input.new_full((batch_size,), self.vocab.sos_id)
            predictions = self.decoder(
                memory=encoder_output, memory_sequence_length=encoder_input_length,
                beam_width=beam_width, length_penalty=length_penalty,
                start_tokens=start_tokens, end_token=self.vocab.eos_id,
                max_decoding_length=self._hparams.max_sentence_length,
                decoding_strategy="infer_greedy")
            return predictions


class LabelSmoothingLoss(nn.Module):
    r"""With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.

    Args:
        label_confidence: the confidence weight on the ground truth label.
        tgt_vocab_size: the size of the final classification.
        ignore_index: The index in the vocabulary to ignore weight.
    """
    one_hot: torch.Tensor

    def __init__(self, label_confidence, tgt_vocab_size, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.tgt_vocab_size = tgt_vocab_size

        label_smoothing = 1 - label_confidence
        assert 0.0 < label_smoothing <= 1.0
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))
        self.confidence = label_confidence

    def forward(self,  # type: ignore
                output: torch.Tensor, target: torch.Tensor, label_lengths: torch.LongTensor) -> torch.Tensor:
        r"""Compute the label smoothing loss.

        Args:
            output (FloatTensor): batch_size x seq_length * n_classes
            target (LongTensor): batch_size * seq_length, specify the label
                target
            label_lengths(torch.LongTensor): specify the length of the labels
        """
        orig_shapes = (output.size(), target.size())
        output = output.contiguous().view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob = model_prob.to(device=target.device)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        output = output.view(orig_shapes[0])
        model_prob = model_prob.view(orig_shapes[0])

        return tx.losses.sequence_softmax_cross_entropy(
            labels=model_prob, logits=output, sequence_length=label_lengths,
            average_across_batch=False, sum_over_timesteps=False)
