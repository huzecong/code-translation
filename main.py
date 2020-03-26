import os
import sys
from pathlib import Path
from typing import Any, Dict

import texar.torch as tx
import torch
import yaml
from argtyped import *
from texar.torch.run import *
from torch import nn

import utils
from data import CodeData, CustomBatchingStrategy
from model import Transformer


class Args(Arguments):
    config_file: str = "config/default.yaml"
    run_mode: Choices["train", "valid", "test"] = "train"
    output_dir: str = "outputs/"
    load_checkpoint: Switch = False
    pdb: Switch = False
    n_procs: int = 0
    curriculum: Switch = True


class ModelWrapper(nn.Module):
    def __init__(self, model: Transformer, beam_width: int, length_penalty: float = 0.0):
        super().__init__()
        self.model = model
        self.beam_width = beam_width
        self.length_penalty = length_penalty

    def forward(self,  # type: ignore
                batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        loss = self.model(encoder_input=batch.source, decoder_input=batch.target_input,
                          labels=batch.target_output)
        return {"loss": loss}

    def predict(self, batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        predictions = self.model(encoder_input=batch.source, beam_width=self.beam_width,
                                 length_penalty=self.length_penalty)
        if self.beam_width == 1:
            decoded_ids = predictions[0].sample_id
        else:
            decoded_ids = predictions["sample_id"][:, :, 0]
        return {"preds": decoded_ids}


def main() -> None:
    args = Args()
    if args.pdb:
        utils.register_ipython_excepthook()

    with open(args.config_file) as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    tx.run.make_deterministic(config["random_seed"])

    # Load data
    vocab = utils.Vocab.load(config["data"]["vocab_file"])
    datasets: Dict[str, CodeData] = {
        split: CodeData(
            path=os.path.join(config["data"]["filename_pattern"].format(split=split)),
            vocab=vocab,
            hparams={
                **config["data"].get("hparams", {}),
                **(
                    # hparams for training set.
                    {"shuffle": True, "curriculum": {"enabled": args.curriculum},
                     "verbose": config["data"]["verbose"], "num_parallel_calls": args.n_procs}
                    if split == "train" else
                    # hparams for valid/test set.
                    {"shuffle": False, "curriculum": {"enabled": False},
                     "batch_size": config["training"]["test_batch_size"],
                     "lazy_strategy": "none", "max_dataset_size": 500 if split == "valid" else -1}
                )
            }
        ) for split in (["train", "valid", "test"] if args.run_mode == "train" else [args.run_mode])
    }
    batching_strategy = CustomBatchingStrategy(config["training"]["max_batch_tokens"])

    # Create model and optimizer
    model = Transformer(vocab, hparams=config["model"])
    model = ModelWrapper(model, beam_width=config["inference"]["beam_width"],
                         length_penalty=config["inference"]["length_penalty"])

    lr_config = config["lr_scheduler"]
    is_static_lr = lr_config["schedule"] == "static"
    scheduler_lambda = utils.get_lr_schedule(static=is_static_lr, warmup_steps=lr_config["warmup_steps"])
    optim = torch.optim.Adam(model.parameters(), lr=lr_config["init_lr"], betas=(0.9, 0.997), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, scheduler_lambda)

    output_dir = Path(args.output_dir)
    encoding = config["data"].get("encoding", None)
    training_config = config["training"]
    executor = Executor(
        model=model,
        train_data=datasets.get("train", None),
        valid_data=datasets.get("valid", None),
        test_data=datasets.get("test", None),
        batching_strategy=batching_strategy,
        optimizer=optim,
        lr_scheduler=scheduler,
        log_destination=[sys.stdout, output_dir / "log.txt"],
        log_every=cond.iteration(training_config["display_steps"]),
        validate_every=cond.iteration(training_config["eval_steps"]),
        stop_training_on=cond.iteration(training_config["max_train_steps"]),
        train_metrics=[("loss", metric.RunningAverage(20)),  # average over 20 iterations
                       ("lr", metric.LR(optim))],
        log_format="{time} : Epoch {epoch:2d} @ {iteration:6d}it "
                   "({progress}%, {speed}), lr = {lr:.3e}, loss = {loss:.3f}",
        valid_metrics=utils.WordPieceBLEU(vocab, decode=True, encoding=encoding,
                                          sample_output_per=len(datasets["valid"]) // 10),
        test_metrics=[utils.FileBLEU(vocab, output_dir / "test.output", encoding=encoding),
                      ("unofficial_bleu", utils.WordPieceBLEU(vocab, decode=True, encoding=encoding))],
        valid_log_format="{time} : Epoch {epoch}, {split} BLEU = {BLEU:.3f}",
        test_progress_log_format=("{time} : Evaluating on test ({progress}%, {speed}), "
                                  "unofficial BLEU = {unofficial_bleu:.2f}"),
        validate_mode='predict',
        checkpoint_dir=args.output_dir,
        save_every=cond.validation(better=True),
        max_to_keep=5,
        show_live_progress=True,
    )

    executor.write_log("Data size: " + repr({key: len(split) for key, split in datasets.items()}))

    if args.curriculum:
        @executor.on_event(cond.Event.Epoch, 'begin')
        def update_dataset_steps(exc: Executor):
            training_set = datasets["train"]
            training_set.update_steps(exc.status["iteration"])
            exc._train_tracker.set_size(len(training_set))
            exc.write_log(f"Epoch {exc.status['epoch']}, competency updated to {training_set.competency * 100:6.2f}%")

    executor.write_log(f"Begin running with {args.run_mode} mode")
    if args.run_mode == "train":
        if args.load_checkpoint:
            load_path = executor.load(allow_failure=True)
            if load_path is not None:
                executor.test({"valid": datasets["valid"]})

        executor.train()
    else:
        executor.load(load_training_state=False)
        split = "test" if args.run_mode == "test" else "valid"
        executor.test({split: datasets[split]})


if __name__ == "__main__":
    main()