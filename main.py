import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import flutes
import texar.torch as tx
import torch
import yaml
from argtyped import *
from termcolor import colored
from texar.torch.run import *
from torch import nn

import cotra


class Args(Arguments):
    config_file: str = "config/default.yaml"
    run_mode: Choices["train", "valid", "test"] = "train"
    output_dir: str = "outputs/"
    test_output_file: str = "{split}.hyp.orig"
    load_checkpoint: Switch = False
    checkpoint_path: Optional[str] = None
    use_alternate_vocab: Optional[str] = None
    pdb: Switch = False
    n_procs: int = 0
    curriculum: Switch = True
    debug: Switch = False


class ModelWrapper(nn.Module):
    def __init__(self, model: cotra.Transformer, beam_width: int, length_penalty: float = 0.0):
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
        flutes.register_ipython_excepthook()
    if args.debug:
        print(colored("Running in debug mode: no checkpoints or logs will be saved", "yellow"))

    with open(args.config_file) as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    # Do some validation before running time-consuming processes.
    assert os.path.exists(config["data"]["training_set"])
    assert all(os.path.exists(path) for path in config["data"]["valid_sets"].values())
    assert all(os.path.exists(path) for path in config["data"]["test_sets"].values())

    tx.run.make_deterministic(config["random_seed"])
    print(f"Random seed set to {config['random_seed']}")

    # Load data
    datasets: Dict[str, flutes.MaybeDict[cotra.CodeData]] = {}
    hparams = copy.deepcopy(config["data"]["hparams"])
    if args.use_alternate_vocab is not None:
        hparams["use_alternate_vocab"] = args.use_alternate_vocab
        vocab = cotra.utils.Vocab.load(args.use_alternate_vocab + ".vocab")
    else:
        vocab = cotra.utils.Vocab.load(config["data"]["vocab_file"])
    if args.run_mode == "train":
        datasets["train"] = cotra.CodeData(path=config["data"]["training_set"], vocab=vocab, hparams={
            "shuffle": True, "curriculum": {"enabled": args.curriculum},
            "verbose": config["data"]["verbose"], "num_parallel_calls": args.n_procs,
            **hparams,
        })
    eval_splits: Dict[str, flutes.MaybeDict[str]] = {
        "valid": config["data"]["valid_sets"],
        "test": config["data"]["test_sets"],
    }
    for split, paths in eval_splits.items():
        datasets[split] = {
            f"{split}_{name}": cotra.CodeData(path=path, vocab=vocab, hparams={
                "shuffle": False, "curriculum": {"enabled": False},
                "batch_size": config["training"]["test_batch_size"],
                "lazy_strategy": "none", "max_dataset_size": 500 if split == "valid" else -1,
                **hparams,
            }) for name, path in paths.items()
        }
    batching_strategy = cotra.PairedTextTokenCountBatchingStrategy(config["training"]["max_batch_tokens"])

    # Create model and optimizer
    model = cotra.Transformer(vocab, hparams=config["model"])
    model = ModelWrapper(model, beam_width=config["inference"]["beam_width"],
                         length_penalty=config["inference"]["length_penalty"])

    lr_config = config["lr_scheduler"]
    is_static_lr = lr_config["schedule"] == "static"
    scheduler_lambda = cotra.utils.get_lr_schedule(static=is_static_lr, warmup_steps=lr_config["warmup_steps"])
    optim = torch.optim.Adam(model.parameters(), lr=lr_config["init_lr"], betas=(0.9, 0.997), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, scheduler_lambda)

    output_dir = Path(args.output_dir)
    encoding = config["data"].get("encoding", None)
    training_config = config["training"]
    test_output_path = output_dir / "test.output"
    executor = Executor(
        model=model,
        train_data=datasets.get("train", None),
        valid_data=datasets["valid"]["valid_repos_included"],
        test_data=datasets["test"],
        batching_strategy=batching_strategy,
        optimizer=optim,
        lr_scheduler=scheduler,
        log_destination=[sys.stdout] + ([output_dir / "log.txt"] if not args.debug else []),
        log_every=cond.iteration(training_config["display_steps"]),
        validate_every=cond.iteration(training_config["eval_steps"]),
        stop_training_on=cond.iteration(training_config["max_train_steps"]),
        train_metrics=[("loss", metric.RunningAverage(20)),  # average over 20 iterations
                       ("lr", metric.LR(optim))],
        log_format="{time} : Epoch {epoch:2d} @ {iteration:6d}it "
                   "({progress}%, {speed}), lr = {lr:.3e}, loss = {loss:.3f}",
        valid_metrics=cotra.utils.WordPieceBLEU(vocab, decode=True, encoding=encoding,
                                                sample_output_per=len(datasets["valid"]) // 10),
        test_metrics=[cotra.utils.FileBLEU(vocab, test_output_path, encoding=encoding),
                      ("unofficial_bleu", cotra.utils.WordPieceBLEU(vocab, decode=True, encoding=encoding))],
        valid_log_format="{time} : Epoch {epoch}, {split} BLEU = {BLEU:.3f}",
        test_progress_log_format=("{time} : Evaluating on {split} ({progress}%, {speed}), "
                                  "unofficial BLEU = {unofficial_bleu:.2f}"),
        validate_mode='predict',
        checkpoint_dir=(args.output_dir if not args.debug else None),
        save_every=(cond.validation(better=True) if not args.debug else None),
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

    @executor.on(cond.validation(better=True))
    def test_on_excluded_validation_set(exc: Executor):
        exc.test({"valid_repos_excluded": datasets["valid"]["valid_repos_excluded"]})

    executor.write_log(f"Begin running with {args.run_mode} mode")
    if args.run_mode == "train":
        if args.load_checkpoint:
            load_path = executor.load(path=args.checkpoint_path, allow_failure=True)
            if load_path is not None:
                executor.test(datasets["valid"])

        executor.train()
    else:
        executor.load(path=args.checkpoint_path, load_training_state=False)
        for name, dataset in datasets[args.run_mode].items():
            executor.test({name: dataset})
            # Manually rename the test output file.
            os.rename(str(test_output_path) + ".hyp", output_dir / args.test_output_file.format(split=name))


if __name__ == "__main__":
    main()
