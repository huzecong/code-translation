import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import flutes
import texar.torch as tx
import torch
from argtyped import *
from termcolor import colored
from texar.torch.run import *
from torch import nn

import cotra


class Args(Arguments):
    config_file: str = "config/default.yaml"
    run_mode: Choices["train", "valid", "test"] = "train"
    output_dir: str = "outputs_new/"
    test_output_file: str = "{split}.hyp.orig"
    load_checkpoint: Switch = False
    checkpoint_path: Optional[str] = None
    pdb: Switch = False
    n_procs: int = 2
    curriculum: Switch = True
    debug: Switch = False
    force: Switch = False
    extra_config: Optional[str]  # if specified, will be parsed as dictionary and merged with config dictionary


class ModelWrapper(nn.Module):
    def __init__(self, model: cotra.Seq2seq, beam_width: int, length_penalty: float = 0.0):
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
        config: Dict[str, Any] = cotra.utils.load_yaml(f)
    if args.extra_config is not None:
        import ast
        extra_config = ast.literal_eval(args.extra_config)
        cotra.utils.merge_dict(config, extra_config)
    # Do some validation before running time-consuming processes.
    assert os.path.exists(config["data"]["training_set"])
    assert all(os.path.exists(path) for path in config["data"]["valid_sets"].values())
    assert all(os.path.exists(path) for path in config["data"]["test_sets"].values())

    tx.run.make_deterministic(config["random_seed"])
    print(f"Random seed set to {config['random_seed']}")

    output_dir = Path(args.output_dir)
    if not args.debug and output_dir.exists() and args.run_mode == "train" and not args.force:
        print(colored(f"Output folder '{str(output_dir)}' exists, use --force to overwrite."))
        sys.exit(1)

    # Load data
    eval_datasets: Dict[str, Dict[str, cotra.CodeData]] = {}
    hparams = copy.deepcopy(config["data"]["hparams"])
    vocab = cotra.utils.Vocab.load(config["data"]["vocab_file"])
    train_dataset = None
    if args.run_mode == "train":
        train_dataset = cotra.CodeData(path=config["data"]["training_set"], vocab=vocab, hparams={
            **hparams,
            "shuffle": True,
            "curriculum": {"enabled": args.curriculum},
            "verbose": config["data"]["verbose"],
            "num_parallel_calls": args.n_procs,
        })
    eval_splits: Dict[str, Dict[str, str]] = {
        "valid": config["data"]["valid_sets"],
        "test": config["data"]["test_sets"],
    }
    for split, paths in eval_splits.items():
        eval_datasets[split] = {
            f"{split}_{name}": cotra.CodeData(path=path, vocab=vocab, hparams={
                **hparams,
                "shuffle": False, "curriculum": {"enabled": False},
                "batch_size": config["training"]["test_batch_size"],
                "max_dataset_size": 500 if split == "valid" else -1,
                # Evaluation must use truncate mode -- no example in the test set should be discarded.
                "length_filter_mode": "truncate",
            }) for name, path in paths.items()
        }
    batching_strategy = cotra.CustomBatchingStrategy(config["training"]["max_batch_tokens"])
    print("Dataset initialized")

    # Create model and optimizer
    model = cotra.Seq2seq(vocab, hparams=config["model"])
    model = ModelWrapper(model, beam_width=config["inference"]["beam_width"],
                         length_penalty=config["inference"]["length_penalty"])

    lr_config = config["lr_scheduler"]
    is_static_lr = lr_config["schedule"] == "static"
    scheduler_lambda = cotra.utils.get_lr_schedule(static=is_static_lr, warmup_steps=lr_config["warmup_steps"])
    optim = torch.optim.Adam(model.parameters(), lr=lr_config["init_lr"], betas=(0.9, 0.997), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, scheduler_lambda)
    print("Model constructed")

    training_config = config["training"]
    test_output_path = output_dir / "test.output"
    valid_set = next(iter(eval_datasets["valid"].values()))  # only validate on first valid split
    executor = Executor(
        model=model,
        train_data=train_dataset,
        valid_data=valid_set,
        test_data=eval_datasets["test"],
        batching_strategy=batching_strategy,
        optimizer=optim,
        lr_scheduler=scheduler,
        log_destination=[sys.stdout, *([output_dir / "log.txt"] if not args.debug else [])],
        log_every=cond.iteration(training_config["display_steps"]),
        validate_every=cond.iteration(training_config["eval_steps"]),
        stop_training_on=cond.iteration(training_config["max_train_steps"]),
        train_metrics=[("loss", metric.RunningAverage(20)),  # average over 20 iterations
                       ("lr", metric.LR(optim))],
        log_format="{time} : Epoch {epoch:2d} @ {iteration:6d}it "
                   "({progress}%, {speed}), lr = {lr:.3e}, loss = {loss:.3f}",
        valid_metrics=cotra.utils.WordPieceBLEU(vocab, decode=True, encoding="spm",
                                                sample_output_per=len(valid_set) // 10),
        test_metrics=[cotra.utils.FileBLEU(vocab, test_output_path, encoding="spm"),
                      ("unofficial_bleu", cotra.utils.WordPieceBLEU(vocab, decode=True, encoding="spm"))],
        valid_log_format="{time} : Epoch {epoch}, {split} BLEU = {BLEU:.3f}",
        test_progress_log_format=("{time} : Evaluating on {split} ({progress}%, {speed}), "
                                  "unofficial BLEU = {unofficial_bleu:.2f}"),
        validate_mode='predict',
        checkpoint_dir=(args.output_dir if not args.debug else None),
        save_every=(cond.validation(better=True) if not args.debug else None),
        max_to_keep=5,
        show_live_progress=True,
    )

    all_datasets = {"train": train_dataset,
                    **{key: value for datasets in eval_datasets.values() for key, value in datasets.items()}}
    executor.write_log("Data size: " +
                       repr({key: len(split) for key, split in all_datasets.items() if split is not None}))

    if args.curriculum:
        @executor.on_event(cond.Event.Epoch, 'begin')
        def update_dataset_steps(exc: Executor):
            assert train_dataset is not None
            train_dataset.update_steps(exc.status["iteration"])
            exc._train_tracker.set_size(len(train_dataset))
            exc.write_log(f"Epoch {exc.status['epoch']}, competency updated to {train_dataset.competency * 100:6.2f}%")

    # @executor.on(cond.validation(better=True))
    # def test_on_excluded_validation_set(exc: Executor):
    #     exc.test({"valid_repos_excluded": eval_datasets["valid"]["valid_repos_excluded"]})

    executor.write_log(f"Begin running with {args.run_mode} mode")
    if args.run_mode == "train":
        if args.load_checkpoint:
            load_path = executor.load(path=args.checkpoint_path, allow_failure=True)
            # if load_path is not None:
            #     executor.test(eval_datasets["valid"])

        executor.train()
    else:
        executor.load(path=args.checkpoint_path, load_training_state=False)
        for name, dataset in eval_datasets[args.run_mode].items():
            executor.test({name: dataset})
            # Manually rename the test output file.
            os.rename(str(test_output_path) + ".hyp", output_dir / args.test_output_file.format(split=name))


if __name__ == "__main__":
    main()
