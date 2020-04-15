import random
from pathlib import Path
from typing import List, Optional, Set, Tuple, TypeVar

import argtyped
import flutes
import ujson
from argtyped import Switch


class Arguments(argtyped.Arguments):
    input_dir: str = "match_output/"
    output_dir: str = "mt_data/"
    # vocab_size: int = 32000
    # use_bpe: Switch = True
    max_files: Optional[int]
    resplit: Switch = False
    train_split: float = 0.98
    dev_split: float = 0.01
    test_split: float = 0.01
    random_seed: int = 19260817


T = TypeVar('T')


def split_data(data: List[T], portion: List[float]) -> List[List[T]]:
    sum_portion = sum(portion)
    pref_por = [0.0]
    for p in portion:
        pref_por.append(pref_por[-1] + p / sum_portion)
    pref_por[-1] = 1.0
    bounds = [int(len(data) * p) for p in pref_por]
    return [data[l:r] for l, r in zip(bounds, bounds[1:])]


def write_paired_text(data: List[Tuple[str, str]], prefix: Path) -> None:
    with prefix.with_suffix(".src").open("w") as f:
        for src, _ in data:
            f.write(src + "\n")
    with prefix.with_suffix(".tgt").open("w") as f:
        for _, tgt in data:
            f.write(tgt + "\n")


def main():
    flutes.register_ipython_excepthook()

    args = Arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("Dataset creation start")

    original_code_set: Set[str] = set()
    total_cnt = 0
    file_cnt = 0
    data: List[Tuple[str, str]] = []

    if args.resplit and (output_dir / "all.src").exists():
        with (output_dir / "all.src").open() as f:
            src_data = [line.strip() for line in f if line]
        with (output_dir / "all.tgt").open() as f:
            tgt_data = [line.strip() for line in f if line]
        assert len(src_data) == len(tgt_data)
        data = list(zip(src_data, tgt_data))

        print(f"Loaded {len(data)} examples")
    else:
        for file in input_dir.iterdir():
            with file.open() as f:
                for line in f:
                    if not line:
                        continue
                    try:
                        matched_func = ujson.loads(line)
                    except Exception:
                        continue
                    if isinstance(matched_func["original_code"], str):
                        continue
                    if "\n" in matched_func["original_code"] or "\n" in matched_func["decompiled_code"]:
                        breakpoint()
                    total_cnt += 1
                    original_code = " ".join(matched_func["original_code"])
                    if original_code in original_code_set:
                        continue
                    decompiled_code = " ".join(matched_func["decompiled_code"])
                    data.append((decompiled_code, original_code))  # (src, tgt)
                    original_code_set.add(original_code)
                file_cnt += 1
                if file_cnt % 200 == 0:
                    print(f"Processed {file_cnt} files")
                if args.max_files is not None and file_cnt >= args.max_files:
                    break

        print(f"Found {total_cnt} examples ({len(data)} unique examples)", "success")
        write_paired_text(data, output_dir / "all")

    random.seed(args.random_seed)
    random.shuffle(data)

    datasets = split_data(data, [args.train_split, args.dev_split, args.test_split])
    for dataset, path in zip(datasets, ["train", "dev", "test"]):
        write_paired_text(dataset, output_dir / path)
        print(f"{path.capitalize()} set written", "success")


if __name__ == '__main__':
    main()
