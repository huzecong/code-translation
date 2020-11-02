import pickle
import random
from pathlib import Path
from typing import Dict

import flutes
from argtyped import *
from tqdm import tqdm

from cotra.data import InputData


class Args(Arguments):
    tranx_data_dir: str = "../tranX/tranx_data/"
    text_data_dir: str = "data_noncanonical/"
    output_path: str = "data_canonical/"
    n_procs: int = 4
    random_seed: int = 19260817
    pdb: Switch = False


TOKEN_SEP = "\0"
TUPLE_SEP = "\1"


class State(flutes.PoolState):
    def __init__(self):
        self.canonical_map: Dict[str, str] = {}

    def clear(self):
        self.canonical_map = {}

    def process_data(self, input_path: Path) -> None:
        with input_path.open("rb") as f:
            data = pickle.load(f)
        for ex in data:
            _src, tgt, _, _, meta = ex
            src = _src.replace("\1", TOKEN_SEP)
            canonical_tgt = tgt.replace("\1", TOKEN_SEP)
            raw_tgt = meta['raw_tgt_code'].replace("\1", TOKEN_SEP)
            var_names = meta['var_names']
            repo = meta['repo']
            sha = meta['hash']
            encoded_example = InputData.encode(src, canonical_tgt, var_names, 0.0, repo, sha)
            self.canonical_map[raw_tgt] = encoded_example


def main():
    args = Args()
    print(args.to_string())
    random.seed(args.random_seed)
    if args.pdb:
        flutes.register_ipython_excepthook()

    splits = {
        "train_extra": "train_extra/",
        "valid": "dev/",
        "test": "test/",
        "train": ".",
    }

    tranx_data_dir = Path(args.tranx_data_dir)
    text_data_dir = Path(args.text_data_dir)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with flutes.safe_pool(args.n_procs, state_class=State) as pool:
        for key, path in splits.items():
            pool.broadcast(State.clear)
            files = [file for file in sorted(Path(tranx_data_dir / path).iterdir())
                     if file.name.startswith("data") and file.suffix == ".pkl"]
            for _ in tqdm(pool.imap_unordered(State.process_data, files), total=len(files)):
                pass
            canonical_tgt_map: Dict[str, str] = {}
            total_size = 0
            for state_map in pool.get_states():
                total_size += len(state_map.canonical_map)
                canonical_tgt_map.update(state_map.canonical_map)
            assert total_size == len(canonical_tgt_map)
            print(f"{key.capitalize()} set processed")
            in_path = text_data_dir / f"{key}.txt"
            out_path = output_dir / f"{key}.txt"
            not_found = set()
            with in_path.open("r") as fin, out_path.open("w") as fout:
                progress = tqdm(fin, total=total_size)
                for line in progress:
                    if not line: continue
                    example = InputData.decode(line)
                    encoded_output = canonical_tgt_map.get(example.original_code, None)
                    if encoded_output is None:
                        not_found.add(example.original_code)
                        progress.set_postfix(not_found=len(not_found))
                    else:
                        del canonical_tgt_map[example.original_code]
                        fout.write(encoded_output)
                        fout.write("\n")
                print(f"{len(not_found)} not found, {len(canonical_tgt_map)} leftover")
                for encoded_output in canonical_tgt_map.values():
                    fout.write(encoded_output)
                    fout.write("\n")
                print(f"{key.capitalize()} set written to {out_path}")


if __name__ == '__main__':
    main()
