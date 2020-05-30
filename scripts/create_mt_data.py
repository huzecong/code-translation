import functools
import json
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set

import argtyped
import flutes
import ghcc
import tqdm
import ujson
from argtyped import Switch


class Arguments(argtyped.Arguments):
    input_dir: str = "match_output/"
    output_dir: str = "mt_data/"
    max_repos: Optional[int]
    quiet: Switch = False
    n_procs: int = 4
    queue_size: int = 1024
    block_size: int = 10000


class RepoInfo(NamedTuple):
    repo_owner: str
    repo_name: str


class _Example(NamedTuple):
    decompiled_code: str
    original_code: str
    decompiled_ast: Dict[str, Any]
    original_ast: Dict[str, Any]
    repo: str  # "owner/name"
    sha: str


SEP = "\0"
END_SIGNATURE = b"END_REPO"
PICKLE_PROTOCOL = 4

QueueElem = bytes


def convert_ast(node):
    # `bytes` are smaller when pickled; all keys are ASCII.
    # However, protocol 4 introduced a "short-string" opcode, so this is no longer necessary.
    if isinstance(node, dict):
        ret = {k.encode("ascii"): convert_ast(v) for k, v in node.items()}
        typ = ret.get(b"_t", None)
        if typ is not None:
            ret[b"_t"] = typ.encode("ascii")
        return ret
    elif isinstance(node, list):
        return [convert_ast(c) for c in node]
    else:
        return node


def convert_code(code: List[str]) -> str:
    code_str = SEP.join(code)
    return code_str


def exception_handler(e: Exception, repo_info: RepoInfo, queue: 'mp.Queue[QueueElem]'):
    repo = f"{repo_info.repo_owner}/{repo_info.repo_name}"
    flutes.log_exception(e, f"Exception occurred when processing {repo}", force_console=True)
    queue.put(END_SIGNATURE)


@flutes.exception_wrapper(exception_handler)
def process(repo_info: RepoInfo, data_dir: str, queue: 'mp.Queue[QueueElem]') -> None:
    repo = f"{repo_info.repo_owner}/{repo_info.repo_name}"
    with open(os.path.join(data_dir, repo_info.repo_owner, repo_info.repo_name, "matched_funcs.jsonl")) as f:
        for line in f:
            if not line:
                continue
            try:
                matched_func = ujson.loads(line)
            except ValueError:
                # `ujson` has a hard-coded depth limit of 1024. If limit is reached, fallback to built-in `json`.
                matched_func = json.loads(line)
            decompiled_code = convert_code(matched_func['decompiled_tokens'])
            original_code = convert_code(matched_func['original_tokens'])
            var_names = {k: (decomp, orig) for k, [decomp, orig] in matched_func['variable_names'].items()}
            sha = matched_func['binary_hash']
            # Dump it here; otherwise the queue thread will do all the dumping.
            example = pickle.dumps((decompiled_code, original_code, var_names, repo, sha),
                                   protocol=PICKLE_PROTOCOL)
            queue.put(example)
    queue.put(END_SIGNATURE)


def main():
    # flutes.register_ipython_excepthook()

    sys.setrecursionlimit(50000)
    args = Arguments()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flutes.log("Dataset creation start")

    db = ghcc.MatchFuncDB()
    original_code_set: Set[str] = set()
    n_duplicate = 0
    n_examples = 0
    manager = mp.Manager()
    example_queue: 'mp.Queue[QueueElem]' = manager.Queue(args.queue_size)
    with flutes.safe_pool(args.n_procs, closing=[db]) as pool:
        repos = [RepoInfo(entry['repo_owner'], entry['repo_name'])
                 for entry in db.collection.find() if entry['funcs_matched'] > 0]
        if args.max_repos is not None:
            repos = repos[:args.max_repos]
        process_fn: Callable[[RepoInfo], None] = functools.partial(
            process, data_dir=args.input_dir, queue=example_queue)
        pool.map_async(process_fn, repos, error_callback=flutes.log_exception)
        end_signals = 0
        progress = tqdm.tqdm(total=len(repos))
        file_cnt = 0
        text_data = []

        def save_file():
            nonlocal file_cnt, text_data
            # Save text & AST separately
            with (output_dir / f"data_{file_cnt:03d}.pkl").open("wb") as f:
                pickle.dump(text_data, f, protocol=PICKLE_PROTOCOL)
            progress.write(f"Saved part {file_cnt:03d}")
            text_data = []
            file_cnt += 1

        while end_signals < len(repos):
            elem = example_queue.get()
            if elem == END_SIGNATURE:
                progress.update(1)
                end_signals += 1
                continue

            ex = pickle.loads(elem)
            original_code = ex[1]
            if original_code not in original_code_set:
                original_code_set.add(original_code)
                text_data.append(ex)  # (decompiled, orig, var_names, repo, sha)
                n_examples += 1
            else:
                n_duplicate += 1
            progress.set_postfix({"duplicate": n_duplicate, "examples": n_examples}, refresh=False)
            if (n_examples + n_duplicate) % 100 == 0:
                progress.refresh()
            if len(text_data) >= args.block_size:
                save_file()

        if len(text_data) > 0:
            save_file()


if __name__ == '__main__':
    main()
