import shutil
from pathlib import Path
from typing import NamedTuple

import flutes
from argtyped import Arguments
from tqdm import tqdm


class Args(Arguments):
    binary_list: str  # list of (repo, sha)
    output_dir: str

    binary_path_pattern: str = "binaries/{repo}/{sha}"
    decompiled_path_pattern: str = "decompile_output_fixed/{sha}.jsonl"
    matched_func_path_pattern: str = "match_output/{repo}/matched_funcs.jsonl"
    preprocessed_code_path_pattern: str = "match_output/{repo}/{sha}.c"


class FileDescription(NamedTuple):
    name: str
    folder: str
    pattern: str
    filename: str


def main():
    args = Args()
    flutes.register_ipython_excepthook()
    with open(args.binary_list) as f:
        binaries = [line.split() for line in f if line]
    output_dir = Path(args.output_dir)

    file_descriptions = [
        FileDescription(name="binaries",
                        folder="binaries", pattern=args.binary_path_pattern, filename="{sha}"),
        FileDescription(name="decompiled output",
                        folder="decompiled", pattern=args.decompiled_path_pattern, filename="{sha}.jsonl"),
        # FileDescription(name="matched functions",
        #                 folder="matched_funcs", pattern=args.matched_func_path_pattern, filename="{sha}.jsonl"),
        FileDescription(name="preprocessed code",
                        folder="code", pattern=args.preprocessed_code_path_pattern, filename="{sha}.c"),
    ]

    for desc in file_descriptions:
        output_folder = output_dir / desc.folder
        output_folder.mkdir(exist_ok=True, parents=True)
        for repo, sha in tqdm(binaries, desc=f"Copying {desc.name}"):
            shutil.copy(desc.pattern.format(repo=repo, sha=sha), output_folder / desc.filename.format(sha=sha))


if __name__ == '__main__':
    main()
