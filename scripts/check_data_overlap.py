from argtyped import Arguments
from bashplotlib.histogram import plot_hist
from tqdm import tqdm

import cotra

import sys
sys.path.append(".")
from examine_output import read_pairs


class Args(Arguments):
    data_pattern: str = "data/processed/{split}.txt"
    output_file: str = "data/processed/overlap_test.txt"


def main():
    args = Args()
    datasets = {
        split: read_pairs(args.data_pattern.format(split=split), decode=False)
        for split in ["train", "test"]
    }

    merged_dataset = sorted([(tgt, idx if key == "test" else -1)
                             for key, (src_data, tgt_data) in datasets.items()
                             for idx, tgt in enumerate(tgt_data)])

    cnt = 0
    progress = tqdm(merged_dataset)
    overlap_ids = []
    overlap_scores = {}
    for idx, (test_sent, key) in enumerate(progress):
        if key == -1: continue
        candidates = []
        left = next((i for i in range(idx - 1, -1, -1) if merged_dataset[i][1] == -1), None)
        right = next((i for i in range(idx + 1, len(merged_dataset)) if merged_dataset[i][1] == -1), None)
        if left: candidates.append(merged_dataset[left][0].split())
        if right: candidates.append(merged_dataset[right][0].split())
        if len(candidates) == 0:
            breakpoint()
            continue
        test_sent = test_sent.split()
        max_overlap, cand_sent = max((cotra.utils.lcs(test_sent, cand) / max(len(test_sent), len(cand)), cand)
                                     for cand in candidates)
        if max_overlap > 0.8:
            # progress.write(" ".join(test_sent))
            # progress.write(" ".join(cand_sent))
            cnt += 1
            progress.set_postfix(cnt=cnt)
            overlap_ids.append(key)
        overlap_scores[key] = max_overlap
    print(cnt)
    print(overlap_ids)
    if len(overlap_scores) != len(datasets["test"]):
        breakpoint()
    overlap_scores = [overlap_scores[idx] for idx in range(len(overlap_scores))]
    plot_hist(overlap_scores, height=10, pch="x", xlab=True, showSummary=True, bincount=70)

    with open(args.output_file, "w") as f:
        f.write("\n".join(map(str, overlap_scores)))


if __name__ == '__main__':
    main()
