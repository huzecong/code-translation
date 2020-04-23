#!/usr/bin/env bash
PYTHONPATH=. python scripts/examine_output.py \
  --data-file data/processed/test_exclude.txt \
  --overlap-score-file data/processed/overlap_test_exclude.txt \
  --hyp-names "Small (100k it),Large (100k it),Large (760k it)" \
  --hyp-files "outputs/test_repos_excluded.hyp.orig,outputs/test_repos_excluded.hyp.100k,outputs/test_repos_excluded.hyp.760k" \
  --pickle-file outputs/test_output_excluded.pkl

PYTHONPATH=. python scripts/examine_output.py \
  --hyp-names "Small (100k it),Large (100k it),Large (760k it)" \
  --hyp-files "outputs/test_repos_included.hyp.orig,outputs/test_repos_included.hyp.100k,outputs/test_repos_included.hyp.760k" \
  --pickle-file outputs/test_output_included.pkl
