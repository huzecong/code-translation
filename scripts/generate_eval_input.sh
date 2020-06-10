#!/usr/bin/env bash
PYTHONPATH=. python scripts/examine_output.py \
  --hyp-names "Decompiled Var Names,Oracle Var Names,Decompiled + Fine-tune,Oracle + Fine-tune" \
  --overlap-score-files "data/processed/overlap_test.txt,data/processed/overlap_test.txt,data/processed/overlap_extra_test.txt,data/processed/overlap_extra_test.txt" \
  --hyp-files "outputs_decomp_varname/test_default.hyp.orig,outputs_orig_varname/test_default.hyp.orig,outputs_decomp_varname_finetune/test_default.hyp.orig,outputs_orig_varname_finetune/test_default.hyp.orig" \
  --pickle-file test_output.pkl \
  --output-file test.annotated
