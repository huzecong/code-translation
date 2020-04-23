#!/usr/bin/env bash
scp metis:~/metis1/code-translation/outputs/test_output*.pkl .
for type in "included" "excluded"; do
  mv "test_output_${type}.pkl" "full_${type}_eval"
  python scripts/eval_output.py --test-file "full_${type}_eval/test_output_${type}.pkl" --output-dir "full_${type}_eval"
done
for type in "included" "excluded"; do
  scp "full_${type}_eval/eval.html" "linux-gp:~/www/eval-${type}.html"
  scp "full_${type}_eval/eval-small.html" "linux-gp:~/www/eval-${type}-small.html"
done
