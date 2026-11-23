#!/bin/sh
# Linear-cavity control (gamma = 0) for the mask and the uniform comb, under the protocol of the paper's Table 1:
# 256 operating points (the same points and masks as the Kerr groups), choice among the best 16 on 3 pairs of new
# runs, report on 6 further pairs, 3 series. Run from the project folder:  sh experiments/run_linear_control.sh [names]
# (the clean runs were done on Colab: results/colab_linear_clean_{mg,ce}*)
PY=${PY:-D:/jupyder-gpu/Scripts/python.exe}
COMMON="--device cuda --chunk 512 --series-seeds 0 1 2 --no-fine-grid --fresh-pairs 6 --select-top 16 --select-pairs 3 --variants 31x1 31x8 --main 31x8 --cache results/linear_spectra --workers 4 --trials 256 --groups lin_comb_mask lin_comb_uniform"

run() { name=$1; shift; if grep -q "^saved" "results/$name.log" 2>/dev/null; then echo "skip $name (already done)"; return; fi; echo "$(date '+%F %T') start $name"; "$PY" experiments/compare_encodings.py $COMMON --name "$name" "$@" > "results/$name.log" 2>&1; echo "$(date '+%F %T') done $name (exit $?)"; }

run linear_noisy_mg  --tasks mackey_glass_h10 --noise realistic
run linear_noisy_ce  --tasks channel_eq --noise realistic
echo "$(date '+%F %T') all done"
