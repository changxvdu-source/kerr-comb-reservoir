#!/bin/sh
# Every table of the paper under one protocol: 256 operating points per encoding (128 for the mask
# ablations), choice among the best 16 by the mean validation error on 3 pairs of new runs, report on
# 6 further pairs of new runs. Most important first; spectra already simulated come from the cache.
# Run from the project folder with the GPU interpreter:  sh experiments/run_robust_batch.sh
PY=${PY:-D:/jupyder-gpu/Scripts/python.exe}
COMMON="--device cuda --chunk 512 --series-seeds 0 1 2 --no-fine-grid --fresh-pairs 6 --select-top 16 --select-pairs 3 --variants 31x1 31x8 --main 31x8 --cache results/encodings_spectra --workers 12"
ABL="comb_uniform comb_static comb_amplitude comb_phase comb_mask4 comb_mask8 comb_mask comb_mask32 comb_mask64"

run() { name=$1; shift; if grep -q "^saved" "results/$name.log" 2>/dev/null; then echo "skip $name (already done)"; return; fi; echo "$(date '+%F %T') start $name"; "$PY" experiments/compare_encodings.py $COMMON --name "$name" "$@" > "results/$name.log" 2>&1; echo "$(date '+%F %T') done $name (exit $?)"; }

run robust_noisy_mg       --trials 256 --tasks mackey_glass_h10 --noise realistic
run robust_noisy_ce       --trials 256 --tasks channel_eq --noise realistic
run robust_ablation_mg    --trials 128 --tasks mackey_glass_h10 --noise realistic --groups $ABL
run robust_ablation_ce    --trials 128 --tasks channel_eq --noise realistic --groups $ABL
# NARMA-10 needs a readout memory of 12 symbols (see encodings_narma_taps): main readout 31x12.
NARMA="--tasks narma10 --variants 31x1 31x8 31x12 --main 31x12"
run robust_noisy_narma    --trials 256 --noise realistic $NARMA
# 4 scoring workers: each worker loads torch with CUDA (~1.3 GB of commit), 12 exhausted the page file
run robust_clean          --trials 256 --tasks mackey_glass_h10 channel_eq --noise none --workers 4
run robust_clean_narma    --trials 256 --noise none $NARMA --workers 4
echo "$(date '+%F %T') all done"
