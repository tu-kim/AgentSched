#!/usr/bin/env bash
# Nsight Compute wrapper: measured DRAM bytes + achieved FLOPs for one config
# subset. Very slow (kernel replay) — run on a few representative configs.
# Usage: bash bench/ncu.sh "--exp exp1 --model meta-llama/Llama-3.1-8B-Instruct"
set -euo pipefail
ARGS="${1:-"--exp exp4"}"
OUT="results/ncu_$(echo "$ARGS" | tr ' /' '__').csv"
ncu --target-processes all \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum \
    --csv --log-file "$OUT" \
    python -m bench.runner $ARGS --trials 1 --warmup 0 --enforce-eager
echo "wrote $OUT  (measured_DRAM_bytes = dram read+write summed over kernels;"
echo " achieved FLOPs ~= 2 * (hfma+ffma) insts * 2/warp-lane-correction — see analyze.py)"
