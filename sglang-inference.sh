PYTHON3="python3"

if [ -n "$SIF" ]; then
    PYTHON3="singularity exec $SIF python3"
fi

echo "PYTHON3=$PYTHON3"

MODEL=meta-llama/Llama-3.2-3B-Instruct
DP=$(( NUM_GPUS  * SLURM_NNODES ))

if [ "$SLURM_NNODES" -gt 1 ]; then
    export DIST_OPTS="--dist-init-addr $(hostname):50000 --nodes=$SLURM_NNODES --rank=\$SLURM_PROCID"
fi

(set -x
 srun bash -c "$PYTHON3 -m sglang.bench_offline_throughput --model-path $MODEL --num-prompts 100 \
      --dp $DP $DIST_OPTS"
)
