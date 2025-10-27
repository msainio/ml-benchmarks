export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

PYTHON3="python3"

if [ -n "$SIF" ]; then
    PYTHON3="singularity exec $SIF python3"
fi

echo "PYTHON3=$PYTHON3"

env | grep NCCL
env | grep MIOPEN

SCRIPT="benchmarks/run_clm.py"

HF_MODEL=EleutherAI/gpt-neo-125M
# HF_MODEL=EleutherAI/gpt-neo-1.3B
export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/$USER/hf-home
export TORCH_HOME=/scratch/$SLURM_JOB_ACCOUNT/$USER/torch-cache

export TOKENIZERS_PARALLELISM=false

if [ "$SLURM_NTASKS" -ne "$SLURM_NNODES" ]; then
    echo "ERROR: this script needs to be run as one task per node."
    echo "SLURM_NNODES = $SLURM_NNODES != SLURM_NTASKS = $SLURM_NTASKS"
    exit 1
fi

if [ -z "$LOCAL_SCRATCH" ]; then
    OUTPUT_DIR="/flash/$SLURM_JOB_ACCOUNT/$USER/run-clm/$SLURM_JOB_ID"
else
    OUTPUT_DIR="$LOCAL_SCRATCH/run-clm/$SLURM_JOB_ID"
fi

SCRIPT_OPTS="--gradient_accumulation_steps $(( 64 / $NUM_GPUS / $SLURM_NNODES ))"

# SCRIPT_OPTS="--gradient_accumulation_steps 8"

NUM_WORKERS=$(( SLURM_CPUS_PER_TASK / NUM_GPUS ))

DIST_OPTS="--standalone --master_port 0"

if [ "$SLURM_NNODES" -gt 1 ]; then
    export RDZV_HOST=$(hostname)
    export RDZV_PORT=29400
    DIST_OPTS="--rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT"
fi

(set -x
 srun $PYTHON3 -m torch.distributed.run $DIST_OPTS \
      --nnodes=$SLURM_NNODES --nproc_per_node=$NUM_GPUS $SCRIPT \
      --model_name_or_path $HF_MODEL \
      --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
      --per_device_train_batch_size 2 --do_train \
      --output_dir $OUTPUT_DIR --overwrite_output_dir \
      --fp16 \
      --num_train_epochs 1 --dataloader_num_workers $NUM_WORKERS \
      $SCRIPT_OPTS $*
 )

rm -rf $OUTPUT_DIR
