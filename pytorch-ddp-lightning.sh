export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

PYTHON3="python3"

if [ -n "$SIF" ]; then
    PYTHON3="singularity exec $SIF python3"
fi

echo "PYTHON3=$PYTHON3"

env | grep NCCL
env | grep MIOPEN

SCRIPT="benchmarks/pytorch_visionmodel_lightning.py"
IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

SCRIPT_OPTS="--strategy=ddp --warmup-steps 10 --workers=$SLURM_CPUS_PER_TASK"

if [ $(( $NUM_GPUS * $SLURM_NNODES )) -ne $SLURM_NTASKS ]; then
    echo "ERROR: this script needs to be run as one task per GPU. Try using slurm/*-mpi.sh scripts."
    echo "NUM_GPUS * SLURM_NNODES = $NUM_GPUS * $SLURM_NNODES != SLURM_NTASKS = $SLURM_NTASKS"
    exit 1
fi

if [ "$1" == "--data" ]; then
    shift
    (set -x
     srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
          tar xf $IMAGENET_DATA -C $LOCAL_SCRATCH
    )
    SCRIPT_OPTS="--datadir ${LOCAL_SCRATCH}/ilsvrc2012-torch $SCRIPT_OPTS"
fi

(set -x
 srun $PYTHON3 $SCRIPT --gpus=$NUM_GPUS --nodes=$SLURM_NNODES $SCRIPT_OPTS $*
)


