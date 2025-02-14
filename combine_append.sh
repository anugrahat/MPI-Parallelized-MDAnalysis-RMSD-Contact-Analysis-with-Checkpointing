#!/bin/bash
#SBATCH --job-name=md-analysis-combine
#SBATCH --account=ucd191
#SBATCH --clusters=expanse
#SBATCH --partition=compute    
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=%x.o%j.%N


echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"

echo "Loading modules..."
module purge
module load slurm
module load cpu
module load gcc/10.2.0


CONDA_BASE="/cm/shared/apps/spack/0.17.3/gpu/b/opt/spack/linux-rocky8-skylake_avx512/gcc-8.5.0/anaconda3-2021.05-kfluefzsateihlamuk2qihp56exwe7si"
echo "Using Conda installation at: $CONDA_BASE"


source "$CONDA_BASE/etc/profile.d/conda.sh" || {
    echo "ERROR: Failed to source conda.sh"
    exit 1
}

echo "Activating MD_Analysis_New environment..."
conda activate MD_Analysis_New || {
    echo "ERROR: Failed to activate MD_Analysis_New environment"
    exit 1
}

echo "Conda Python:"
which python
python --version

#  replace '32' with the number of MPI ranks you used for the analysis.

echo "Starting combination of results..."
python append.py 32

echo "Job finished at: $(date)"
