#!/bin/bash
#SBATCH --job-name=md-analysis-mpi
#SBATCH --account=ucd191
#SBATCH --clusters=expanse
#SBATCH --partition=compute    
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=%x.o%j.%N

# Print job information
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"

# Load required modules
echo "Loading modules..."
module purge
module load slurm
module load cpu
module load gcc/10.2.0

# Set Conda paths (using your specific path)
CONDA_BASE="/cm/shared/apps/spack/0.17.3/gpu/b/opt/spack/linux-rocky8-skylake_avx512/gcc-8.5.0/anaconda3-2021.05-kfluefzsateihlamuk2qihp56exwe7si"
echo "Using Conda installation at: $CONDA_BASE"

# Source conda
source $CONDA_BASE/etc/profile.d/conda.sh || {
    echo "ERROR: Failed to source conda.sh"
    exit 1
}

# Activate the environment
echo "Activating MD_Analysis_New environment..."
conda activate MD_Analysis_New || {
    echo "ERROR: Failed to activate MD_Analysis_New environment"
    exit 1
}

# Print environment info
which python
python --version
python -c "from mpi4py import MPI; print(f'MPI version: {MPI.Get_version()}')"

# Create checkpoint directory
mkdir -p checkpoints

# Run MPI job using srun
echo -e "\nStarting MPI job..."
export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=1

srun --mpi=pmi2 python -u pmf_check_point.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Job completed successfully at: $(date)"
else
    echo "Job failed at: $(date)"
    exit 1
fi