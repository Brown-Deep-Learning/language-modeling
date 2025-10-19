#!/bin/bash
#SBATCH --nodes=1               # node count
#SBATCH -p gpu --gres=gpu:1     # number of gpus per node
#SBATCH --ntasks-per-node=1     # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -t 02:30:00             # total run time limit (HH:MM:SS)
#SBATCH --mem=16000MB           # Note: We do not explicitly request GPU memory, most GPUs on Oscar are more than sufficient for this assignment
#SBATCH --job-name='my_job'     # Note: You should change this name...
#SBATCH --output=slurm_logs/R-%x.%j.out
#SBATCH --error=slurm_logs/R-%x.%j.err

# Force unbuffered output (Useful for prints...)
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

module purge
unset LD_LIBRARY_PATH
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"

echo "Starting main Python script at $(date)"
srun apptainer exec --nv tensorflow-24.03-tf2-py3.simg python -u main.py
echo "Python script finished at $(date)"
