#!/bin/bash
#SBATCH --job-name=pi_recursivo
## custom file name with job number using SLURM_ARRAY_JOB_ID
#SBATCH --output=output_pi_recursivo/%A/%a.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --mem=512MB
#SBATCH --array=1-500

# There are three different kinds of files: ./pi_recursivo, ./pi_recursivo_for and ./pi_recursivo_task
# Each file should be run 5 times, so we have 15 files * 5 runs = 75 jobs

# File names are ./pi_recursivo1, ./pi_recursivo2, through 

RANDOM_FILENUMBER=$((SLURM_ARRAY_TASK_ID % 5 + 1))
RANDOM_SCHEDULER=$((SLURM_ARRAY_TASK_ID % 3 + 1))

case $RANDOM_SCHEDULER in
    1)
        ./pi_recursivo_$RANDOM_FILENUMBER
        ;;
    2)
        ./pi_recursivo_for_$RANDOM_FILENUMBER
        ;;
    3)
        ./pi_recursivo_task_$RANDOM_FILENUMBER
        ;;
esac