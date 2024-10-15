#!/bin/bash
#SBATCH --job-name=montecarlo
#SBATCH --output=$BASENAME_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --mem=512MB

./montecarlo_parallel_randr_preseed