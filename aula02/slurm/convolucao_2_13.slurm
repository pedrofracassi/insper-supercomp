#!/bin/bash
#SBATCH --job-name=c_2_13           # Nome do job
#SBATCH --output=./out/magic.csv           # Nome do arquivo de saída
#SBATCH --ntasks=2                              # Número de tarefas (4 processos)
#SBATCH --cpus-per-task=13                       # Número de threads por tarefa (2 threads por processo)
#SBATCH --mem=2048                              # Memória total alocada por nó (1024 MB = 1 GB)
#SBATCH --time=00:10:00                         # Tempo máximo de execução (10 minutos)
#SBATCH --partition=espec                       # fila do cluster a ser utilizada

# Exporta a variável de ambiente para configurar o número de threads OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Executa o programa usando MPI e OpenMP
mpirun ./convolucao