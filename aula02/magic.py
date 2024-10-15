import subprocess

max_ntasks = 4
max_cpus_per_task = 16

f = open("convolucao_template.slurm", "r")
base_text = f.read()

for ntasks in range(1, max_ntasks+1):
  for cpus_per_task in range(1, max_cpus_per_task+1):
    text = base_text
    f = open(f"slurm/convolucao_{ntasks}_{cpus_per_task}.slurm", "w")

    text = text.replace("{NTASKS}", str(ntasks))
    text = text.replace("{CPUS_PER_TASK}", str(cpus_per_task))

    f.write(text)
    f.close()

    subprocess.run(["sbatch", f"slurm/convolucao_{ntasks}_{cpus_per_task}.slurm"]) 