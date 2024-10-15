# Generate list of multipliers from 1 to 10
for i in {1..5}
do
  g++ -fopenmp pi_recursivo.cpp -o pi_recursivo_$i -DMULTIPLIER=$i
  g++ -fopenmp pi_recursivo_task.cpp -o pi_recursivo_task_$i -DMULTIPLIER=$i
  g++ -fopenmp pi_recursivo_for.cpp -o pi_recursivo_for_$i -DMULTIPLIER=$i
done