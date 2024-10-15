#include <iostream>
#include <unistd.h>
#include <vector>

double conta_complexa(int i)
{
    return 2 * i;
}

#include <chrono>

int main()
{
  int N = 10000;
  std::vector<double> vec;


  auto start = std::chrono::high_resolution_clock::now();

  vec.resize(N);

#pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    {
        vec[i] = conta_complexa(i);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  for (int i = 0; i < N; i++)
  {
    std::cout << i << " ";
  }

  std::cout << "\nExecution time: " << elapsed.count() << " seconds\n";

  return 0;
}