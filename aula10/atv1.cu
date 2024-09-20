#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <string>

int main()
{
  std::ifstream file("stocks-google.txt");
  thrust::host_vector<float> h_prices;
  float price;

  while (file >> price)
  {
    h_prices.push_back(price);
  }
  file.close();

  std::cout << "Número de preços lidos: " << h_prices.size() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  thrust::device_vector<float> d_prices = h_prices;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  std::cout << "Tempo de alocação e cópia para a GPU: " << elapsed.count() << " ms" << std::endl;

  return 0;
}