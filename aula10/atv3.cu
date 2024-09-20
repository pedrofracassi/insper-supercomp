#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// Functor para calcular a diferença entre dois preços
struct price_difference : public thrust::binary_function<float, float, float>
{
  __host__ __device__ float operator()(float apple_price, float microsoft_price) const
  {
    return apple_price - microsoft_price;
  }
};

int main()
{
  std::ifstream file("stocks2.txt");
  thrust::host_vector<float> h_apple_prices;
  thrust::host_vector<float> h_microsoft_prices;

  if (!file.is_open())
  {
    std::cerr << "Erro ao abrir o arquivo!" << std::endl;
    return 1;
  }

  std::string line;
  while (std::getline(file, line))
  {
    std::istringstream iss(line);
    float apple_price, microsoft_price;
    char comma;
    if (iss >> apple_price >> comma >> microsoft_price)
    {
      h_apple_prices.push_back(apple_price);
      h_microsoft_prices.push_back(microsoft_price);
    }
  }
  file.close();

  int num_days = h_apple_prices.size();
  std::cout << "Número de dias processados: " << num_days << std::endl;

  thrust::device_vector<float> d_apple_prices = h_apple_prices;
  thrust::device_vector<float> d_microsoft_prices = h_microsoft_prices;

  thrust::device_vector<float> d_price_differences(num_days);
  thrust::transform(d_apple_prices.begin(), d_apple_prices.end(),
                    d_microsoft_prices.begin(),
                    d_price_differences.begin(),
                    price_difference());

  float sum_differences = thrust::reduce(d_price_differences.begin(), d_price_differences.end());
  float avg_difference = sum_differences / num_days;

  std::cout << "Diferença média de preço (Apple - Microsoft): $" << avg_difference << std::endl;

  return 0;
}