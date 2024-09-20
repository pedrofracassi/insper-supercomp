#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <string>
#include <algorithm>

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

  int total_days = h_prices.size();
  std::cout << "Número de preços lidos: " << total_days << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  thrust::device_vector<float> d_prices = h_prices;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  std::cout << "Tempo de alocação e cópia para a GPU: " << elapsed.count() << " ms" << std::endl;

  // media ultimos 10 dias
  float sum = thrust::reduce(d_prices.begin(), d_prices.end());
  float avg_price = sum / total_days;
  std::cout << "Preço médio das ações nos últimos 10 anos: $" << avg_price << std::endl;

  // media ultimo ano
  int last_year_days = std::min(365, total_days);
  sum = thrust::reduce(d_prices.end() - last_year_days, d_prices.end());
  float avg_price_last_year = sum / last_year_days;
  std::cout << "Preço médio das ações no último ano: $" << avg_price_last_year << std::endl;

  // ultimos 10 anos
  auto minmax_total = thrust::minmax_element(d_prices.begin(), d_prices.end());
  float min_price_total = *minmax_total.first;
  float max_price_total = *minmax_total.second;
  std::cout << "Menor preço nos últimos 10 anos: $" << min_price_total << std::endl;
  std::cout << "Maior preço nos últimos 10 anos: $" << max_price_total << std::endl;

  // ultimo ano
  auto minmax_last_year = thrust::minmax_element(d_prices.end() - last_year_days, d_prices.end());
  float min_price_last_year = *minmax_last_year.first;
  float max_price_last_year = *minmax_last_year.second;
  std::cout << "Menor preço no último ano: $" << min_price_last_year << std::endl;
  std::cout << "Maior preço no último ano: $" << max_price_last_year << std::endl;

  return 0;
}