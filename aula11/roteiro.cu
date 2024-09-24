#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

struct variance_functor
{
  const float mean;
  variance_functor(float _mean) : mean(_mean) {}
  __host__ __device__ float operator()(const float x) const
  {
    float diff = x - mean;
    return diff * diff;
  }
};

struct adjacent_difference
{
  __host__ __device__ float operator()(const thrust::tuple<float, float> &t) const
  {
    return thrust::get<1>(t) - thrust::get<0>(t);
  }
};

struct is_positive : public thrust::unary_function<float, bool>
{
  __host__ __device__ bool operator()(const float x) const
  {
    return x > 0;
  }
};

int main()
{
  auto start = std::chrono::steady_clock::now();

  std::ifstream file("stocks-google.txt");
  std::vector<float> stocks_host;
  float price;
  while (file >> price)
  {
    stocks_host.push_back(price);
  }

  std::cout << "Tamanho dos dados: " << stocks_host.size() << std::endl;

  thrust::device_vector<float> stocks(stocks_host);

  auto transfer_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> transfer_time = transfer_end - start;
  std::cout << "Tempo de transferência para GPU: " << transfer_time.count() << " segundos" << std::endl;

  auto variance_start = std::chrono::steady_clock::now();

  float mean = thrust::reduce(stocks.begin(), stocks.end()) / stocks.size();

  float variance = thrust::transform_reduce(
                       stocks.begin(), stocks.end(),
                       variance_functor(mean),
                       0.0f,
                       thrust::plus<float>()) /
                   stocks.size();

  auto variance_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> variance_time = variance_end - variance_start;
  std::cout << "Variância: " << variance << std::endl;
  std::cout << "Tempo de cálculo da variância: " << variance_time.count() << " segundos" << std::endl;

  auto daily_gain_start = std::chrono::steady_clock::now();

  thrust::device_vector<float> ganho_diario(stocks.size() - 1);
  thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(stocks.begin(), stocks.begin() + 1)),
      thrust::make_zip_iterator(thrust::make_tuple(stocks.end() - 1, stocks.end())),
      ganho_diario.begin(),
      adjacent_difference());

  auto daily_gain_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> daily_gain_time = daily_gain_end - daily_gain_start;
  std::cout << "Tempo de cálculo do ganho diário: " << daily_gain_time.count() << " segundos" << std::endl;

  auto count_start = std::chrono::steady_clock::now();

  int dias_aumento = thrust::count_if(ganho_diario.begin(), ganho_diario.end(), is_positive());

  auto count_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> count_time = count_end - count_start;
  std::cout << "Dias com aumento no preço: " << dias_aumento << std::endl;
  std::cout << "Tempo de contagem: " << count_time.count() << " segundos" << std::endl;

  auto avg_increase_start = std::chrono::steady_clock::now();

  thrust::replace_if(ganho_diario.begin(), ganho_diario.end(), thrust::not1(is_positive()), 0.0f);
  float soma_aumentos = thrust::reduce(ganho_diario.begin(), ganho_diario.end());
  float aumento_medio = soma_aumentos / dias_aumento;

  auto avg_increase_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> avg_increase_time = avg_increase_end - avg_increase_start;
  std::cout << "Aumento médio nos dias de alta: " << aumento_medio << std::endl;
  std::cout << "Tempo de cálculo do aumento médio: " << avg_increase_time.count() << " segundos" << std::endl;

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> total_time = end - start;
  std::cout << "Tempo total de execução: " << total_time.count() << " segundos" << std::endl;

  return 0;
}