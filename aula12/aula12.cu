#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <iostream>
#include <chrono>

struct variance_op
{
  float mean;
  variance_op(float mean_) : mean(mean_) {}

  __host__ __device__ float operator()(const float &x) const
  {
    float diff = x - mean;
    return diff * diff;
  }
};

float calculate_mean(const thrust::device_vector<float> &d_vec)
{
  return thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>()) / d_vec.size();
}

float calculate_variance_fusion(const thrust::device_vector<float> &d_vec, float mean)
{
  return thrust::transform_reduce(d_vec.begin(), d_vec.end(), variance_op(mean), 0.0f, thrust::plus<float>()) / d_vec.size();
}

float calculate_variance_separate(const thrust::device_vector<float> &d_vec, float mean)
{
  thrust::device_vector<float> d_squared_diff(d_vec.size());
  thrust::transform(d_vec.begin(), d_vec.end(), d_squared_diff.begin(), variance_op(mean));
  return thrust::reduce(d_squared_diff.begin(), d_squared_diff.end(), 0.0f, thrust::plus<float>()) / d_vec.size();
}

int main()
{
  const int N = 1000000;
  thrust::device_vector<float> d_vec(N);

  thrust::default_random_engine rng(std::chrono::steady_clock::now().time_since_epoch().count());
  thrust::uniform_real_distribution<float> dist(0, 100);

  thrust::generate(d_vec.begin(), d_vec.end(),
                   [=] __device__()
                   {
                     thrust::default_random_engine rng;
                     thrust::uniform_real_distribution<float> dist(0, 100);
                     return dist(rng);
                   });

  // cálculo da média
  float mean = calculate_mean(d_vec);
  std::cout << "Média: " << mean << std::endl;

  // fusion kernel
  auto start_fusion = std::chrono::steady_clock::now();
  float var_fusion = calculate_variance_fusion(d_vec, mean);
  auto end_fusion = std::chrono::steady_clock::now();

  // etapas separadas
  auto start_separate = std::chrono::steady_clock::now();
  float var_separate = calculate_variance_separate(d_vec, mean);
  auto end_separate = std::chrono::steady_clock::now();

  std::cout << "Variância (Fusion Kernel): " << var_fusion << std::endl;
  std::cout << "Variância (Etapas Separadas): " << var_separate << std::endl;

  auto duration_fusion = std::chrono::duration_cast<std::chrono::microseconds>(end_fusion - start_fusion);
  auto duration_separate = std::chrono::duration_cast<std::chrono::microseconds>(end_separate - start_separate);

  std::cout << "Tempo de execução (Fusion Kernel): " << duration_fusion.count() << " microssegundos" << std::endl;
  std::cout << "Tempo de execução (Etapas Separadas): " << duration_separate.count() << " microssegundos" << std::endl;

  return 0;
}