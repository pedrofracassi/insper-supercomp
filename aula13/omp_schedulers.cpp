#include <algorithm>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <string>
#include <vector>
void print_iterations(const std::string &description,
                      const std::vector<std::vector<int>> &vectors,
                      const int n)
{
    std::vector<std::string> strings(4, std::string());
    for (int i = 0; i != n; i++)
    {
        for (int j = 0; j != 4; j++)
        {
            const auto &vector = vectors[j];
            auto it = std::find(vector.begin(), vector.end(), i);
            if (it != vector.end())
            {
                strings[j] += "*";
            }
            else
            {
                strings[j] += " ";
            }
        }
    }
    std::cout << description << std::endl;
    for (auto &s : strings)
    {
        std::cout << s << "\n";
    }
    std::cout << std::endl;
}
template <typename T>
void schedule(T function,
              const std::string &description,
              const int n)
{
    std::vector<std::vector<int>> vectors(4, std::vector<int>());
    double init_time, final_time, avg_time, variancia;
    std::vector<double> times;

    int maxIters = 1000000;

    int i = 0;
    for (int i = 0; i < maxIters; i++)
    {
        init_time = omp_get_wtime();
        function(vectors, n);
        final_time = omp_get_wtime() - init_time;
        times.push_back(final_time);
        avg_time += final_time / maxIters;
    }

    for (const auto &time : times)
    {
        variancia += (time - avg_time) * (time - avg_time);
    }
    variancia /= maxIters;

    double std_dev = sqrt(variancia);

    // print_iterations(description, vectors, n);
    std::cout << description << avg_time << " secs // " << std_dev << std::endl;

    avg_time;
}
void scheduleDefault(std::vector<std::vector<int>> &vectors,
                     int n)
{
#pragma omp parallel num_threads(4), shared(vectors, n)
    {
#pragma omp for
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}
void scheduleStatic(std::vector<std::vector<int>> &vectors,
                    int n)
{
#pragma omp parallel num_threads(4), shared(vectors, n)
    {
#pragma omp for schedule(static)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}
void scheduleStatic4(std::vector<std::vector<int>> &vectors,
                     int n)
{
#pragma omp parallel num_threads(4), shared(vectors, n)
    {
#pragma omp for schedule(static, 4)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}
void scheduleStatic8(std::vector<std::vector<int>> &vectors,
                     int n)
{
#pragma omp parallel num_threads(4), shared(vectors, n)
    {
#pragma omp for schedule(static, 8)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}
void scheduleDynamic(std::vector<std::vector<int>> &vectors,
                     int n)
{
#pragma omp parallel num_threads(4), shared(vectors, n)
    {
#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}
void scheduleDynamic1(std::vector<std::vector<int>> &vectors,
                      int n)
{
#pragma omp parallel num_threads(4), shared(vectors, n)
    {
#pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}
void scheduleDynamic4(std::vector<std::vector<int>> &vectors,
                      int n)
{
#pragma omp parallel num_threads(4), shared(vectors, n)
    {
#pragma omp for schedule(dynamic, 4)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}
void scheduleDynamic8(std::vector<std::vector<int>> &vectors,
                      int n)
{
#pragma omp parallel num_threads(4), shared(vectors, n)
    {
#pragma omp for schedule(dynamic, 8)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}
void scheduleGuided(std::vector<std::vector<int>> &vectors,
                    int n)
{
#pragma omp parallel num_threads(4), shared(vectors, n)
    {
#pragma omp for schedule(guided)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}
void scheduleGuided2(std::vector<std::vector<int>> &vectors,
                     int n)
{
#pragma omp parallel num_threads(4), shared(vectors, n)
    {
#pragma omp for schedule(guided, 2)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}
void scheduleGuided4(std::vector<std::vector<int>> &vectors,
                     int n)
{
#pragma omp parallel num_threads(4), shared(vectors, n)
    {
#pragma omp for schedule(guided, 4)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}
void scheduleGuided8(std::vector<std::vector<int>> &vectors,
                     int n)
{
#pragma omp parallel num_threads(4), shared(vectors, n)
    {
#pragma omp for schedule(guided, 8)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}
void scheduleAuto(std::vector<std::vector<int>> &vectors,
                  int n)
{
#pragma omp parallel num_threads(4), shared(vectors, n)
    {
#pragma omp for schedule(auto)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}
void scheduleRuntime(std::vector<std::vector<int>> &vectors,
                     int n)
{
#pragma omp parallel num_threads(4), shared(vectors, n)
    {
#pragma omp for schedule(auto)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}

int main()
{
    const int n = 64;
    schedule(scheduleDefault, "default:               ", n);
    schedule(scheduleStatic, "schedule(static):      ", n);
    schedule(scheduleStatic4, "schedule(static, 4):   ", n);
    schedule(scheduleStatic8, "schedule(static, 8):   ", n);
    schedule(scheduleDynamic, "schedule(dynamic):     ", n);
    schedule(scheduleDynamic1, "schedule(dynamic, 1):  ", n);
    schedule(scheduleDynamic4, "schedule(dynamic, 4):  ", n);
    schedule(scheduleDynamic8, "schedule(dynamic, 8):  ", n);
    schedule(scheduleGuided, "schedule(guided):      ", n);
    schedule(scheduleGuided2, "schedule(guided, 2):   ", n);
    schedule(scheduleGuided4, "schedule(guided, 4):   ", n);
    schedule(scheduleGuided8, "schedule(guided, 8):   ", n);
    schedule(scheduleAuto, "schedule(auto):        ", n);
    schedule(scheduleRuntime, "schedule(runtime):     ", n);
    return 0;
}