#include <iostream>
#include <omp.h>
#include <vector>

double areaSize = 1.0;


bool isWithinCircle(double x, double y)
{
    return x * x + y * y <= areaSize * areaSize;
}

int main()
{
    int N = 1000000000;

    int inside = 0;
    int outside = 0;

    double init_time, final_time;
    init_time = omp_get_wtime();
    std::vector<int> vec;
    int num_threads;

    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }

    vec.resize(num_threads);

    srand(time(NULL));
    int preseed = rand();
    std::cout << "preseed:" << preseed << std::endl;

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        unsigned int seed = id + preseed;
        for (int i = 0; i < N/num_threads; i++)
        {
            double x, y;

            x = rand_r(&seed) / (double)RAND_MAX * areaSize;
            y = rand_r(&seed) / (double)RAND_MAX * areaSize;

            if (isWithinCircle(x, y))
            {
                vec[id]++;
            }
        }
    }   

    for (int i = 0; i < num_threads; i++)
    {
        inside += vec[i];
    }

    outside = N - inside;

    final_time = omp_get_wtime() - init_time;

    std::cout << "In: " << inside << " Out: " << outside << std::endl;
    std::cout << "Pi: " << 4.0 * inside / N << std::endl;
    std::cout << "Time: " << final_time << " s" << std::endl;
}