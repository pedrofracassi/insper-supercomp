#include <iomanip>
#include <iostream>
#include <omp.h>
static long num_steps = 1024l * 1024 * 1024 * 2;

#define MIN_BLK 1024 * 1024 * 256

double sum = 0;

double pi_r(long Nstart, long Nfinish, double step)
{
    long i, iblk;
    if (Nfinish - Nstart < MIN_BLK * MULTIPLIER)
    {
        double local_sum = 0.0;
        for (i = Nstart; i < Nfinish; i++)
        {
            double x = (i + 0.5) * step;
            local_sum += 4.0 / (1.0 + x * x);
        }
        return local_sum;
    }
    else
    {
        iblk = Nfinish - Nstart;
        double a, b;

        #pragma omp task shared(a)
        a = pi_r(Nstart, Nfinish - iblk / 2, step);
        #pragma omp task shared(b)
        b = pi_r(Nfinish - iblk / 2, Nfinish, step);
        #pragma omp taskwait

        return a + b;
    }
}

int main()
{
    long i;
    double step, pi;
    double init_time, final_time;
    sum = 0.0;
    step = 1.0 / (double)num_steps;
    init_time = omp_get_wtime();
    #pragma omp parallel
    {
      sum = pi_r(0, num_steps, step);
    }
    pi = step * sum;
    final_time = omp_get_wtime() - init_time;

    // std::cout << "" << num_steps << "," << std::setprecision(15) << pi << "," << final_time << "\n";
    std::cout << MULTIPLIER << ",task," << final_time << "\n";
}