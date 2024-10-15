#include <iostream>
#include <omp.h>

double areaSize = 1.0;

bool isWithinCircle(double x, double y)
{
    return x * x + y * y <= areaSize * areaSize;
}

int main()
{
    int N = 1000000;

    int inside = 0;
    int outside = 0;

    double init_time, final_time;
    init_time = omp_get_wtime();
    #pragma omp parallel for reduction(+:inside, outside)
    for (int i = 0; i < N; i++)
    {
        double x, y;
        x = rand() / (double)RAND_MAX * areaSize;
        y = rand() / (double)RAND_MAX * areaSize;

        if (isWithinCircle(x, y))
        {
            inside++;
        }
        else
        {
            outside++;
        }
    }
    final_time = omp_get_wtime() - init_time;

    std::cout << "In: " << inside << " Out: " << outside << std::endl;
    std::cout << "Pi: " << 4.0 * inside / N << std::endl;
    std::cout << "Time: " << final_time << " s" << std::endl;
}