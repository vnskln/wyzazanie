#ifndef SIMULATED_ANNEALING_H
#define SIMULATED_ANNEALING_H

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

typedef std::function<double(const std::vector<double>&, uint32_t)> calc_function_t;

std::pair<std::vector<double>, double> perform_sequential_algorithm(
    const calc_function_t& calc_value,
    std::vector<double> starting_x_0,
    const uint32_t n,
    const int a,
    const int b
);

#endif // SIMULATED_ANNEALING_H
