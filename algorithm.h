#ifndef SIMULATED_ANNEALING_H
#define SIMULATED_ANNEALING_H

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

// Function type for sequential calculation (full vector)
typedef std::function<double(const std::vector<double> &, uint32_t)>
    calc_function_t;

// Function type for parallel calculation (local chunk only)
// Parameters: local_x, local_n, global_start_index
typedef std::function<double(const std::vector<double> &, uint32_t, uint32_t)>
    calc_function_partial_t;

// Sequential version of Simulated Annealing
std::pair<std::vector<double>, double>
perform_sequential_algorithm(const calc_function_t &calc_value,
                             std::vector<double> starting_x_0, const uint32_t n,
                             const int a, const int b);

// Parallel version of Simulated Annealing (MPI)
// Returns full result vector on rank 0, empty on other ranks
std::pair<std::vector<double>, double> perform_parallel_algorithm(
    const calc_function_partial_t &calc_value_partial,
    std::vector<double> starting_x_0, const uint32_t n, const int a,
    const int b,
    const uint32_t block_alignment = 1 // Set to 4 for Woods/Powell functions
);

#endif // SIMULATED_ANNEALING_H
