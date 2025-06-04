#include <opencl.hpp>
#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

// Laplace Solver Constants
const float TOP_V = 5.0f;
const float BOTTOM_V = -5.0f;
const float SIDE_V = 0.0f;
const float TOLERANCE = 1e-5f;
const int MAX_ITERATIONS = 10000;

// Timing structure
struct myClock {
    typedef std::chrono::high_resolution_clock clock;
    std::chrono::time_point<clock> a1, a2;
    void Start() { a1 = clock::now(); }
    void Stop()  { a2 = clock::now(); }
    double ElapsedTime() {
        duration<double, std::milli> time_ms = a2 - a1;
        return time_ms.count();
    }
};

// CSV Data Structure
struct BenchmarkResult {
    int grid_size;
    std::string implementation;
    int workgroup_size;
    double execution_time;
    int iterations;
    double final_delta;
    bool converged;
};

myClock Clock;

void initializeGrid(float* grid, int n) {
    // Initialize all points to 0
    std::fill(grid, grid + n*n, 0.0f);

    // Top boundary
    for (int j = 0; j < n; ++j) grid[j] = TOP_V;
    // Bottom boundary
    for (int j = 0; j < n; ++j) grid[(n-1)*n + j] = BOTTOM_V;
    // Left & right boundaries
    for (int i = 0; i < n; ++i) {
        grid[i*n] = SIDE_V;
        grid[i*n + (n-1)] = SIDE_V;
    }
}

// Revised CPU implementation without pointer swapping
void laplace_cpu(float* grid, int n, double* elapsed_time, int* iterations, double* final_delta, bool* converged) {
    vector<float> current(n*n);
    vector<float> next(n*n);
    
    // Initialize buffers
    std::copy(grid, grid + n*n, current.data());
    std::copy(grid, grid + n*n, next.data());
    
    myClock timer;
    timer.Start();
    
    double max_delta = 0.0;
    int iter = 0;
    *converged = false;
    
    for (iter = 0; iter < MAX_ITERATIONS; iter++) {
        max_delta = 0.0;
        
        // Update inner grid points
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                int idx = i*n + j;
                float new_val = (current[(i-1)*n+j] + current[(i+1)*n+j] +
                                current[i*n+(j-1)] + current[i*n+(j+1)]) * 0.25f;
                double diff = std::abs(new_val - current[idx]);
                if (diff > max_delta) max_delta = diff;
                next[idx] = new_val;
            }
        }
        
        // Copy boundaries to next (they remain fixed)
        for (int j = 0; j < n; ++j) next[j] = TOP_V;  // Top
        for (int j = 0; j < n; ++j) next[(n-1)*n + j] = BOTTOM_V;  // Bottom
        for (int i = 0; i < n; ++i) {  // Sides
            next[i*n] = SIDE_V;
            next[i*n + (n-1)] = SIDE_V;
        }
        
        // Swap vector contents (efficient pointer swap)
        std::swap(current, next);
        
        // Check convergence
        if (max_delta < TOLERANCE) {
            *converged = true;
            break;
        }
    }
    
    timer.Stop();
    
    // Copy final result back to output grid
    std::copy(current.begin(), current.end(), grid);
    
    *elapsed_time = timer.ElapsedTime();
    *iterations = iter + 1;
    *final_delta = max_delta;
}

// Revised OpenMP implementation without pointer swapping
void laplace_omp(float* grid, int n, double* elapsed_time, int* iterations, double* final_delta, bool* converged) {
    vector<float> current(n*n);
    vector<float> next(n*n);
    
    // Initialize buffers
    std::copy(grid, grid + n*n, current.data());
    std::copy(grid, grid + n*n, next.data());
    
    myClock timer;
    timer.Start();
    
    double max_delta = 0.0;
    int iter = 0;
    *converged = false;
    
    for (iter = 0; iter < MAX_ITERATIONS; iter++) {
        double local_max = 0.0;
        
        // Update inner grid points
        #pragma omp parallel for reduction(max:local_max)
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                int idx = i*n + j;
                float new_val = (current[(i-1)*n+j] + current[(i+1)*n+j] +
                                current[i*n+(j-1)] + current[i*n+(j+1)]) * 0.25f;
                double diff = std::abs(new_val - current[idx]);
                if (diff > local_max) local_max = diff;
                next[idx] = new_val;
            }
        }
        
        max_delta = local_max;
        
        // Copy boundaries to next (they remain fixed)
        #pragma omp parallel for
        for (int j = 0; j < n; ++j) next[j] = TOP_V;  // Top
        
        #pragma omp parallel for
        for (int j = 0; j < n; ++j) next[(n-1)*n + j] = BOTTOM_V;  // Bottom
        
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {  // Sides
            next[i*n] = SIDE_V;
            next[i*n + (n-1)] = SIDE_V;
        }
        
        // Swap vector contents (efficient pointer swap)
        std::swap(current, next);
        
        // Check convergence
        if (max_delta < TOLERANCE) {
            *converged = true;
            break;
        }
    }
    
    timer.Stop();
    
    // Copy final result back to output grid
    std::copy(current.begin(), current.end(), grid);
    
    *elapsed_time = timer.ElapsedTime();
    *iterations = iter + 1;
    *final_delta = max_delta;
}

void writeResultsToCSV(const vector<BenchmarkResult>& results, const string& filename) {
    std::ofstream csvFile(filename);
    
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    
    // Write CSV header
    csvFile << "Grid_Size,Implementation,Workgroup_Size,Execution_Time_ms,Iterations,Final_Delta,Converged\n";
    
    // Write data
    for (const auto& result : results) {
        csvFile << result.grid_size << ","
                << result.implementation << ","
                << result.workgroup_size << ","
                << std::fixed << std::setprecision(6) << result.execution_time << ","
                << result.iterations << ","
                << std::scientific << std::setprecision(6) << result.final_delta << ","
                << (result.converged ? "true" : "false") << "\n";
    }
    
    csvFile.close();
    cout << "Results saved to " << filename << endl;
}

int main() {
    vector<int> grid_sizes = {64, 128, 256, 512, 1024};
    vector<int> workgroup_sizes = {8, 16, 32};
    vector<BenchmarkResult> all_results;

    cout << "=== Laplace Equation Solver Performance Comparison ===" << endl;
    cout << "Boundary Conditions: Top=" << TOP_V << "V, Bottom=" << BOTTOM_V
         << "V, Sides=" << SIDE_V << "V" << endl;
    cout << "Tolerance: " << TOLERANCE << ", Max Iterations: " << MAX_ITERATIONS << endl << endl;

    for (int size : grid_sizes) {
        cout << "Grid size: " << size << "x" << size << endl;
        cout << "--------------------------------------------" << endl;

        int total_size = size * size;
        vector<float> cpu_grid(total_size);
        vector<float> omp_grid(total_size);
        
        // Initialize grids
        initializeGrid(cpu_grid.data(), size);
        initializeGrid(omp_grid.data(), size);

        // CPU Version
        double cpu_time;
        int cpu_iterations;
        double cpu_final_delta;
        bool cpu_converged;
        laplace_cpu(cpu_grid.data(), size, &cpu_time, &cpu_iterations, &cpu_final_delta, &cpu_converged);
        
        BenchmarkResult cpu_result;
        cpu_result.grid_size = size;
        cpu_result.implementation = "CPU";
        cpu_result.workgroup_size = 0;
        cpu_result.execution_time = cpu_time;
        cpu_result.iterations = cpu_iterations;
        cpu_result.final_delta = cpu_final_delta;
        cpu_result.converged = cpu_converged;
        all_results.push_back(cpu_result);
        
        cout << "CPU: " << std::fixed << std::setprecision(3) << cpu_time << " ms, "
             << cpu_iterations << " iterations, Final Δ=" 
             << std::scientific << std::setprecision(6) << cpu_final_delta;
        if (cpu_converged) cout << " (CONVERGED)";
        cout << endl;

        // OpenMP Version
        double omp_time;
        int omp_iterations;
        double omp_final_delta;
        bool omp_converged;
        laplace_omp(omp_grid.data(), size, &omp_time, &omp_iterations, &omp_final_delta, &omp_converged);
        
        BenchmarkResult omp_result;
        omp_result.grid_size = size;
        omp_result.implementation = "OpenMP";
        omp_result.workgroup_size = 0;
        omp_result.execution_time = omp_time;
        omp_result.iterations = omp_iterations;
        omp_result.final_delta = omp_final_delta;
        omp_result.converged = omp_converged;
        all_results.push_back(omp_result);
        
        cout << "OpenMP: " << std::fixed << std::setprecision(3) << omp_time << " ms, "
             << omp_iterations << " iterations, Final Δ=" 
             << std::scientific << std::setprecision(6) << omp_final_delta;
        if (omp_converged) cout << " (CONVERGED)";
        cout << endl;

        // OpenCL Setup
        Device device(select_device_with_most_flops(), "kernels.cl");
        Memory<float> cl_grid(device, total_size, 1);
        Memory<float> cl_next(device, total_size, 1);

        // Initialize OpenCL grid
        initializeGrid(cl_grid.data(), size);
        initializeGrid(cl_next.data(), size);
        cl_grid.write_to_device();
        cl_next.write_to_device();

        for (int ws : workgroup_sizes) {
            if (size % ws != 0) continue;

            // Create two kernels for ping-pong buffering
            Kernel kernel_even(
                device, size, size, true, ws, ws,
                "laplace_step", cl_next, cl_grid, size
            );
            Kernel kernel_odd(
                device, size, size, true, ws, ws,
                "laplace_step", cl_grid, cl_next, size
            );

            vector<double> times;
            int iterations = 0;
            double max_delta = 0.0;
            bool converged = false;

            for (int run = 0; run < 3; ++run) {
                // Reset to initial state
                initializeGrid(cl_grid.data(), size);
                initializeGrid(cl_next.data(), size);
                cl_grid.write_to_device();
                cl_next.write_to_device();

                Clock.Start();
                converged = false;
                for (iterations = 0; iterations < MAX_ITERATIONS; ++iterations) {
                    // Alternate between kernels
                    if (iterations % 2 == 0) {
                        kernel_even.run();
                    } else {
                        kernel_odd.run();
                    }

                    // Check convergence every 100 iterations
                    if (iterations % 100 == 0) {
                        // Read both buffers
                        cl_grid.read_from_device();
                        cl_next.read_from_device();
                        
                        // Calculate maximum delta
                        max_delta = 0.0;
                        for (int i = 0; i < size; ++i) {
                            for (int j = 0; j < size; ++j) {
                                int idx = i*size + j;
                                double diff = std::abs(
                                    cl_grid.data()[idx] - cl_next.data()[idx]
                                );
                                if (diff > max_delta) max_delta = diff;
                            }
                        }
                        
                        if (max_delta < TOLERANCE) {
                            converged = true;
                            break;
                        }
                    }
                }
                
                // Final convergence check if not already converged
                if (!converged) {
                    cl_grid.read_from_device();
                    cl_next.read_from_device();
                    max_delta = 0.0;
                    for (int i = 0; i < size; ++i) {
                        for (int j = 0; j < size; ++j) {
                            int idx = i*size + j;
                            double diff = std::abs(
                                cl_grid.data()[idx] - cl_next.data()[idx]
                            );
                            if (diff > max_delta) max_delta = diff;
                        }
                    }
                }
                Clock.Stop();
                times.push_back(Clock.ElapsedTime());
            }

            double avg_time = (times[1] + times[2]) / 2.0;
            cout << "OpenCL laplace_step (" << ws << "x" << ws << "): "
                 << std::fixed << std::setprecision(3) << avg_time
                 << " ms, " << iterations << " iterations, Final Δ=" 
                 << std::scientific << std::setprecision(6) << max_delta;
            if (converged) cout << " (CONVERGED)";
            cout << endl;

            BenchmarkResult ocl_result;
            ocl_result.grid_size = size;
            ocl_result.implementation = "OpenCL";
            ocl_result.workgroup_size = ws;
            ocl_result.execution_time = avg_time;
            ocl_result.iterations = iterations;
            ocl_result.final_delta = max_delta;
            ocl_result.converged = converged;
            all_results.push_back(ocl_result);
        }
        cout << endl;
    }

    // Write results to CSV
    writeResultsToCSV(all_results, "laplace_benchmarks.csv");

    return 0;
}