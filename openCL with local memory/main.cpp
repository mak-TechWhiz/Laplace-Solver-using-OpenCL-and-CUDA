#include <opencl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// Laplace Solver Constants
const float TOP_V = 5.0f;
const float BOTTOM_V = -5.0f;
const float SIDE_V = 0.0f;
const float TOLERANCE = 1e-5f;
const int MAX_ITERATIONS = 10000;

struct BenchmarkResult {
    int grid_size;
    int workgroup_size;
    double execution_time;
    int iterations;
    double final_delta;
    bool converged;
};

void initializeGrid(float* grid, int n) {
    // Initialize all interior points to 0
    std::fill(grid, grid + n*n, 0.0f);
    
    // Set boundary conditions
    for (int j = 0; j < n; ++j) {
        grid[0*n + j] = TOP_V;           // Top boundary
        grid[(n-1)*n + j] = BOTTOM_V;    // Bottom boundary
    }
    
    for (int i = 0; i < n; ++i) {
        grid[i*n + 0] = SIDE_V;          // Left boundary
        grid[i*n + (n-1)] = SIDE_V;      // Right boundary
    }
}

void writeResultsToCSV(const std::vector<BenchmarkResult>& results, 
                      const std::string& filename) {
    std::ofstream csvFile(filename);
    csvFile << "Grid_Size,Workgroup_Size,Execution_Time_ms,Iterations,Final_Delta,Converged\n";
    for (const auto& res : results) {
        csvFile << res.grid_size << "," << res.workgroup_size << ","
                << std::fixed << std::setprecision(6) << res.execution_time << ","
                << res.iterations << "," << std::scientific << std::setprecision(6) 
                << res.final_delta << "," << (res.converged ? "true" : "false") << "\n";
    }
    csvFile.close();
}

double calculateMaxDelta(const std::vector<float>& current, 
                        const std::vector<float>& previous, 
                        int n) {
    double max_delta = 0.0;
    // Only check interior points for convergence
    for (int i = 1; i < n-1; i++) {
        for (int j = 1; j < n-1; j++) {
            int idx = i*n + j;
            double diff = std::abs(current[idx] - previous[idx]);
            max_delta = std::max(max_delta, diff);
        }
    }
    return max_delta;
}

int main() {
    std::vector<int> grid_sizes = {256, 512, 1024, 2048};
    std::vector<int> workgroup_sizes = {8, 16, 32};
    std::vector<BenchmarkResult> results;
    
    std::cout << "=== OpenCL Laplace Solver with Local Memory (Fixed) ===" << std::endl;
    
    for (int size : grid_sizes) {
        std::cout << "\nGrid size: " << size << "x" << size << std::endl;
        int total_size = size * size;
        
        Device device(select_device_with_most_flops(), "kernels.cl");
        Memory<float> grid(device, total_size, 1);
        Memory<float> next(device, total_size, 1);
        
        initializeGrid(grid.data(), size);
        initializeGrid(next.data(), size);
        grid.write_to_device();
        next.write_to_device();
        
        for (int ws : workgroup_sizes) {
            if (size % ws != 0) {
                std::cout << "Skipping workgroup " << ws << "x" << ws 
                         << " (grid size not divisible)" << std::endl;
                continue;
            }
            
            // Tiled kernel with local memory
            localMemory<float> local_mem((ws+2) * (ws+2));
            
            // Create kernels for ping-pong buffering
            Kernel kernel_A(device, size, size, true, ws, ws,
                "tiled_laplace_step", next, grid, size, local_mem);
            Kernel kernel_B(device, size, size, true, ws, ws,
                "tiled_laplace_step", grid, next, size, local_mem);
            
            std::vector<double> times;
            int iterations = 0;
            double max_delta = 0.0;
            bool converged = false;
            
            // Run benchmark 3 times and take average of last 2 (skip first for warmup)
            for (int run = 0; run < 3; run++) {
                // Reset to initial state
                initializeGrid(grid.data(), size);
                initializeGrid(next.data(), size);
                grid.write_to_device();
                next.write_to_device();
                
                auto start = std::chrono::high_resolution_clock::now();
                converged = false;
                
                // Create host buffers for convergence checking
                std::vector<float> current_host(total_size);
                std::vector<float> prev_host(total_size);
                
                // Initialize prev_host with initial grid state
                std::copy(grid.data(), grid.data() + total_size, prev_host.begin());
                
                for (iterations = 0; iterations < MAX_ITERATIONS; iterations++) {
                    // Run kernel (ping-pong between buffers)
                    if (iterations % 2 == 0) {
                        kernel_A.run();  // grid -> next
                    } else {
                        kernel_B.run();  // next -> grid
                    }
                    
                    // Check convergence every 50 iterations to reduce overhead
                    if (iterations % 50 == 49) {
                        // Read current solution to host
                        if (iterations % 2 == 0) {
                            next.read_from_device();
                            std::copy(next.data(), next.data() + total_size, current_host.begin());
                        } else {
                            grid.read_from_device();
                            std::copy(grid.data(), grid.data() + total_size, current_host.begin());
                        }
                        
                        // Calculate delta from previous solution
                        max_delta = calculateMaxDelta(current_host, prev_host, size);
                        
                        // Update previous solution for next check
                        prev_host = current_host;
                        
                        // Check for convergence
                        if (max_delta < TOLERANCE) {
                            converged = true;
                            iterations++; // Account for the iteration we just completed
                            break;
                        }
                    }
                }
                
                // Final convergence check if we didn't converge during the loop
                if (!converged && iterations >= MAX_ITERATIONS) {
                    // Read final solution
                    if ((iterations-1) % 2 == 0) {
                        next.read_from_device();
                        std::copy(next.data(), next.data() + total_size, current_host.begin());
                    } else {
                        grid.read_from_device();
                        std::copy(grid.data(), grid.data() + total_size, current_host.begin());
                    }
                    
                    // Calculate final delta
                    max_delta = calculateMaxDelta(current_host, prev_host, size);
                    
                    // Check if we converged at the last iteration
                    if (max_delta < TOLERANCE) {
                        converged = true;
                    }
                }
                
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duration = end - start;
                times.push_back(duration.count());
                
                // For consistency, use the same iteration count and convergence for all runs
                if (run == 0) {
                    // Store results from first run for consistency
                }
            }
            
            // Average the last two runs (skip first for warmup)
            double avg_time = (times[1] + times[2]) / 2.0;
            
            std::cout << "Workgroup " << std::setw(2) << ws << "x" << ws << ": "
                      << std::fixed << std::setprecision(2) << std::setw(8) << avg_time << " ms, "
                      << std::setw(4) << iterations << " iter, Δ=" 
                      << std::scientific << std::setprecision(3) << max_delta
                      << (converged ? " ✓" : " ✗") << std::endl;
            
            results.push_back({
                size, ws, avg_time, iterations, max_delta, converged
            });
        }
    }
    
    std::cout << "\nWriting results to opencl_tiled_benchmarks.csv..." << std::endl;
    writeResultsToCSV(results, "opencl_tiled_benchmarks.csv");
    
    std::cout << "\nBenchmark Summary:" << std::endl;
    std::cout << "=================" << std::endl;
    for (const auto& result : results) {
        std::cout << "Grid " << result.grid_size << ", WG " << result.workgroup_size 
                  << ": " << std::fixed << std::setprecision(2) << result.execution_time 
                  << "ms, " << result.iterations << " iter" 
                  << (result.converged ? " ✓" : " ✗") << std::endl;
    }
    
    return 0;
}