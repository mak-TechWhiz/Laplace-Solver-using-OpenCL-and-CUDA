__kernel void laplace_step(__global float* output, 
                          __global const float* input, 
                          int n) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    // Only update interior points (boundaries are fixed)
    if (i > 0 && i < n-1 && j > 0 && j < n-1) {
        int idx = i*n + j;
        output[idx] = (input[(i-1)*n + j] + 
                      input[(i+1)*n + j] +
                      input[i*n + (j-1)] + 
                      input[i*n + (j+1)]) * 0.25f;
    }
}