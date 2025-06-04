__kernel void tiled_laplace_step(__global float* output, 
                                 __global const float* input, 
                                 int n,
                                 __local float* tile) {
    int li = get_local_id(0);
    int lj = get_local_id(1);
    int group_i = get_group_id(0);
    int group_j = get_group_id(1);
    int ls = get_local_size(0);  // Block size (assumed square)
    int tile_width = ls + 2;     // Tile width including halo
    
    // Load main tile data
    int global_i = group_i * ls + li;
    int global_j = group_j * ls + lj;
    
    float value;
    if (global_i < 0) {
        value = 5.0f;  // TOP_V
    } else if (global_i >= n) {
        value = -5.0f; // BOTTOM_V
    } else if (global_j < 0) {
        value = 0.0f;  // SIDE_V
    } else if (global_j >= n) {
        value = 0.0f;  // SIDE_V
    } else if (global_i < n && global_j < n) {
        value = input[global_i * n + global_j];
    } else {
        value = 0.0f;  // Default fallback
    }
    tile[(li+1) * tile_width + (lj+1)] = value;
    
    // Load halo regions with proper boundary handling
    if (li == 0) {
        // Top halo
        int halo_i = group_i * ls - 1;
        int halo_j = group_j * ls + lj;
        if (halo_i < 0) {
            value = 5.0f;  // TOP_V
        } else if (halo_j < 0 || halo_j >= n) {
            value = 0.0f;  // SIDE_V
        } else {
            value = input[halo_i * n + halo_j];
        }
        tile[0 * tile_width + (lj+1)] = value;
    }
    
    if (li == ls-1) {
        // Bottom halo
        int halo_i = group_i * ls + ls;
        int halo_j = group_j * ls + lj;
        if (halo_i >= n) {
            value = -5.0f; // BOTTOM_V
        } else if (halo_j < 0 || halo_j >= n) {
            value = 0.0f;  // SIDE_V
        } else {
            value = input[halo_i * n + halo_j];
        }
        tile[(ls+1) * tile_width + (lj+1)] = value;
    }
    
    if (lj == 0) {
        // Left halo
        int halo_i = group_i * ls + li;
        int halo_j = group_j * ls - 1;
        if (halo_j < 0) {
            value = 0.0f;  // SIDE_V
        } else if (halo_i < 0) {
            value = 5.0f;  // TOP_V
        } else if (halo_i >= n) {
            value = -5.0f; // BOTTOM_V
        } else {
            value = input[halo_i * n + halo_j];
        }
        tile[(li+1) * tile_width + 0] = value;
    }
    
    if (lj == ls-1) {
        // Right halo
        int halo_i = group_i * ls + li;
        int halo_j = group_j * ls + ls;
        if (halo_j >= n) {
            value = 0.0f;  // SIDE_V
        } else if (halo_i < 0) {
            value = 5.0f;  // TOP_V
        } else if (halo_i >= n) {
            value = -5.0f; // BOTTOM_V
        } else {
            value = input[halo_i * n + halo_j];
        }
        tile[(li+1) * tile_width + (ls+1)] = value;
    }
    
    // Load corner halos (critical for correctness)
    if (li == 0 && lj == 0) {
        // Top-left corner
        int corner_i = group_i * ls - 1;
        int corner_j = group_j * ls - 1;
        if (corner_i < 0 && corner_j < 0) {
            value = 5.0f;  // TOP_V wins at corners
        } else if (corner_i < 0) {
            value = 5.0f;  // TOP_V
        } else if (corner_j < 0) {
            value = 0.0f;  // SIDE_V
        } else {
            value = input[corner_i * n + corner_j];
        }
        tile[0 * tile_width + 0] = value;
    }
    
    if (li == 0 && lj == ls-1) {
        // Top-right corner
        int corner_i = group_i * ls - 1;
        int corner_j = group_j * ls + ls;
        if (corner_i < 0 && corner_j >= n) {
            value = 5.0f;  // TOP_V wins at corners
        } else if (corner_i < 0) {
            value = 5.0f;  // TOP_V
        } else if (corner_j >= n) {
            value = 0.0f;  // SIDE_V
        } else {
            value = input[corner_i * n + corner_j];
        }
        tile[0 * tile_width + (ls+1)] = value;
    }
    
    if (li == ls-1 && lj == 0) {
        // Bottom-left corner
        int corner_i = group_i * ls + ls;
        int corner_j = group_j * ls - 1;
        if (corner_i >= n && corner_j < 0) {
            value = -5.0f; // BOTTOM_V wins at corners
        } else if (corner_i >= n) {
            value = -5.0f; // BOTTOM_V
        } else if (corner_j < 0) {
            value = 0.0f;  // SIDE_V
        } else {
            value = input[corner_i * n + corner_j];
        }
        tile[(ls+1) * tile_width + 0] = value;
    }
    
    if (li == ls-1 && lj == ls-1) {
        // Bottom-right corner
        int corner_i = group_i * ls + ls;
        int corner_j = group_j * ls + ls;
        if (corner_i >= n && corner_j >= n) {
            value = -5.0f; // BOTTOM_V wins at corners
        } else if (corner_i >= n) {
            value = -5.0f; // BOTTOM_V
        } else if (corner_j >= n) {
            value = 0.0f;  // SIDE_V
        } else {
            value = input[corner_i * n + corner_j];
        }
        tile[(ls+1) * tile_width + (ls+1)] = value;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Compute updated value for interior points only
    int i = group_i * ls + li;
    int j = group_j * ls + lj;
    
    // Only update interior points, never boundary points
    if (i > 0 && i < n-1 && j > 0 && j < n-1) {
        // Access neighbors from local tile
        float top = tile[li * tile_width + (lj+1)];
        float bottom = tile[(li+2) * tile_width + (lj+1)];
        float left = tile[(li+1) * tile_width + lj];
        float right = tile[(li+1) * tile_width + (lj+2)];
        
        output[i * n + j] = (top + bottom + left + right) * 0.25f;
    } else if (i < n && j < n) {
        // Copy boundary values unchanged
        output[i * n + j] = input[i * n + j];
    }
}

__kernel void copy_grid(__global float* dst, __global const float* src, int n) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < n && j < n) {
        int idx = i * n + j;
        dst[idx] = src[idx];
    }
}