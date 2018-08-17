__kernel void sparse_mxv(uint num_rows, __global uint* row_col_offset, __global uint* col_edges_map,
                         __global double* edges_val, __local double* tmp_vals, __global double* x,
                         __global double* y, __global double* z, double alpha) {
    int thread_id = get_global_id(0);
    int local_id = get_local_id(0);
    int row = thread_id;

    int row_A_start = row_col_offset[row];
    int row_A_end = row_col_offset[row + 1];

    tmp_vals[local_id] = 0;
    for (int jj = row_A_start; jj < row_A_end; jj += 1) {
        tmp_vals[local_id] += edges_val[jj] * x[col_edges_map[jj]];
    }

    y[row] = alpha * tmp_vals[local_id] + z[row];
    return;
}

__kernel void dense_mxv(uint num_rows, __global double* A, __local double* tmp_vals,
                        __global double* x, __global double* y, __global double* z, double alpha) {
    int thread_id = get_global_id(0);
    int row = thread_id;

    tmp_vals[thread_id] = 0;
    int offset = thread_id * num_rows;
    int num_col = num_rows;
    for (int jj = 0; jj < num_col; jj += 1) {
        tmp_vals[thread_id] += A[offset + jj] * x[jj];
    }

    y[row] = alpha * tmp_vals[thread_id] + z[row];
    return;
}

__kernel void approximate_mxv(uint num_rows, __global double* A, __local double* tmp_vals,
                              __global double* x, __global double* y, __global int* if_active) {
    int thread_id = get_global_id(0);
    int row = thread_id;

    tmp_vals[thread_id] = 0;
    int offset = thread_id * num_rows;
    int num_col = num_rows;
    if (if_active[thread_id] != 0) {
        for (int jj = 0; jj < num_col; jj += 1) {
            tmp_vals[thread_id] += A[offset + jj] * x[jj];
        }
        y[row] = tmp_vals[thread_id];
    } else {
        y[row] = x[row];
    }
    return;
}