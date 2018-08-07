__kernel void page_rank(uint num_rows, __global uint* row_col_offset, __global uint* col_edges_map,
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