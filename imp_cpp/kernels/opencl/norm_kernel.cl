void norm(uint num_rows, __global double* x, __global double* y, __local double* tmp_vals) {
    int thread_id = get_global_id(0);
    tmp_vals[thread_id] = (x[thread_id] - y[thread_id])*(x[thread_id] - y[thread_id]);
    if(thread_id == 0) {
        double ret = 0;
        for(int i = 0; i < num_rows; i++) {
            ret += tmp_vals[i];
        }
        tmp_vals[0] = ret;
    }
    return;
}