// Generated by the Tensor Algebra Compiler (tensor-compiler.org)
for (int32_t iA = 0; iA < A1_dimension; iA++) {
  double tj = 0;
  for (int32_t jA = 0; jA < A2_dimension; jA++) {
    int32_t pA2 = iA * A2_dimension + jA;
    tj += A_vals[pA2] * x_vals[jA];
  }
  y_vals[iA] = tj;
}
