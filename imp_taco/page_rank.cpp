// On Linux and MacOS, you can compile and run this program like so:
//   g++ -std=c++11 -O3 -DNDEBUG -DTACO -I ../../include -L../../build/lib -ltaco spmv.cpp -o spmv
//   LD_LIBRARY_PATH=../../build/lib ./spmv

#include <taco.h>
#include <iostream>
#include <random>

using namespace taco;

#define ERROR_HANDLE_                                                                  \
    do {                                                                               \
        if (ret_code != 0) {                                                           \
            std::cout << "[line " << __LINE__ << "] error: " << ret_code << std::endl; \
        }                                                                              \
    } while (0)

int assemble(taco_tensor_t *y, taco_tensor_t *alpha, taco_tensor_t *A, taco_tensor_t *x, taco_tensor_t *z) {
  int y1_dimension = (int)(y->dimensions[y->mode_ordering[0]]);
  double* __restrict y_vals = (double*)(y->vals);
  int y_vals_size = y->vals_size;
  std::cout << "y1_dimension = " << y1_dimension << std::endl;
  std::cout << "y_vals_size = " << y_vals_size << std::endl;
  int* __restrict A1_pos = (int*)(A->indices[0][0]);
  int* __restrict A1_coord = (int*)(A->indices[0][1]);
  int* __restrict A2_pos = (int*)(A->indices[1][0]);
  int* __restrict A2_coord = (int*)(A->indices[1][1]);
  int z1_dimension = (int)(z->dimensions[z->mode_ordering[0]]);
  int32_t pA1 = A1_pos[0];
  int32_t A1_end = A1_pos[1];
  int32_t iz = 0;
  int32_t z1_end = z1_dimension;
  while (pA1 < A1_end) {
    int32_t iA = A1_coord[pA1];
    int32_t pz1 = iz;
    int32_t py1 = iz;
    if (iA == iz) {
      for (int32_t pA2 = A2_pos[pA1]; pA2 < A2_pos[(pA1 + 1)]; pA2++) {
        int32_t jA = A2_coord[pA2];
      }
    }
    else {
    }
    pA1 += (int32_t)(iA == iz);
    iz++;
  }
  while (iz < z1_end) {
    int32_t pz1 = iz;
    int32_t py1 = iz;
    iz++;
  }

  y_vals = (double*)malloc(sizeof(double) * y1_dimension);
  y_vals_size = y1_dimension;
  y->vals = (uint8_t*)y_vals;
  y->vals_size = y_vals_size;
  return 0;
}

int compute(taco_tensor_t *y, taco_tensor_t *alpha, taco_tensor_t *A, taco_tensor_t *x, taco_tensor_t *z) {
  int y1_dimension = (int)(y->dimensions[y->mode_ordering[0]]);
  double* __restrict y_vals = (double*)(y->vals);
  double* __restrict alpha_vals = (double*)(alpha->vals);
  int* __restrict A1_pos = (int*)(A->indices[0][0]);
  int* __restrict A1_coord = (int*)(A->indices[0][1]);
  int* __restrict A2_pos = (int*)(A->indices[1][0]);
  int* __restrict A2_coord = (int*)(A->indices[1][1]);
  double* __restrict A_vals = (double*)(A->vals);
  double* __restrict x_vals = (double*)(x->vals);
  int z1_dimension = (int)(z->dimensions[z->mode_ordering[0]]);
  double* __restrict z_vals = (double*)(z->vals);
  for (int32_t py = 0; py < y1_dimension; py++) {
    y_vals[py] = 0;
  }
  int32_t pA1 = A1_pos[0];
  int32_t A1_end = A1_pos[1];
  int32_t iz = 0;
  int32_t z1_end = z1_dimension;
  while (pA1 < A1_end) {
    int32_t iA = A1_coord[pA1];
    int32_t pz1 = iz;
    int32_t py1 = iz;
    if (iA == iz) {
      double tj = 0;
      for (int32_t pA2 = A2_pos[pA1]; pA2 < A2_pos[(pA1 + 1)]; pA2++) {
        int32_t jA = A2_coord[pA2];
        tj += A_vals[pA2] * x_vals[jA];
      }
      y_vals[py1] = alpha_vals[0] * tj + z_vals[pz1];
    }
    else {
      y_vals[py1] = z_vals[pz1];
    }
    pA1 += (int32_t)(iA == iz);
    iz++;
  }
  while (iz < z1_end) {
    int32_t pz1 = iz;
    int32_t py1 = iz;
    y_vals[py1] = z_vals[pz1];
    iz++;
  }
  return 0;
}

int loop_compute(taco_tensor_t *x, taco_tensor_t *y) {
  double* __restrict x_vals = (double*)(x->vals);
  int y1_dimension = (int)(y->dimensions[y->mode_ordering[0]]);
  double* __restrict y_vals = (double*)(y->vals);
  #pragma omp parallel for
  for (int32_t iy = 0; iy < y1_dimension; iy++) {
    x_vals[iy] = y_vals[iy];
  }
  return 0;
}

int main(int argc, char* argv[]) {
    std::default_random_engine gen(0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    Format csr({Sparse, Sparse});
    Format dv({Dense});

    Tensor<double> A = read("../data/page_map.mtx", csr);
    std::cout << "LOAD FINISHED" << std::endl;
    std::cout << "A.getDimension(1) = " << A.getDimension(1) << std::endl;

    Tensor<double> x({A.getDimension(1)}, dv);
    int length = x.getDimension(0);
    for (int i = 0; i < length; ++i) {
        x.insert({i}, (double)(1.0f / length));
    }
    x.pack();

    Tensor<double> alpha(0.85);

    Tensor<double> z({A.getDimension(0)}, dv);
    for (int i = 0; i < z.getDimension(0); ++i) {
        z.insert({i}, (double)(0.15f / length));
    }
    z.pack();

    taco_tensor_t* c_tensor_alpha = alpha.getTacoTensorT();
    taco_tensor_t* c_tensor_A = A.getTacoTensorT();
    taco_tensor_t* c_tensor_x = x.getTacoTensorT();
    taco_tensor_t* c_tensor_z = z.getTacoTensorT();

    Tensor<double> y({A.getDimension(0)}, dv);
    taco_tensor_t* c_tensor_y = y.getTacoTensorT();
    int ret_code = 0;
    std::cout << "assemble" << std::endl;
    ret_code = assemble(c_tensor_y, c_tensor_alpha, c_tensor_A, c_tensor_x, c_tensor_z);
    ERROR_HANDLE_;
    for (int t = 0; t < 23; t++) {
        std::cout << "compute" << std::endl;
        ret_code = compute(c_tensor_y, c_tensor_alpha, c_tensor_A, c_tensor_x, c_tensor_z);
        ERROR_HANDLE_;
        std::cout << "loop" << std::endl;
        ret_code = loop_compute(c_tensor_x, c_tensor_y);
        ERROR_HANDLE_;
    }
    write("x.tns", x);
}