#include <taco.h>

int taco_upload(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
                taco_tensor_t* z);
int taco_page_rank(bool flag_x2y);
int taco_download(bool flag_x2y, taco_tensor_t** x, taco_tensor_t** y);
int taco_swap_vector(taco_tensor_t* x, taco_tensor_t* y);
double taco_vetor_norm(taco_tensor_t* x, taco_tensor_t* y);