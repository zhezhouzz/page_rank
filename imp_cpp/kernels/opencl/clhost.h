#include <taco.h>

int prepare(taco_tensor_t* c_tensor_A, taco_tensor_t* c_tensor_alpha, taco_tensor_t* c_tensor_x,
            taco_tensor_t* c_tensor_y, taco_tensor_t* c_tensor_z);
int run();
int finish();