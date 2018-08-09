#pragma once

#include <taco.h>

int prepare();

int upload(taco_tensor_t* c_tensor_A, taco_tensor_t* c_tensor_alpha, taco_tensor_t* c_tensor_x,
           taco_tensor_t* c_tensor_y, taco_tensor_t* c_tensor_z);

int run(bool flag_x2y);

int download(bool flag_x2y, taco_tensor_t** c_tensor_x, taco_tensor_t** c_tensor_y);

int finish();