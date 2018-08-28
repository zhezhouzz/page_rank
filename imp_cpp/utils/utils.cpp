#include "utils.h"

void print_vector_if_active(const std::vector<bool>& if_active) {
    FP_LOG(FP_LEVEL_INFO, "if_active: [");
    for(int i = 0 ; i < if_active.size(); i++) {
        if(if_active[i]) {
            FP_LOG(FP_LEVEL_INFO, "true, ");
        } else {
            FP_LOG(FP_LEVEL_INFO, "false, ");
        }
    }
    FP_LOG(FP_LEVEL_INFO, "]\n");
    return;
}