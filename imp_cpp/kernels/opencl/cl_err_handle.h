#pragma once

#define CL_ERR_HANDLE                                                                      \
    do {                                                                                   \
        if (ret_code != CL_SUCCESS) {                                                      \
            FP_LOG(FP_LEVEL_ERROR, "[%s:%d] cl err = %d\n", __FILE__, __LINE__, ret_code); \
        }                                                                                  \
    } while (0)