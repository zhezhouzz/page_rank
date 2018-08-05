#pragma once

#include <sys/time.h>
#include <string>

#define STRING_(x) #x
#define GET_VALUE_(x) STRING_(x)
#define PRINT_MACRO(x) #x "=" STRING_(x)

typedef enum FpDebugLevel {
    FP_LEVEL_NONE = 0,
    FP_LEVEL_ERROR = 1,
    FP_LEVEL_WARNING = 2,
    FP_LEVEL_INFO = 3,
    FP_LEVEL_SPLIT_NORMAL = 4,
} FpDebugLevel;

static const char* const FP_DEBUG_LEVEL_NAME[] = {"[INFO] ", "[WARNING] ", "[ERROR] "};

void _FP_LOG(int level, const char* format, ...);

#define __T(first, second) #first #second

#ifndef CUR_DEBUG_LEVEL
#define CUR_DEBUG_LEVEL 0
#endif

#if CUR_DEBUG_LEVEL > 0
#define FP_LOG(lv, ...)             \
    do {                            \
        _FP_LOG((lv), __VA_ARGS__); \
    } while (0)
#define FP_LOG_D(lv, ...)                              \
    do {                                               \
        _FP_LOG((lv), "[%s::%d]", __FILE__, __LINE__); \
        _FP_LOG((lv), __VA_ARGS__);                    \
    } while (0)
#else
#define FP_LOG(lv, ...)
#define FP_LOG_D(lv, ...)
#endif

#define DEFAULT_DEBUG FP_LOG(FP_LEVEL_ERROR, "[%s:%s:%d]\n", __FILE__, __FUNCTION__, __LINE__)

class FPDebugTimer {
private:
    long SubTime(const struct timeval& start_, const struct timeval& end_);

public:
    FPDebugTimer(FpDebugLevel level_, const std::string& file_, int line_, int times_ = 0);
    void OnceStart();
    void OnceEnd(FpDebugLevel level_, const std::string& file_, int line_);
    ~FPDebugTimer();
    FpDebugLevel level;
    std::string file;
    int line;
    int times;
    long lasting_time;
    struct timeval s_time;
    struct timeval o_time;
    struct timeval e_time;
};
