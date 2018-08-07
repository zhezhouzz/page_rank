#include "utils_debug.h"
#include <stdio.h>
#include <stdarg.h>
#include <unistd.h>
#include <string>

static bool check_level(int level) {
    if (CUR_DEBUG_LEVEL < FP_LEVEL_SPLIT_NORMAL) {
        if (level > CUR_DEBUG_LEVEL) {
            return false;
        }
    } else {
        if (level != CUR_DEBUG_LEVEL) {
            return false;
        }
    }
    return true;
}

void _FP_LOG(int level, const char* format, ...) {
    if (check_level(level) == false) {
        return;
    }
    va_list argptr;
    va_start(argptr, format);
    va_list nil;
    vprintf(format, argptr);
    va_end(argptr);
    return;
}

long FPDebugTimer::SubTime(const struct timeval& start_, const struct timeval& end_) {
    return static_cast<long>((end_.tv_sec - start_.tv_sec) * 1000000 +
                             (end_.tv_usec - start_.tv_usec));
}

FPDebugTimer::FPDebugTimer(FpDebugLevel level_, const std::string& file_, int line_, int times_) {
    level = level_;
    file = file_;
    line = line_;
    times = times_;
    lasting_time = 0;
    gettimeofday(&s_time, NULL);
    return;
}

FPDebugTimer::~FPDebugTimer() {
    gettimeofday(&e_time, NULL);
    FP_LOG(level, "[%s:%d]time: %d us\n", file.c_str(), line, SubTime(s_time, e_time));
    return;
}

void FPDebugTimer::OnceStart() {
    gettimeofday(&o_time, NULL);
    return;
}

void FPDebugTimer::OnceEnd(FpDebugLevel level_, const std::string& file_, int line_) {
    if (times < 0) {
        times++;
        return;
    }
    times++;
    struct timeval o_new_time;
    gettimeofday(&o_new_time, NULL);
    float once_time = SubTime(o_time, o_new_time);
    FP_LOG(level_, "[%s:%d]once[%d] time: %f us\n", file_.c_str(), line_, times, once_time);
    lasting_time = lasting_time + once_time;
    float avg_time = static_cast<float>(lasting_time) / times;
    FP_LOG(level_, "[%s:%d]avg time in [%d]: %f us\n", file_.c_str(), line_, times, avg_time);
    return;
}