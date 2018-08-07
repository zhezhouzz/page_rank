#pragma once

#include <string>
#include <sstream>

template<typename T>
std::string fp_to_string(const T input)
{
    std::stringstream fp_ss;
    fp_ss.clear();
    fp_ss << input;
    return fp_ss.str();
}
