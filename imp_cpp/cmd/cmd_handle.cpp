#include "cmd_handle.h"
#include "vendor/include/cxxopts.hpp"

static KernelType kernel_type_handle(const std::string& kernel_type_str) {
    if (kernel_type_str == "opencl") {
        return KernelType::opencl;
    } else if (kernel_type_str == "taco") {
        return KernelType::taco;
    } else {
        std::cout << "bad kernel_type" << std::endl;
        exit(0);
    }
}

static AlgoType algo_type_handle(const std::string& algo_type_str) {
    if (algo_type_str == "sparse") {
        return AlgoType::sparse;
    } else if (algo_type_str == "dense") {
        return AlgoType::dense;
    } else if (algo_type_str == "approximate") {
        return AlgoType::approximate;
    } else {
        std::cout << "bad algo_type" << std::endl;
        exit(0);
    }
}

static std::string data_set_handle(const std::string& data_set_path_str) {
    if (data_set_path_str != "") {
        return data_set_path_str;
    } else {
        std::cout << "bad data_set" << std::endl;
        exit(0);
    }
}

static double epsilon_handle(double epsilon_double) {
    if (epsilon_double != 0) {
        return epsilon_double;
    } else {
        std::cout << "bad epsilon" << std::endl;
        exit(0);
    }
}

static int inactive_tolerance_handle(int inactive_tolerance_path_int) {
    if (inactive_tolerance_path_int != 0) {
        return inactive_tolerance_path_int;
    } else {
        std::cout << "bad inactive_tolerance" << std::endl;
        exit(0);
    }
}

static double terminate_active_rate_handle(double terminate_active_rate_double) {
    if (terminate_active_rate_double != 0) {
        return terminate_active_rate_double;
    } else {
        std::cout << "bad terminate_active_rate" << std::endl;
        exit(0);
    }
}

CmdOpt cmd_handle(int argc, char* argv[]) {
    CmdOpt ret_opt;
    try {
        cxxopts::Options options(argv[0], " - page_rank command line options");
        options.positional_help("[optional args]").show_positional_help();

        std::string kernel_type_str;
        std::string algorithm_type_str;
        std::string data_set_path_str;
        double epsilon_double = 0;
        int inactive_tolerance_int = 0;
        double terminate_active_rate_double = 0;

        options.add_options()("k,kernel", "kernel type",
                              cxxopts::value<std::string>(kernel_type_str))(
            "a,algorithm", "algorithm type", cxxopts::value<std::string>(algorithm_type_str))(
            "d,data", "data set path", cxxopts::value<std::string>(data_set_path_str))(
            "e,epsilon", "the accuracy of result", cxxopts::value<double>(epsilon_double))(
            "t,inactive_tolerance", "inactive tolerance",
            cxxopts::value<int>(inactive_tolerance_int))(
            "r,terminate_active_rate", "terminate active rate",
            cxxopts::value<double>(terminate_active_rate_double))("h,help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(0);
        }
        std::cout << "k = " << kernel_type_str << std::endl;
        ret_opt.kernel_type = kernel_type_handle(kernel_type_str);
        std::cout << "a = " << algorithm_type_str << std::endl;
        ret_opt.algo_type = algo_type_handle(algorithm_type_str);
        std::cout << "d = " << data_set_path_str << std::endl;
        ret_opt.data_set_path = data_set_handle(data_set_path_str);
        std::cout << "e = " << epsilon_double << std::endl;
        ret_opt.eps = epsilon_handle(epsilon_double);
        std::cout << "t = " << inactive_tolerance_int << std::endl;
        ret_opt.inactive_tolerance = inactive_tolerance_handle(inactive_tolerance_int);
        std::cout << "r = " << terminate_active_rate_double << std::endl;
        ret_opt.terminate_active_rate = terminate_active_rate_handle(terminate_active_rate_double);

    } catch (const cxxopts::OptionException& e) {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }
    return ret_opt;
}