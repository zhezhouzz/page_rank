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

CmdOpt cmd_handle(int argc, char* argv[]) {
    CmdOpt ret_opt;
    try {
        cxxopts::Options options(argv[0], " - page_rank command line options");
        options.positional_help("[optional args]").show_positional_help();

        std::string kernel_type_str;
        std::string algorithm_type_str;
        std::string data_set_path_str;

        options.add_options()("k,kernel", "kernel type",
                              cxxopts::value<std::string>(kernel_type_str))(
            "a,algorithm", "algorithm type", cxxopts::value<std::string>(algorithm_type_str))(
            "d,data", "data set path", cxxopts::value<std::string>(data_set_path_str));

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << "help" << std::endl;
            exit(0);
        }
        std::cout << "k = " << kernel_type_str << std::endl;
        ret_opt.kernel_type = kernel_type_handle(kernel_type_str);
        std::cout << "a = " << algorithm_type_str << std::endl;
        ret_opt.algo_type = algo_type_handle(algorithm_type_str);
        std::cout << "d = " << data_set_path_str << std::endl;
        ret_opt.data_set_path = data_set_handle(data_set_path_str);

    } catch (const cxxopts::OptionException& e) {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }
    return ret_opt;
}