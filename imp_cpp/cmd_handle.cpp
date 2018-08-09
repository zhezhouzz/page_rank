#include "cmd_handle.h"
#include "vendor/include/cxxopts.hpp"

KernelType kernel_type_handle(std::string& kernel_type_str) {
    if (kernel_type_str == "opencl") {
        return KernelType::opencl;
    } else if (kernel_type_str == "taco") {
        return KernelType::taco;
    } else {
        std::cout << "bad kernel_type" << std::endl;
        exit(0);
    }
}

CmdOpt cmd_handle(int argc, char* argv[]) {
    CmdOpt ret_opt;
    try {
        cxxopts::Options options(argv[0], " - page_rank command line options");
        options.positional_help("[optional args]").show_positional_help();

        std::string kernel_type_str;

        options.add_options()("k,kernel", "kernel type",
                              cxxopts::value<std::string>(kernel_type_str));

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << "help" << std::endl;
            exit(0);
        }
        std::cout << "k = " << kernel_type_str << std::endl;
        ret_opt.kernel_type = kernel_type_handle(kernel_type_str);

    } catch (const cxxopts::OptionException& e) {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }
    return ret_opt;
}