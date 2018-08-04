// On Linux and MacOS, you can compile and run this program like so:
//   g++ -std=c++11 -O3 -DNDEBUG -DTACO -I ../../include -L../../build/lib -ltaco spmv.cpp -o spmv
//   LD_LIBRARY_PATH=../../build/lib ./spmv

#include <taco.h>
#include <iostream>
#include <random>

using namespace taco;

int main(int argc, char* argv[]) {
    std::default_random_engine gen(0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    Format csr({Sparse, Sparse});
    Format dv({Dense});

    Tensor<double> A = read("../data/page_map.mtx", csr);
    std::cout << "LOAD FINISHED" << std::endl;
    std::cout << "A.getDimension(1) = " << A.getDimension(1) << std::endl;

    Tensor<double> x({A.getDimension(1)}, dv);
    int length = x.getDimension(0);
    for (int i = 0; i < length; ++i) {
        x.insert({i}, (double)(1.0f/length));
    }
    x.pack();

    Tensor<double> alpha(0.85);

    Tensor<double> z({A.getDimension(0)}, dv);
    for (int i = 0; i < z.getDimension(0); ++i) {
        z.insert({i}, (double)(0.15f/length));
    }
    z.pack();

    for(int t = 0; t < 23; t++) {
        Tensor<double> y({A.getDimension(0)}, dv);

        IndexVar i, j;
        y(i) = alpha() * (A(i, j) * x(j)) + z(i);

        y.compile();
        y.assemble();
        y.compute();

        x = y;
    }

    write("y.tns", x);
}