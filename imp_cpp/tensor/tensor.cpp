#include "tensor.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>

int Tensor::load_sparce_mtx(const std::string &mtx_path) {
    DEFAULT_DEBUG;
    std::ifstream read_file(mtx_path);
    std::string line_str;
    std::stringstream ss;
    if (read_file.is_open()) {
        int row_length = 0;
        int col_length = 0;
        int element_num = 0;
        while (std::getline(read_file, line_str)) {
            if (line_str == "" || line_str[0] == '%') {
                continue;
            }
            ss.clear();
            ss << line_str;
            ss >> row_length >> col_length >> element_num;
            break;
        }
        dimensions.push_back(row_length);
        dimensions.push_back(col_length);
        unit_num = element_num;
        unit_size = sizeof(double);
        vals_size = unit_size * element_num;
        double *vals_double = new double[element_num];
        vals = reinterpret_cast<uint8_t *>(vals_double);
        cols = new int[element_num];

        indices.resize(row_length, 0);

        auto m_buffer =
            std::map<std::pair<int, int>, double,
                     std::function<bool(const std::pair<int, int> &, const std::pair<int, int> &)>>{
                [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                    if (a.first == b.first) {
                        return a.second < b.second;
                    }
                    return a.first < b.first;
                }};

        for (int i = 0; i < element_num; i++) {
            if (not std::getline(read_file, line_str)) {
                delete[] vals_double;
                std::cout << "wrong mtx format!" << std::endl;
                exit(1);
            }
            int x = 0;
            int y = 0;
            double v = 0.0f;
            ss.clear();
            ss << line_str;
            ss >> x >> y >> v;
            m_buffer.insert(std::make_pair(std::make_pair(x - 1, y - 1), v));
        }
        DEFAULT_ERROR_DEBUG;
        int iter_index = 0;
        for (const auto &element : m_buffer) {
            indices[element.first.first] = iter_index + 1;
            vals_double[iter_index] = element.second;
            cols[iter_index] = element.first.second;
            // std::cout << "indices[" << element.first.first << "] = " <<
            // indices[element.first.first]
            //           << std::endl;
            // std::cout << "vals_double[" << iter_index << "] = " << vals_double[iter_index]
            //           << std::endl;
            // std::cout << "cols[" << iter_index << "] = " << cols[iter_index]
            //           << std::endl;
            iter_index++;
        }
        // for (const auto &element : indices) {
        //     std::cout << element << " ";
        // }
        // std::cout << std::endl;
        // for (int i = 0; i < element_num; i++) {
        //     std::cout << vals_double[i] << " ";
        // }
        // std::cout << std::endl;
        // for (int i = 0; i < element_num; i++) {
        //     std::cout << cols[i] << " ";
        // }
        // std::cout << std::endl;
        DEFAULT_ERROR_DEBUG;
    } else {
        std::cout << "can't open" << mtx_path << "!" << std::endl;
        exit(1);
    }
    read_file.close();
    return 0;
}

int Tensor::load_dense_mtx(const std::string &mtx_path) {
    std::ifstream read_file(mtx_path);
    std::string line_str;
    std::stringstream ss;
    if (read_file.is_open()) {
        int row_length = 0;
        int col_length = 0;
        int element_num = 0;
        while (std::getline(read_file, line_str)) {
            if (line_str == "" || line_str[0] == '%') {
                continue;
            }
            ss.clear();
            ss << line_str;
            ss >> row_length >> col_length >> element_num;
            break;
        }
        dimensions.push_back(row_length);
        dimensions.push_back(col_length);
        unit_num = row_length * col_length;
        unit_size = sizeof(double);
        vals_size = unit_size * unit_num;
        double *vals_double = new double[unit_num];
        std::memset(vals_double, 0, sizeof(double) * unit_num);
        vals = reinterpret_cast<uint8_t *>(vals_double);
        for (int i = 0; i < row_length; i++) {
            for (int j = 0; j < col_length; j++) {
                vals_double[i] = 0.0f;
            }
        }

        for (int i = 0; i < element_num; i++) {
            if (not std::getline(read_file, line_str)) {
                delete[] vals_double;
                std::cout << "wrong mtx format!" << std::endl;
                exit(1);
            }
            int x = 0;
            int y = 0;
            double v = 0.0f;
            ss.clear();
            ss << line_str;
            ss >> x >> y >> v;
            vals_double[(x - 1) * col_length + y - 1] = v;
        }
    } else {
        std::cout << "can't open" << mtx_path << "!" << std::endl;
        exit(1);
    }
    return 0;
}

Tensor::Tensor(TENSOR_MODE mode_, const std::string &mtx_path) {
    if (mode_ == TENSOR_MODE::TENSOR_MODE_DENSE) {
        load_dense_mtx(mtx_path);
    } else if (mode_ == TENSOR_MODE::TENSOR_MODE_SPARSE) {
        load_sparce_mtx(mtx_path);
    } else {
        std::cout << "tensor_mode error!" << std::endl;
        exit(1);
    }
    mode = mode_;
}

void Tensor::print_(int d_index, int v_index, FpDebugLevel level) {
    int length = dimensions[d_index];
    v_index = v_index * length;
    FP_LOG(level, "[");
    for (int i = 0; i < length; i++) {
        if (d_index == (dimensions.size() - 1)) {
            FP_LOG(level, "%.10e ", reinterpret_cast<double *>(vals)[v_index + i]);
        } else {
            print_(d_index + 1, v_index + i, level);
        }
    }
    FP_LOG(level, "]\n");
    return;
}

void Tensor::save_(int d_index, int v_index, std::ofstream &ofstr) {
    int length = dimensions[d_index];
    v_index = v_index * length;
    ofstr << "[";
    for (int i = 0; i < length; i++) {
        if (d_index == (dimensions.size() - 1)) {
            ofstr << reinterpret_cast<double *>(vals)[v_index + i] << " ";
        } else {
            save_(d_index + 1, v_index + i, ofstr);
        }
    }
    ofstr << "]" << std::endl;
    return;
}

void Tensor::print(FpDebugLevel level) {
    print_(0, 0, level);
    return;
}

void Tensor::save(const std::string &mtx_path) {
    std::ofstream ofstr(mtx_path);
    if (ofstr.is_open()) {
        ofstr << std::scientific;
        save_(0, 0, ofstr);
    } else {
        std::cout << "can't open" << mtx_path << "!" << std::endl;
        exit(1);
    }
    ofstr.close();
    return;
}

Tensor::Tensor(TENSOR_MODE mode_, const std::vector<int> &dimensions_) {
    mode = mode_;
    dimensions = dimensions_;
    unit_size = sizeof(double);
    if (mode_ == TENSOR_MODE::TENSOR_MODE_DENSE) {
        vals_size = 1;
        for (int i = 0; i < dimensions.size(); i++) {
            vals_size = vals_size * dimensions[i];
        }
        vals = reinterpret_cast<uint8_t *>(new double[vals_size]);
    } else if (mode_ == TENSOR_MODE::TENSOR_MODE_SPARSE) {
        if (dimensions.size() > 1) {
            int indices_size = 1;
            for (int i = 0; i < dimensions.size() - 1; i++) {
                indices_size = indices_size * dimensions[i];
            }
            indices.resize(indices_size, 0);
        }
        vals = reinterpret_cast<uint8_t *>(new double[0]);
    } else {
        std::cout << "tensor_mode error!" << std::endl;
        exit(1);
    }
}

Tensor::Tensor(double v) {
    double *vals_double = new double[1];
    vals_double[0] = v;
    unit_size = sizeof(double);
    vals = reinterpret_cast<uint8_t *>(vals_double);
}

Tensor::~Tensor() {
    delete[] vals;
    if (cols != nullptr) {
        delete[] cols;
    }
}
