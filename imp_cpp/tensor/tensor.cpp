#include "tensor.h"
#include <fstream>
#include <iostream>
#include <sstream>

int Tensor::load_sparce_mtx(const std::string &mtx_path) {
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

        indices.resize(row_length, 0);

        double *m_buffer = new double[row_length * col_length];
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
            m_buffer[(x - 1) * col_length + y - 1] = v;
        }
        int val_iter = 0;
        for (int i = 0; i < row_length; i++) {
            for (int j = 0; j < col_length; j++) {
                if (m_buffer[i * col_length + j] != 0) {
                    vals_double[val_iter] = m_buffer[i * col_length + j];
                    val_iter++;
                }
            }
            indices[i] = val_iter;
        }
        delete[] m_buffer;
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

Tensor::Tensor(TENSOR_MODE mode_, const std::vector<uint32_t> &dimensions_) {
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

Tensor::~Tensor() { delete[] vals; }
