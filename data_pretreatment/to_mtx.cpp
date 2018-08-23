#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

std::vector<std::string> split_str(const std::string& input, const std::vector<char>& delimiter) {
    std::vector<std::string> ret;
    if (input == "") {
        return ret;
    }
    size_t pos_start = 0;
    size_t pos_end = 0;
    int input_length = input.size();
    int delimiter_length = delimiter.size();
    while (pos_start < input_length) {
        while (pos_start < input_length) {
            int i = 0;
            for (; i < delimiter_length; i++) {
                if (input[pos_start] != delimiter[i]) {
                    break;
                }
            }
            if (i != delimiter_length) {
                break;
            }
            pos_start++;
        }
        pos_end = pos_start + 1;
        while (pos_end < input_length) {
            int i = 0;
            for (; i < delimiter_length; i++) {
                if (input[pos_end] == delimiter[i]) {
                    break;
                }
            }
            if (i != delimiter_length) {
                break;
            }
            pos_end++;
        }
        if (pos_start != 0) {
            pos_start++;
        }
        ret.push_back(input.substr(pos_start, pos_end - pos_start));
        pos_start = pos_end;
    }
    return ret;
}
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "argc: " << argc << std::endl;
        std::cout << "paramaters error!" << std::endl;
        exit(1);
    }
    std::ifstream read_file(argv[1]);
    std::string line_str;
    std::vector<char> delimiter;
    delimiter.push_back(' ');
    delimiter.push_back('\t');
    int node_num = 0;
    int edge_num = 0;
    std::stringstream ss;
    std::vector<int> outflow_table;
    std::vector<std::pair<int, int>> edge_list;
    if (read_file.is_open()) {
        while (std::getline(read_file, line_str)) {
            if (line_str == "") {
                continue;
            }
            if (line_str[0] == '#') {
                auto token_list = split_str(line_str, delimiter);
                for (int i = 0; i < token_list.size(); i++) {
                    if(token_list[i] == "Nodes:") {
                        ss.clear();
                        ss << token_list[i+1];
                        ss >> node_num;
                        outflow_table.resize(node_num, 0);
                    } else if(token_list[i] == "Edges:") {
                        ss.clear();
                        ss << token_list[i+1];
                        ss >> edge_num;
                    }
                }
            } else {
                int out_node;
                int in_node;
                ss.clear();
                ss << line_str;
                ss >> out_node >> in_node;
                outflow_table[out_node]++;
                edge_list.push_back(std::make_pair(in_node, out_node));
            }
        }
    } else {
        std::cout << "can't open" << argv[1] << "!" << std::endl;
        exit(1);
    }
    read_file.close();
    std::ofstream write_file(argv[2]);
    if(write_file) {
        write_file << "%%MatrixMarket matrix coordinate real general" << std::endl;
        write_file << node_num << " " << node_num << " " << edge_num << std::endl;
        for(auto e: edge_list) {
            write_file << e.first << " " << e.second << " " << 1.0f/outflow_table[e.second] << std::endl;
        }
    } else {
        std::cout << "can't write" << argv[2] << "!" << std::endl;
        exit(1);
    }
    return 0;
}