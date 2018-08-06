"""Page Map Generate

   by Zhou Zhe"""

#!/usr/local/bin/python3

import argparse
import random
from datetime import datetime


def main():
    """main"""
    n, e = parse_args()
    if e < n:
        print("ERROR: e should greater than n!\n")
        return
    generate_graph(n, e)


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser(
        description='Generate Graph for the FDEB benchmark')

    parser.add_argument('num_nodes', metavar='N', type=int,
                        help='The number of nodes')
    parser.add_argument('num_edges', metavar='E', type=int,
                        help='The number of edges')

    args = parser.parse_args()
    return args.num_nodes, args.num_edges


def generate_graph(n, e):
    """generate_graph"""
    random.seed(datetime.now())
    page_map = [[0 for x in range(n)] for y in range(n)]

    for i in range(0, n):
        row_index = int(random.random() * n)
        while page_map[i][row_index] == 1:
            row_index = int(random.random() * n)
        page_map[i][row_index] = 1

    for i in range(0, e-n):
        col_index = int(random.random() * n)
        row_index = int(random.random() * n)
        while page_map[col_index][row_index] == 1:
            col_index = int(random.random() * n)
            row_index = int(random.random() * n)
        page_map[col_index][row_index] = 1

    f_edges = open('../data/' + str(n) + 'x' + str(n) + '-' + str(e) + '.mtx', 'w')

    f_edges.write("%%MatrixMarket matrix coordinate real general\n")
    f_edges.write(str(n) + ' ' + str(n) + ' ' + str(e) + '\n')

    for i in range(0, n):
        edges_num = 0
        for j in range(0, n):
            if page_map[i][j] == 1:
                edges_num = edges_num + 1
        for j in range(0, n):
            if page_map[i][j] == 1:
                page_map[i][j] = (1.0/edges_num)
                # inverse the matrix
                f_edges.write(str(j+1) + ' ' + str(i+1) + ' ' + str(page_map[i][j]) + '\n')


if __name__ == '__main__':
    main()
