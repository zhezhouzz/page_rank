"""Page Rank

   by Zhou Zhe"""

#!/usr/local/bin/python3

import math
import numpy as np
import scipy.io


def pagerank(page_map, eps=1.0e-8, d=0.85):
    """pagerank v1
        With the np.linalg.norm
        """

    vertex_num = page_map.shape[1]
    v_rank = np.ones((vertex_num, 1), dtype=np.float32)
    v_rank = v_rank/vertex_num
    last_v_rank = np.ones((vertex_num, 1), dtype=np.float32)
    page_map_hat = (d * page_map) + (((1 - d) / vertex_num) * np.ones((vertex_num, vertex_num),
                                                                      dtype=np.float32))
    print(v_rank)
    print(last_v_rank)
    print(np.linalg.norm(v_rank - last_v_rank, 2))
    loop_num = 0
    while np.linalg.norm(v_rank - last_v_rank, 2) > eps:
        print("loop num = %d", loop_num)
        print("current eps = %f", np.linalg.norm(v_rank - last_v_rank, 2))
        last_v_rank = v_rank
        v_rank = np.matmul(page_map_hat, v_rank)
        loop_num = loop_num + 1
    return v_rank


def eucidean_distance(vector_v):
    """eucidean_distance"""
    element_sum = 0
    for element in vector_v:
        element_sum = element_sum + element[0]*element[0]
    return math.sqrt(element_sum)


def pagerank_v2(page_map, eps=1.0e-8, d=0.85):
    """pagerank v1
        With the eucidean_distance
        """

    vertex_num = page_map.shape[1]
    v_rank = np.ones((vertex_num, 1), dtype=np.float32)
    v_rank = v_rank/vertex_num
    last_v_rank = np.ones((vertex_num, 1), dtype=np.float32)
    page_map_hat = (d * page_map) + (((1 - d) / vertex_num) * np.ones((vertex_num, vertex_num),
                                                                      dtype=np.float32))

    loop_num = 0
    while eucidean_distance((v_rank - last_v_rank).tolist()) > eps:
        print("loop num = %d", loop_num)
        print("current eps = %f", eucidean_distance((v_rank - last_v_rank).tolist()))
        last_v_rank = v_rank
        v_rank = np.matmul(page_map_hat, v_rank)
        print("v_rank=")
        print(v_rank)
        loop_num = loop_num + 1
    return v_rank


def rank_once(page_map, v_rank, d, vertex_num):
    """calculate page rank only once"""
    print((1-d)/vertex_num)
    return d*np.matmul(page_map, v_rank) + (1-d)/vertex_num


def pagerank_v3(page_map, eps=1.0e-8, d=0.85):
    """pagerank v3
        With the eucidean_distance
        sparse matrix multiply vector, then add d
        """

    vertex_num = page_map.shape[1]
    v_rank = np.ones((vertex_num, 1), dtype=np.float32)
    v_rank = v_rank/vertex_num
    last_v_rank = np.ones((vertex_num, 1), dtype=np.float32)

    loop_num = 0
    while eucidean_distance((v_rank - last_v_rank).tolist()) > eps:
        print("loop num =", loop_num)
        print("current eps =", eucidean_distance((v_rank - last_v_rank).tolist()))
        last_v_rank = v_rank
        v_rank = rank_once(page_map, v_rank, d, vertex_num)
        print("v_rank=")
        print(v_rank)
        loop_num = loop_num + 1

    return v_rank


def main():
    """main func of page rank"""

    page_map = np.array([[0, 0, 0, 0, 1],
                         [0.5, 0, 0, 0, 0],
                         [0.5, 0, 0, 0, 0],
                         [0, 1, 0.5, 0, 0],
                         [0, 0, 0.5, 1, 0]])

    v_rank = pagerank_v3(page_map, 1.0e-8, 0.85)

    print(v_rank)


if __name__ == "__main__":
    main()
