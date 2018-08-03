"""Page Rank

   by Zhou Zhe"""

#!/usr/local/bin/python3

import numpy as np


def pagerank(page_map, eps=1.0e-8, d=0.85):
    vertex_num = page_map.shape[1]
    v_rank = np.random.rand(vertex_num, 1)
    v_rank = v_rank / np.linalg.norm(v_rank, 1)
    last_v_rank = np.ones((vertex_num, 1), dtype=np.float32) * np.inf
    page_map_hat = (d * page_map) + (((1 - d) / vertex_num) * np.ones((vertex_num, vertex_num),
                                                                      dtype=np.float32))

    while np.linalg.norm(v_rank - last_v_rank, 2) > eps:
        last_v_rank = v_rank
        v_rank = np.matmul(page_map_hat, v_rank)
    return v_rank


def main():
    """main func of page rank"""

    page_map = np.array([[0, 0, 0, 0, 1],
                         [0.5, 0, 0, 0, 0],
                         [0.5, 0, 0, 0, 0],
                         [0, 1, 0.5, 0, 0],
                         [0, 0, 0.5, 1, 0]])

    v_rank = pagerank(page_map, 0.001, 0.85)
    print(v_rank)


if __name__ == "__main__":
    main()
