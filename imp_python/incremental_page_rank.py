"""Page Rank

   by Zhou Zhe"""

#!/usr/local/bin/python3

import numpy as np
import scipy.io as sio

ANS = [0.25394207, 0.13826207, 0.13826207, 0.20617133, 0.26336239]


def normalize(v_rank, active_index, inactive_index):
    """normalize verson one:
        reallocate the flow in active set, guarantee the ratios unchanged.
        Would cause the algorithm could not terminated"""
    inactive_flow = 0
    inactive_num = 0
    print("input:")
    print(v_rank)
    for iai in inactive_index:
        inactive_flow = inactive_flow + v_rank[iai]
        inactive_num = inactive_num + 1
    active_flow = 0
    for ai in active_index:
        active_flow = active_flow + v_rank[ai]
    print("active_flow:")
    print(active_flow)
    print("inactive_flow:")
    print(inactive_flow)
    for ai in active_index:
        v_rank[ai] = v_rank[ai]*(1 - inactive_flow)/active_flow
    print("normalized:")
    print(v_rank)
    return v_rank


def normalize2(v_rank):
    """normalize verson two:
        reallocate the flow of whole vector, guarantee the ratios unchanged.
        Work well."""
    ret = v_rank/np.sum(v_rank)
    print(ret)
    return ret


def normalize3(v_rank, active_index, inactive_index):
    """normalize verson two:
        reallocate the flow of whole vector, guarantee the ratios unchanged.
        Work well."""
    inactive_flow = 0
    for iai in inactive_index:
        inactive_flow = inactive_flow + v_rank[iai]
    active_flow = 0
    for ai in active_index:
        active_flow = active_flow + v_rank[ai]
    for ai in inactive_index:
        v_rank[ai] = v_rank[ai]*(1 - active_flow)/inactive_flow
    print(v_rank)
    return v_rank


def rank_once_active(morkov_mat, v_rank, active_index, inactive_index):
    """calculate page rank only once"""
    ret = np.ones((v_rank.shape[0], 1), dtype=np.float32)
    for ai in active_index:
        tmp = 0
        for j in active_index:
            tmp = tmp + v_rank[j]*morkov_mat[ai][j]
        for iai in inactive_index:
            tmp = tmp + v_rank[iai]*morkov_mat[ai][iai]
        ret[ai] = tmp
    for iai in inactive_index:
        ret[iai] = v_rank[iai]
    print("page ranked:")
    print(ret)
    return ret


def gen_markov_matrix(page_map, damping_factor):
    """gen_markov_matrix"""
    np.transpose(page_map)
    vertex_num = page_map.shape[1]
    remain_mat = (1-damping_factor)/vertex_num*np.ones((vertex_num, vertex_num), dtype=np.float32)
    print(page_map)
    print(remain_mat)
    return damping_factor*page_map+remain_mat


def find_stationary(active_table, v_rank_new, v_rank, eps,
                    active_index, inactive_index):
    """find_stationary"""
    print("v_rank")
    print(v_rank)
    print("v_rank_new")
    print(v_rank_new)
    for i in active_index:
        if abs((v_rank_new[i] - v_rank[i])/v_rank[i]) < eps:
            active_table[i] = active_table[i] + 1
            if active_table[i] > 2:
                active_index.remove(i)
                inactive_index.append(i)
    return active_index, inactive_index


def main():
    """main func of page rank"""
    # mm_page_map = sio.mmread("../data/5x5-7.mtx")
    # mm_page_map = sio.mmread("../data/10x10-20.mtx")
    mm_page_map = sio.mmread("../data/page_map.mtx")
    print(mm_page_map.A)
    page_map = np.array(mm_page_map.A)
    morkov_mat = gen_markov_matrix(page_map, 0.85)
    v_rank = np.ones((page_map.shape[1], 1), dtype=np.float32)/page_map.shape[1]
    v_rank_new = np.ones((page_map.shape[1], 1), dtype=np.float32)
    print(morkov_mat)
    print(v_rank)
    eps = 0.0001
    times = 0
    active_index_list = []
    inactive_index_list = []
    active_table = []
    for i in range(0, page_map.shape[1]):
        active_index_list.append(i)
        active_table.append(0)
    while True:
        print(">>>>>>>>>>>>>>LOOP", times)
        # if not to normalize, the sum of result vector would be greater than 1.
        # v_rank = normalize2(v_rank)
        v_rank = normalize3(v_rank, active_index_list, inactive_index_list)
        v_rank_new = rank_once_active(morkov_mat, v_rank,
                                      active_index_list, inactive_index_list)
        active_index_list, inactive_index_list = find_stationary(
            active_table, v_rank_new, v_rank, eps, active_index_list, inactive_index_list)
        print("active_index_list", active_index_list)
        print("inactive_index_list", inactive_index_list)
        if not active_index_list:
            break
        v_rank = v_rank_new
        times = times + 1
    print(v_rank)
    print(times)
    return


if __name__ == "__main__":
    main()
