# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 20:58:17 2023

@author: Ra
"""

import numpy as np
import pandas as pd
import math
import time


def get_size_and_length_after_setup_padding(word_set, real_size, real_length, group,
                                            min_file_size, max_file_size):
    """
    Padding of setup phase
    """
    queries_set = real_length.keys()
    size_after_setup_padding = pd.Series(0, queries_set)
    length_after_setup_padding = pd.Series(0, queries_set)
    total_size_after_padding = 0
    total_size_no_padding = 0
    for gp in group:
        max_length_of_each_cluster = 0
        for i in gp:
            k = word_set[i]
            if k not in queries_set:
                continue
            if real_length[k] > max_length_of_each_cluster:
                max_length_of_each_cluster = real_length[k]
        for i in gp:
            k = word_set[i]
            if k not in queries_set:
                continue
            size_after_setup_padding[k] = real_size[k]
            total_size_no_padding += real_size[k]
            total_size_after_padding += real_size[k]
            for _ in range(max_length_of_each_cluster - real_length[k]):
                random_size = np.random.randint(min_file_size, max_file_size)
                size_after_setup_padding[k] += random_size
                total_size_after_padding += random_size
            length_after_setup_padding[k] = max_length_of_each_cluster

    return size_after_setup_padding, length_after_setup_padding, total_size_after_padding / total_size_no_padding


def get_size_and_length_after_injection_padding(word_set, size_after_setup_padding,
                                                length_after_setup_padding, group,
                                                min_file_size, max_file_size,
                                                injection_length, injection_size):
    """
    dict: {[keyword, inject length]} // {[keyword, inject size]}
    """
    queries_set = length_after_setup_padding.keys()
    size_after_injection_padding = {}
    length_after_injection_padding = {}
    total_size_after_padding = 0
    total_size_no_padding = 0
    for Gp in group:
        max_length_of_each_cluster = 0
        for i in Gp:
            k = word_set[i]
            if k not in queries_set:
                continue
            if injection_length[k] > max_length_of_each_cluster:
                max_length_of_each_cluster = injection_length[k]
        for i in Gp:
            k = word_set[i]
            if k not in queries_set:
                continue
            size_after_injection_padding[k] = size_after_setup_padding[k]
            length_after_injection_padding[k] = length_after_setup_padding[k]

            size_after_injection_padding[k] += injection_size[k]

            total_size_after_padding += injection_size[k]
            total_size_no_padding += injection_size[k]

            if max_length_of_each_cluster - injection_length[k] > 20:
                random_size = np.random.randint(min_file_size, max_file_size)
                total_size_after_padding += (max_length_of_each_cluster - injection_length[k]) * random_size
                size_after_injection_padding[k] += (max_length_of_each_cluster - injection_length[k]) * random_size
            else:
                for _ in range(max_length_of_each_cluster - injection_length[k]):
                    random_size = np.random.randint(min_file_size, max_file_size)
                    size_after_injection_padding[k] += random_size
                    total_size_after_padding += random_size
            length_after_injection_padding[k] += max_length_of_each_cluster
    return pd.Series(size_after_injection_padding), pd.Series(
        length_after_injection_padding), total_size_after_padding / total_size_no_padding


def BVA_inject(kws, word_total_size, gamma):
    """
    injection: injection size and real_size_after_injection
    """

    word_len = len(kws)
    real_size_after_injection = pd.Series(0, index=kws)
    inject_length = pd.Series(0, index=kws)
    total_inject_size = 0
    total_inject_length = math.ceil(np.log2(len(kws)))

    F = []
    for i in range(total_inject_length):
        f = []
        for j in range(word_len):
            if bin(j)[2:].rjust(32, '0')[-(i + 1)] == '1':
                f.append(kws[j])
        for pad in range((2 ** i) * gamma - len(f)):
            f.append('not_in_chosen_kws')
        F.append(f)
        total_inject_size += len(f)

    for w in kws:
        for i, d in enumerate(F):
            if w in d:
                real_size_after_injection[w] += len(d)
                inject_length[w] += 1

    inj_file_size = [len(size) for size in F]
    max_inj_size = max(inj_file_size)
    min_inj_size = min(inj_file_size)
    sizeMean = np.mean(inj_file_size)
    return total_inject_size, real_size_after_injection, total_inject_length, inject_length, min_inj_size, max_inj_size


def BVA_recover(queries, wordSet, observed_size_in_baseline, real_size_after_injection, gamma):
    real_tag = {}
    recover_tag = {}
    recover_queries_num = 0
    test_list = []

    for query in queries:
        real_tag[query] = wordSet.index(query)
        recover_tag[query] = -1
        for kw_id in observed_size_in_baseline.keys():
            if query in real_size_after_injection.keys():
                if (real_size_after_injection[query] - observed_size_in_baseline[kw_id]) % gamma == 0:
                    u = int((real_size_after_injection[query] - observed_size_in_baseline[kw_id]) / gamma)
                    recover_tag[query] = u
                    break
        if recover_tag[query] == real_tag[query]:
            recover_queries_num += 1

    acc = recover_queries_num / len(queries)
    return acc


def BVA_main(wordSet, word_set, query, total_size, real_length, gamma, min_file_size, max_file_size, group):
    word_total_size = total_size[wordSet]

    word_total_size, real_length, setup_Overhead = get_size_and_length_after_setup_padding(word_set, total_size,
                                                                                           real_length, group,
                                                                                           min_file_size, max_file_size)

    s = time.time()
    total_inject_size, injection_size, total_inject_length, injection_length, min_inj_size, max_inj_size = BVA_inject(
        wordSet, word_total_size, gamma)

    size_after_injection_padding, length_after_injection_padding, inj_Overhead = (
        get_size_and_length_after_injection_padding(word_set, word_total_size, real_length, group,
                                                    min_inj_size, max_inj_size, injection_length, injection_size))

    acc = BVA_recover(query, wordSet, word_total_size, size_after_injection_padding, gamma)
    runtime = time.time() - s
    return acc, total_inject_size, total_inject_length, runtime, setup_Overhead, inj_Overhead
