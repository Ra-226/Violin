# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 20:58:17 2023

@author: Ra
"""

import numpy as np
import pandas as pd
import math
import time


def get_size_and_length_after_setup_padding(real_size, real_length, x, min_file_size, max_file_size):
    """
    Padding of setup phase
    """
    if x == 0:
        return real_size, real_length
    for k in real_length.index:
        m = x
        while real_length[k] > m:
            m *= x
        for _ in range(m - real_length[k]):
            real_size[k] += np.random.randint(min_file_size, max_file_size)
        real_length[k] = m
    return real_size, real_length


def get_size_and_length_after_injection_without_padding(real_size, real_length, real_size_after_injection,
                                                        inject_length):
    length_after_injection_without_padding = pd.Series(0, index=inject_length.index)
    size_after_injection_without_padding = pd.Series(0, index=inject_length.index)
    for k in real_length.index:
        if k in inject_length.index:
            length_after_injection_without_padding[k] = real_length[k] + inject_length[k]
            size_after_injection_without_padding[k] = real_size[k] + real_size_after_injection[k]
    return size_after_injection_without_padding, length_after_injection_without_padding


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

    return total_inject_size, real_size_after_injection, total_inject_length, inject_length


def BVA_recover(queries, wordSet, observed_size_in_baseline, real_size_after_injection, gamma):
    real_tag = {}
    recover_tag = {}
    recover_queries_num = 0
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


def BVA_main(wordSet, query, wordAccess, total_size, real_length, x, gamma, min_file_size, max_file_size):
    word_total_size = total_size[wordSet]

    word_total_size, real_length = get_size_and_length_after_setup_padding(word_total_size, real_length, x,
                                                                           min_file_size, max_file_size)

    s = time.time()
    total_inject_size, inj_size, total_inject_length, inject_length = BVA_inject(wordSet, word_total_size, gamma)
    query_total_size = inj_size[query] + word_total_size[query]
    acc = BVA_recover(query, wordSet, word_total_size, query_total_size, gamma)
    runtime = time.time() - s
    return acc, total_inject_size, total_inject_length, runtime
