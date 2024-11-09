# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:29:16 2024

@author: Ra
"""

import math
import pandas as pd
import numpy as np
import tqdm
import time
from collections import Counter
import multiprocessing
from multiprocessing import Pool
from functools import partial
import pickle
import copy

NUM_CORES = multiprocessing.cpu_count()


def Group_cluster(ShieldAlpha):
    """
    Caching cluster
    """
    new_keyword_identifier = np.random.permutation(3000)

    Group = []
    subgroup = []
    count = 0
    for i in range(len(new_keyword_identifier)):
        if count == ShieldAlpha:
            Group.append(subgroup)
            count = 0
            subgroup = []
        subgroup.append(new_keyword_identifier[i])
        count += 1
    if len(subgroup) != 0:
        Group.append(subgroup)
    return Group


def get_size_and_length_after_setup_padding(word_set, real_size, real_length, group,
                                            min_file_size, max_file_size, word_size_set):
    """
    Padding of setup phase
    """
    queries_set = real_length.keys()
    size_after_setup_padding = pd.Series(0, queries_set)
    length_after_setup_padding = pd.Series(0, queries_set)
    size_set_after_setup_padding = copy.deepcopy(word_size_set)
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
                size_set_after_setup_padding[k].append(random_size)
                total_size_after_padding += random_size
            length_after_setup_padding[k] = max_length_of_each_cluster

    return size_after_setup_padding, length_after_setup_padding, size_set_after_setup_padding, total_size_after_padding / total_size_no_padding


def get_size_and_length_after_injection_padding(word_set, size_after_setup_padding,
                                                length_after_setup_padding, group,
                                                min_file_size, max_file_size, size_set_after_setup_padding,
                                                injection_length, injection_size, injection_size_set):
    """
    dict: {[keyword, inject length]} // {[keyword, inject size]}
    """
    queries_set = length_after_setup_padding.keys()
    size_after_injection_padding = {}
    length_after_injection_padding = {}
    size_set_after_injection_padding = copy.deepcopy(size_set_after_setup_padding)
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
            size_set_after_injection_padding[k] += injection_size_set[k]

            total_size_after_padding += injection_size[k]
            total_size_no_padding += injection_size[k]

            if max_length_of_each_cluster - injection_length[k] > 20:
                random_size = np.random.randint(min_file_size, max_file_size)
                total_size_after_padding += (max_length_of_each_cluster - injection_length[k]) * random_size
                size_after_injection_padding[k] += (max_length_of_each_cluster - injection_length[k]) * random_size
                size_set_after_injection_padding[k] += [random_size] * (
                            max_length_of_each_cluster - injection_length[k])
            else:
                for _ in range(max_length_of_each_cluster - injection_length[k]):
                    random_size = np.random.randint(min_file_size, max_file_size)
                    size_after_injection_padding[k] += random_size
                    size_set_after_injection_padding[k].append(random_size)
                    total_size_after_padding += random_size
            length_after_injection_padding[k] += max_length_of_each_cluster
    return pd.Series(size_after_injection_padding), pd.Series(
        length_after_injection_padding), size_set_after_injection_padding, total_size_after_padding / total_size_no_padding


def searchPattern(wordSet, word_size_set, query_size_set, query):
    word_len = len(wordSet)
    K = {}
    for i in tqdm.tqdm(iterable=range(len(query)), desc="Discovering search pattern ......"):
        rsp = {}
        tmp1 = Counter(query_size_set[query[i]])
        for j in range(word_len):
            tmp2 = Counter(word_size_set[wordSet[j]])
            rsp[wordSet[j]] = sum((tmp2 & tmp1).values()) / sum((tmp2 | tmp1).values())
        max_key = max(rsp, key=rsp.get)
        K[query[i]] = max_key
    return K


def Splits(query, num):
    list = [[] for i in range(num)]
    id = 0
    for i in query:
        list[id].append(i)
        if id < num - 1:
            id += 1
        else:
            id = 0
    return list


def co_o(access):
    index = access.index
    access = access.values
    access = access.astype(np.int32)
    access = np.ascontiguousarray(access)  # 降低运算时间
    M = np.dot(access, access.T)
    M = pd.DataFrame(M, index, index)
    return M


def recovery_acc(result, size):
    h = 0
    for i in result.keys():
        if i == result[i]:
            h += 1
    return h / size


def Violin_inject(kws, word_total_size, padding):
    word_len = len(kws)
    total_inject_size = 0
    total_inject_length = math.ceil(math.log(word_len, 2))
    F = []
    for i in range(total_inject_length):
        f = []
        for j in range(word_len):
            if bin(j)[2:].rjust(32, '0')[-(i + 1)] == '1':
                f.append(kws[j])
        for pad in range(int(word_len / 2) - len(f) + i * padding):
            f.append('not_in_chosen_kws')
        F.append(f)
        total_inject_size += len(f)

    inj = pd.DataFrame(0, index=kws, columns=['inj' + str(i) for i in range(len(F))])
    inj_size = pd.Series(0, index=kws)
    inject_length = pd.Series(0, index=kws)
    inj_size_set = {word: [] for word in kws}
    for w in kws:
        for i, d in enumerate(F):
            if w in d:
                inj.loc[w]['inj' + str(i)] = 1
                inj_size.loc[w] += len(d)
                inj_size_set[w].append(len(d))
                inject_length[w] += 1

    inj_M = inj.dot(inj.T)
    inj_file_size = [len(size) for size in F]
    sizeMean = np.mean(inj_file_size)
    max_inj_size = max(inj_file_size)
    min_inj_size = min(inj_file_size)

    return total_inject_size, inj_size, inj_M, inj_size_set, total_inject_length, inj, inject_length, min_inj_size, max_inj_size


def Violin_set_recover(wordSet, word_set, query, group, min_file_size, max_file_size,
                       total_size, length_after_setup_padding, word_size_set, pad):
    word_len = len(wordSet)
    word_total_size = total_size[wordSet]
    (total_inject_size, injection_size, inj_M, injection_size_set, total_inject_length,
     inj, injection_length, min_inj_size, max_inj_size) = Violin_inject(wordSet, word_total_size, pad)

    (size_after_injection_padding, length_after_injection_padding,
     size_set_after_injection_padding, inj_Overhead) = get_size_and_length_after_injection_padding(word_set, total_size,
                                                                                                   length_after_setup_padding,
                                                                                                   group,
                                                                                                   min_inj_size,
                                                                                                   max_inj_size,
                                                                                                   word_size_set,
                                                                                                   injection_length,
                                                                                                   injection_size,
                                                                                                   injection_size_set)

    K = {}
    partial_function = partial(searchPattern, wordSet, word_size_set, size_set_after_injection_padding)
    sub_query = Splits(query, NUM_CORES)
    with Pool(processes=NUM_CORES) as pool:
        for sub_K in pool.map(partial_function, sub_query):
            K.update(sub_K)

    begin = time.time()
    result = {}
    for key in K.keys():
        temp1, temp2 = Counter(size_set_after_injection_padding[key]), Counter(word_size_set[K[key]])

        dif_size_set = temp1 - temp2
        rsp = {}
        for w in wordSet:
            temp = Counter(injection_size_set[w])
            if len(temp) != 0:
                rsp[w] = sum((dif_size_set & temp).values()) / sum((dif_size_set | temp).values())
            else:
                rsp[w] = 0

        max_key = max(rsp, key=rsp.get)
        result[key] = max_key
    print("Attack time: ", time.time() - begin)

    acc = recovery_acc(result, len(query))
    return acc, result, total_inject_size, total_inject_length, inj_Overhead


def Violin_set_main(wordSet, word_set, query, total_size, real_length,
                    min_file_size, max_file_size, word_size_set, group, pad):
    size_after_setup_padding, length_after_setup_padding, size_set_after_setup_padding, setup_Overhead = get_size_and_length_after_setup_padding(
        word_set, total_size, real_length, group, min_file_size, max_file_size, word_size_set)

    s = time.time()
    acc, result, total_inject_size, total_inject_length, inj_Overhead = Violin_set_recover(
        wordSet, word_set, query, group, min_file_size, max_file_size,
        size_after_setup_padding, length_after_setup_padding, size_set_after_setup_padding, pad)
    runtime = time.time() - s
    return acc, result, total_inject_size, total_inject_length, runtime, setup_Overhead, inj_Overhead
