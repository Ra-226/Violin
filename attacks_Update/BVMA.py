# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:30:44 2023

@author: SU
"""
import math
import pandas as pd
import numpy as np
import time
import random


def get_size_and_length_after_injection_update(wordSet, length_at_setup, size_at_setup,
                                               size_set_at_setup, injection_length, injection_size,
                                               update_length, update_size):
    """
    dict: {[keyword, inject length]} // {[keyword, inject size]}
    """
    queries_set = length_at_setup.keys()
    size_after_injection_and_update = {}
    length_after_injection_and_update = {}

    for k in queries_set:
        length_after_injection_and_update[k] = length_at_setup[k]
        size_after_injection_and_update[k] = size_at_setup[k]

        if k in injection_length.keys():
            length_after_injection_and_update[k] += injection_length[k]
            size_after_injection_and_update[k] += injection_size[k]

        if k in update_length.keys():
            length_after_injection_and_update[k] += update_length[k]
            size_after_injection_and_update[k] += update_size[k]

    return pd.Series(size_after_injection_and_update), pd.Series(length_after_injection_and_update)


def Violin_inject(kws, word_total_size):
    word_len = len(kws)
    total_inject_size = 0
    total_inject_length = math.ceil(math.log(word_len, 2))
    F = []
    for i in range(total_inject_length):
        f = []
        for j in range(word_len):
            if bin(j)[2:].rjust(32, '0')[-(i + 1)] == '1':
                f.append(kws[j])
        for pad in range(int(word_len / 2) - len(f) + 2 ** i):
            f.append('not_in_chosen_kws')
        F.append(f)
        total_inject_size += len(f)

    inj = pd.DataFrame(0, index=kws, columns=['inj' + str(i) for i in range(len(F))])
    inj_size = pd.Series(0, index=kws)
    inject_length = pd.Series(0, index=kws)
    # inj_size_set = { word:[] for word in kws}
    for w in kws:
        for i, d in enumerate(F):
            if w in d:
                inj.loc[w]['inj' + str(i)] = 1
                inj_size.loc[w] += len(d)
                inject_length[w] += 1

    inj_file_size = [len(size) for size in F]
    max_inj_size = max(inj_file_size)
    min_inj_size = min(inj_file_size)
    return total_inject_size, inj_size, total_inject_length, inj, inject_length, min_inj_size, max_inj_size


def BVMA_NoSP_recover(Q, chosen_kws, observed_size_in_baseline, observed_length_in_baseline, real_size_after_injection,
                      real_length_after_injection):
    real_tag = {}
    recover_tag = {}
    recover_queries_num = 0
    for query in Q:
        real_tag[query] = query
        recover_tag[query] = -1
        # findflag = False
        for kw_id in observed_size_in_baseline.keys():
            if query in real_size_after_injection.keys() and real_size_after_injection[query] - \
                    observed_size_in_baseline[kw_id] >= 0 and (
                    real_size_after_injection[query] - observed_size_in_baseline[kw_id]) - (
                    len(chosen_kws) / 2) * math.log2(len(chosen_kws)) < len(chosen_kws):
                diffReBa = (int)((real_size_after_injection[query] - observed_size_in_baseline[kw_id]))
                t_counteae = 0
                num_tF = 0
                while (diffReBa >= 0):
                    diffReBa -= (int)(len(chosen_kws) / 2)
                    if diffReBa < 0:
                        break
                    t_counteae += 1
                    if diffReBa < len(chosen_kws):
                        re_kw_id = diffReBa
                        tmp_kw_id = re_kw_id  # recovery kw_id
                        num_tF = 0  # theoretical injection length
                        while tmp_kw_id != 0:
                            if tmp_kw_id & 1 == 1:
                                num_tF += 1
                            tmp_kw_id >>= 1
                        if t_counteae != num_tF:
                            continue
                        if real_length_after_injection[query] - observed_length_in_baseline[kw_id] == num_tF:
                            # recover_tag[query] = re_kw_id
                            recover_tag[query] = chosen_kws[re_kw_id]

        if recover_tag[query] == real_tag[query]:
            recover_queries_num += 1
        # total_queries_num += 1
    acc = recover_queries_num / len(Q)
    return recover_tag, real_tag, acc


def random_update_database(wordSet, percentage, inj_num, update_dataset, client_dataset,
                           doc, min, max):
    operation_type = ['add', 'delete']
    update_length = {}
    update_size = {}

    up_dis = 'Uniform'
    update_count = (int)(percentage * inj_num)
    update_dataset, client_dataset = list(update_dataset), list(client_dataset)
    for _ in range(update_count):

        if up_dis == 'AllAdd':
            op = 'add'
        elif up_dis == 'Uniform':
            op = random.choice(operation_type)
        else:
            op = 'delete'
        if len(update_dataset) == 0 and len(client_dataset) == 0:
            break
        if op == 'add':
            if len(update_dataset) == 0:
                continue
            add_doc_id = random.choice(update_dataset)
            add_doc = doc[add_doc_id]
            if len(add_doc) == 0:
                add_doc = random.sample(wordSet, np.random.randint(min, max))
            for kw in add_doc:
                if kw not in wordSet:
                    continue
                if kw in update_length.keys():
                    update_length[kw] += 1
                    update_size[kw] += len(add_doc)
                else:
                    update_length[kw] = 1
                    update_size[kw] = len(add_doc)
            client_dataset.append(add_doc_id)
            update_dataset.remove(add_doc_id)
        else:
            if len(client_dataset) == 0:
                continue
            delete_doc_id = random.choice(client_dataset)
            delete_doc = doc[delete_doc_id]
            for kw in delete_doc:
                if kw not in wordSet:
                    continue
                if kw in update_length.keys():
                    update_length[kw] -= 1
                    update_size[kw] -= len(delete_doc)
                else:
                    update_length[kw] = -1
                    update_size[kw] = -len(delete_doc)

                    # client_dataset.pop(delete_doc_id)
            update_dataset.append(delete_doc_id)
    return pd.Series(update_length, dtype='float64'), pd.Series(update_size, dtype='float64')


def BVMA_NoSP_main(wordSet, query, total_size, real_length, update_dataset, client_dataset,
                   min_file_size, max_file_size, word_size_set, percentage, doc):
    word_total_size = total_size[wordSet]
    s = time.time()
    total_inject_size, injection_size, total_inject_length, inj, injection_length, min_inj_size, max_inj_size = Violin_inject(
        wordSet, word_total_size)

    update_length, update_size = random_update_database(
        wordSet, percentage, total_inject_length, update_dataset, client_dataset, doc, min_inj_size, max_inj_size)

    size_after_injection_and_update, length_after_injection_and_update = get_size_and_length_after_injection_update(
        wordSet, real_length, total_size,
        word_size_set, injection_length, injection_size,
        update_length, update_size)

    volumeToken = length_after_injection_and_update
    volumeKeyword = real_length

    recover_tag, real_tag, acc = BVMA_NoSP_recover(query, wordSet, word_total_size, volumeKeyword,
                                                   size_after_injection_and_update, volumeToken)
    runtime = time.time() - s
    return acc, total_inject_size, total_inject_length, runtime
