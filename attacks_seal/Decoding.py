# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:17:57 2023

@author: SU
"""
import numpy as np
import pandas as pd
import time

from numba import njit


@njit(nopython=True) 
def find_offset(dif, kws_len):
    offset = kws_len + 1
    while(True):
       temp = dif % offset
       if 0 not in temp:
           break
       offset += 1
    return offset

def Decoding_inject(kws, word_total_size):
    """
    injection: injection size and real_size_after_injection
    """
    dif = []
    real_size_after_injection = pd.Series(0,index = kws)
    total_inject_size = 0
    total_inject_length = len(kws)
    for i in range(total_inject_length):
        for j in range(i+1,total_inject_length):
            dif.append(abs(word_total_size[i]-word_total_size[j]))
    
    while(True):
        if 0 in dif:
            dif.remove(0)
        else:
            break
    dif = np.array(dif)
    offset = find_offset(dif, len(kws))
    
    for kw_id in range(total_inject_length):
        total_inject_size += (kw_id+1)*offset 
        real_size_after_injection[kws[kw_id]] += (kw_id+1)*offset
    return total_inject_size, real_size_after_injection, total_inject_length, offset


def Decoding_recover(queries, wordSet, observed_size_in_baseline, real_size_after_injection, offset):
    real_tag = {}
    recover_tag = {}
    recover_queries_num = 0

    for query in queries:
        real_tag[query] = query
        recover_tag[query] = -1
        for kw_id in observed_size_in_baseline.keys():
            if query in real_size_after_injection.keys(): 
                if (real_size_after_injection[query] - observed_size_in_baseline[kw_id]) % offset == 0:
                    u = int( (real_size_after_injection[query] - observed_size_in_baseline[kw_id]) / offset )
                    recover_tag[query] = wordSet[u-1]
                    break
        if recover_tag[query] == real_tag[query]:
            recover_queries_num += 1
    
    acc = recover_queries_num / len(queries)
    return acc



def Decoding_main(wordSet, query, wordAccess, total_size):
    word_total_size = total_size[wordSet]
    
    s = time.time()
    total_inject_size, inj_size, total_inject_length, offset = Decoding_inject(wordSet, word_total_size)
    query_total_size = inj_size[query] + word_total_size[query]
    acc = Decoding_recover(query,wordSet,word_total_size,query_total_size,offset)
    runtime = time.time() - s
    return acc, total_inject_size, total_inject_length, offset,runtime



