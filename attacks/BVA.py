# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 20:58:17 2023

@author: Ra
"""

import numpy as np
import pandas as pd
import math
import time

def BVA_inject(kws, word_total_size, gamma):
    """
    injection: injection size and real_size_after_injection
    """
    
    word_len = len(kws)
    real_size_after_injection = pd.Series(0,index = kws)
    total_inject_size = 0
    total_inject_length = math.ceil(np.log2(len(kws)))
    
    F = []
    for i in range(total_inject_length):
        f=[]
        for j in range(word_len):
            if bin(j)[2:].rjust(32,'0')[-(i+1)] == '1':
                f.append(kws[j])
        for pad in range( (2**i)*gamma - len(f) ):
            f.append('not_in_chosen_kws')
        F.append(f)
        total_inject_size += len(f)
    
    for w in kws:
        for i,d in enumerate(F):
            if w in d:
                real_size_after_injection[w] += len(d)
                
    return total_inject_size, real_size_after_injection, total_inject_length


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
                    u = int( (real_size_after_injection[query] - observed_size_in_baseline[kw_id]) / gamma )
                    recover_tag[query] = u
                    #print(query,wordSet[u-1],wordSet[u])
                    break
        if recover_tag[query] == real_tag[query]:
            recover_queries_num += 1
        
    acc = recover_queries_num / len(queries)
    return acc

def BVA_main(wordSet, query, wordAccess, total_size, gamma):
    word_total_size = total_size[wordSet]
    s = time.time()
    total_inject_size, inj_size, total_inject_length = BVA_inject(wordSet, word_total_size, gamma)
    
    query_total_size = inj_size[query] + word_total_size[query]

    
    acc = BVA_recover(query,wordSet,word_total_size,query_total_size,gamma)
    runtime = time.time() - s
    return acc, total_inject_size, total_inject_length,runtime






