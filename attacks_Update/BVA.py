# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 20:58:17 2023

@author: Ra
"""

import numpy as np
import pandas as pd
import math
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

def random_update_database(wordSet, percentage, inj_num, update_dataset, client_dataset,
                           doc, min, max): 
        operation_type = ['add', 'delete']
        update_length = {}
        update_size = {}

        up_dis = 'Uniform'
        update_count = (int) (percentage * inj_num)
        update_dataset, client_dataset = list(update_dataset), list(client_dataset)
        for _ in range(update_count):    
            
            if up_dis=='AllAdd':
                op = 'add'
            elif up_dis=='Uniform':
                op = random.choice(operation_type)
            else:
                op = 'delete'     
            if len(update_dataset)==0 and len(client_dataset)==0:
                break
            if op=='add':
                if len(update_dataset)==0:
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
                if len(client_dataset)==0:
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

                #client_dataset.pop(delete_doc_id)
                update_dataset.append(delete_doc_id)
        return pd.Series(update_length, dtype='float64'), pd.Series(update_size, dtype='float64')


def BVA_inject(kws, word_total_size, gamma):
    """
    injection: injection size and real_size_after_injection
    """
    
    word_len = len(kws)
    real_size_after_injection = pd.Series(0,index = kws)
    inject_length = pd.Series(0,index = kws)
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
                inject_length[w] += 1
    
                
    inj_file_size = [len(size) for size in F]
    max_inj_size = max(inj_file_size)
    min_inj_size = min(inj_file_size)
    return total_inject_size, real_size_after_injection, total_inject_length, inject_length, min_inj_size, max_inj_size 


def BVA_recover(queries, wordSet, observed_size_in_baseline, real_size_after_injection, gamma):
    real_tag = {}
    recover_tag = {}
    recover_queries_num = 0

    for query in queries:
        real_tag[query] = wordSet.index(query)
        # 会有计算的非索引值，所以不用真实的query了，记录其索引
        recover_tag[query] = -1
        for kw_id in observed_size_in_baseline.keys():
            if query in real_size_after_injection.keys(): 
                if (real_size_after_injection[query] - observed_size_in_baseline[kw_id]) % gamma == 0:
                    u = int( (real_size_after_injection[query] - observed_size_in_baseline[kw_id]) / gamma )
                    recover_tag[query] = u
                    break
        if recover_tag[query] == real_tag[query]:
            recover_queries_num += 1
        
    acc = recover_queries_num / len(queries)
    return acc

def BVA_main(wordSet, query, total_size, real_length, update_dataset, client_dataset, 
                  min_file_size, max_file_size, word_size_set, percentage, doc, gamma):
    
    word_total_size = total_size[wordSet]
    
    s = time.time()
    total_inject_size, injection_size, total_inject_length, injection_length, min_inj_size, max_inj_size = BVA_inject(wordSet, word_total_size, gamma)
    
    update_length, update_size = random_update_database(
        wordSet, percentage, total_inject_length, update_dataset, client_dataset, doc, min_file_size, max_file_size)
    
    size_after_injection_and_update, length_after_injection_and_update = get_size_and_length_after_injection_update(
        wordSet, real_length, total_size,
        word_size_set, injection_length, injection_size,
        update_length, update_size)
    
    acc = BVA_recover(query,wordSet,word_total_size,size_after_injection_and_update,gamma)
    runtime = time.time() - s
    return acc, total_inject_size, total_inject_length,runtime






