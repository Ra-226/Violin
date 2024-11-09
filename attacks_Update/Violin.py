# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:28:09 2023

@author: SU
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
import random

NUM_CORES = multiprocessing.cpu_count()

def get_size_and_length_after_injection_update(wordSet, length_at_setup, size_at_setup,
                                                size_set_at_setup, injection_length, injection_size,
                                            injection_size_set, update_length, update_size, 
                                            update_size_set, update_size_pop_set):
    """
    dict: {[keyword, inject length]} // {[keyword, inject size]}
    """
    queries_set = length_at_setup.keys()
    size_after_injection_and_update = {}
    length_after_injection_and_update = {}
    size_set_after_injection_update = {}
    total_size_after_padding = 0
    total_size_no_padding = 0
    
    
    for k in queries_set:
        length_after_injection_and_update[k] = length_at_setup[k]
        size_after_injection_and_update[k] = size_at_setup[k]
        size_set_after_injection_update[k] = copy.deepcopy(size_set_at_setup[k])
        if k in injection_length.keys():
            length_after_injection_and_update[k] += injection_length[k]
            size_after_injection_and_update[k] += injection_size[k]
            size_set_after_injection_update[k] += injection_size_set[k]
        if k in update_length.keys():
            length_after_injection_and_update[k] += update_length[k]
            size_after_injection_and_update[k] += update_size[k]
        if k in update_size_set.keys():
            size_set_after_injection_update[k] += update_size_set[k]
        if k in update_size_pop_set.keys():
            for i in update_size_pop_set[k]:
                size_set_after_injection_update[k].remove(i)

    return pd.Series(size_after_injection_and_update), pd.Series(length_after_injection_and_update), size_set_after_injection_update

def searchPattern(wordSet, word_size_set, query_size_set, query):
    word_len = len(wordSet)
    K={}
    for i in tqdm.tqdm(iterable=range(len(query)),desc="Discovering search pattern ......"):
        rsp={}
        tmp1 = Counter(query_size_set[query[i]])
        for j in range(word_len):
            tmp2 = Counter(word_size_set[wordSet[j]])
            rsp[wordSet[j]] = sum((tmp2 & tmp1).values()) / sum((tmp2 | tmp1).values())
        max_key = max(rsp, key = rsp.get)
        K[query[i]] = max_key
    return K

def Splits(query, num):
    list = [ [] for i in range(num) ]
    id = 0
    for i in query:
        list[id].append(i)
        if id < num-1:
            id += 1
        else:
            id = 0
    return list
    
    

def co_o(access):
    index = access.index
    access = access.values 
    access = access.astype(np.int32)
    access = np.ascontiguousarray(access)  #降低运算时间
    M = np.dot(access,access.T)
    M = pd.DataFrame(M,index,index)
    return M

def recovery_acc(result, size):
    h = 0
    for i in result.keys():
        if i==result[i]:
            h += 1
    return h / size



def Violin_inject(kws, word_total_size):
    word_len = len(kws)
    total_inject_size = 0
    total_inject_length = math.ceil(math.log(word_len,2))
    F = []
    for i in range(total_inject_length):
        f=[]
        for j in range(word_len):
            if bin(j)[2:].rjust(32,'0')[-(i+1)] == '1':
                f.append(kws[j])
        for pad in range(int(word_len/2) - len(f) + i):
            f.append('not_in_chosen_kws')
        F.append(f)
        total_inject_size += len(f)
        
    inj = pd.DataFrame(0,index = kws,columns = [ 'inj'+str(i) for i in range(len(F)) ])
    inj_size = pd.Series(0,index = kws)
    inject_length = pd.Series(0,index = kws)
    inj_size_set = { word:[] for word in kws}
    for w in kws:
        for i,d in enumerate(F):
            if w in d:
                inj.loc[w]['inj'+str(i)] = 1
                inj_size.loc[w] += len(d)
                inj_size_set[w].append(len(d)) 
                inject_length[w] += 1
    
    inj_M = inj.dot(inj.T)
    inj_file_size = [len(size) for size in F]
    sizeMean = np.mean(inj_file_size)
    max_inj_size = max(inj_file_size)
    min_inj_size = min(inj_file_size)

    return total_inject_size, inj_size, inj_M, inj_size_set, total_inject_length, inj, inject_length, min_inj_size,max_inj_size



def Violin_set_recover(wordSet, query, update_dataset, client_dataset, min_file_size, max_file_size,
                     total_size, length_after_setup, word_size_set, percentage, doc):
    word_total_size = total_size[wordSet]
    total_inject_size, injection_size, inj_M, injection_size_set, total_inject_length, inj, injection_length, min_inj_size,max_inj_size = Violin_inject(
        wordSet, word_total_size)
    
    update_length, update_size, update_size_set, update_size_pop_set = random_update_database(
        wordSet, percentage, total_inject_length, update_dataset, client_dataset, doc, min_inj_size,max_inj_size)
    
    size_after_injection_and_update, length_after_injection_and_update, size_set_after_injection_update = get_size_and_length_after_injection_update(
        wordSet, length_after_setup, total_size,
        word_size_set, injection_length, injection_size,
        injection_size_set, update_length, update_size, 
        update_size_set, update_size_pop_set)
    

    K = {}
    partial_function = partial(searchPattern, wordSet, word_size_set, size_set_after_injection_update)
    sub_query = Splits(query, NUM_CORES)
    with Pool(processes = NUM_CORES) as pool:
        for sub_K in pool.map(partial_function, sub_query):
            K.update(sub_K)
    begin =time.time()
    result = {}
    for key in K.keys():
        temp1, temp2 = Counter(size_set_after_injection_update[key]), Counter(word_size_set[K[key]])
        dif_size_set = temp1 - temp2
        rsp = {}
        for w in wordSet:
            temp = Counter(injection_size_set[w])
            if len(temp) != 0:
                rsp[w] = sum((dif_size_set & temp).values()) / sum((dif_size_set | temp).values())
            else:
                rsp[w] = 0
            
        max_key = max(rsp, key = rsp.get)
        result[key] = max_key

    print("Attack time: ",time.time() - begin)
    acc = recovery_acc(result, len(query))
    return acc, result, total_inject_size, total_inject_length

def random_update_database(wordSet, percentage, inj_num, update_dataset, client_dataset,
                           doc, min, max): 
        operation_type = ['add', 'delete']
        update_length = {}
        update_size = {}
        update_size_set = {}
        update_size_pop_set = {}
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
                    if kw in update_size_set.keys():
                        update_size_set[kw].append(len(add_doc))
                    else:
                        update_size_set[kw] = [len(add_doc)]
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
                    if kw in update_size_pop_set.keys():
                        update_size_pop_set[kw].append(len(delete_doc))
                    else:
                        update_size_pop_set[kw] = [len(delete_doc)]
                client_dataset.remove(delete_doc_id)
                update_dataset.append(delete_doc_id)
        return pd.Series(update_length, dtype='float64'), pd.Series(update_size, dtype='float64'), update_size_set, update_size_pop_set

def Violin_set_main(wordSet, query, total_size, real_length, update_dataset, client_dataset, 
                  min_file_size, max_file_size, word_size_set, percentage, doc):
    s = time.time()
    acc, result, total_inject_size, total_inject_length = Violin_set_recover(
        wordSet, query, update_dataset, client_dataset, min_file_size, max_file_size,
        total_size, real_length, word_size_set, percentage, doc)
    runtime = time.time() - s
    return acc, result, total_inject_size, total_inject_length, runtime


    