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

NUM_CORES = multiprocessing.cpu_count()

def searchPattern(wordSet, word_size_set, query_size_set, query):
    word_len = len(wordSet)
    K={}
    for i in tqdm.tqdm(iterable=range(len(query)),desc="Discovering search pattern ......"):
        rsp={}
        for j in range(word_len):
            rsp[wordSet[j]]=len(set(word_size_set[wordSet[j]]) & set(query_size_set[query[i]]))/len(set(word_size_set[wordSet[j]]) | set(query_size_set[query[i]]))
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

def recovery_acc(result, length):
    h = 0
    for i in result.keys():
        if i==result[i]:
            h += 1
    return h / length



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
    inj_size_set = { word:[] for word in kws}
    for w in kws:
        for i,d in enumerate(F):
            if w in d:
                inj.loc[w]['inj'+str(i)] = 1
                inj_size.loc[w] += len(d)
                inj_size_set[w].append(len(d)) 
    
    inj_M = inj.dot(inj.T)
    inj_file_size = [len(size) for size in F]
    sizeMean = np.mean(inj_file_size)
    max_inj_size = max(inj_file_size)
    min_inj_size = min(inj_file_size)
    
    return total_inject_size, inj_size, inj_M, inj_size_set, total_inject_length, inj

    
    

def Violin_set_recover(wordSet, query, wordAccess, queryAccess, total_size, word_size_set):
    word_total_size = total_size[wordSet]
    total_inject_size, inj_size, inj_M, inj_size_set, total_inject_length, inj = Violin_inject(wordSet, word_total_size)
    
    queryAccess = pd.concat([wordAccess.loc[query],inj.loc[query]],axis=1)
    M = co_o(wordAccess)
    N = co_o(queryAccess)

    
    volumeToken = pd.Series(np.diag(N),index=N.index)
    volumeKeyword = pd.Series(np.diag(M),index=M.index)
    inj_volume = pd.Series(np.diag(inj_M),index=M.index)
    
    query_size_set = {i:word_size_set[i]+inj_size_set[i] for i in query}
    
    s = time.time()
    K = {}
    partial_function = partial(searchPattern, wordSet, word_size_set, query_size_set)
    sub_query = Splits(query, NUM_CORES)
    with Pool(processes = NUM_CORES) as pool:
        for sub_K in pool.map(partial_function, sub_query):
            K.update(sub_K)
    
    begin =time.time()
    result = {}
    for key in K.keys():
        CS = []
        for i in range(len(inj_volume)):
            if inj_volume[i] == volumeToken[key] - volumeKeyword[K[key]]:
                CS.append(wordSet[i])
        temp1, temp2 = Counter(query_size_set[key]), Counter(word_size_set[K[key]])
        
        dif_size_set = temp1 - temp2
        rsp = {}
        for w in CS:
            temp = Counter(inj_size_set[w])
            if len(temp) != 0:
                rsp[w] = sum((dif_size_set & temp).values()) / sum((dif_size_set | temp).values())
            else:
                rsp[w] = 1
        max_key = max(rsp, key = rsp.get)
        result[key] = max_key

    print("Attack time: ",time.time() - begin)
    
    acc = recovery_acc(result, len(query))
    
    return acc, result, total_inject_size, total_inject_length



def Violin_set_main(wordSet, query, wordAccess, total_size, word_size_set):
    queryAccess = wordAccess.loc[query]
    s = time.time()
    acc, result, total_inject_size, total_inject_length = Violin_set_recover(wordSet, query, wordAccess, queryAccess, total_size, word_size_set)
    runtime = time.time() - s
    return acc, result, total_inject_size, total_inject_length,runtime


    