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

def get_size_and_length_after_setup_padding(real_size, real_length, x, min_file_size, max_file_size,word_size_set):
    """
    Padding of setup phase
    """
    if x==0:
        return real_size, real_length, word_size_set
    for k in real_length.index:
        m = x
        while real_length[k]>m:
            m *= x  
        for _ in range(m - real_length[k]):
            random_size = np.random.randint(min_file_size, max_file_size)
            real_size[k] += random_size
            word_size_set[k].append(random_size)
        real_length[k] = m
    return real_size, real_length,word_size_set

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
    #M = access.dot(access.T)
    M = np.dot(access,access.T)
    M = pd.DataFrame(M,index,index)
    return M

def recovery_acc(result):
    h = 0
    for i in result.keys():
        if i==result[i]:
            h += 1
    return h / len(result)



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
    
    
    return total_inject_size, inj_size, inj_M, inj_size_set, total_inject_length, inj, inject_length


def Violin_recover(wordSet, query, wordAccess, queryAccess, total_size, word_size_set,real_length):
    
    word_len = len(wordSet)
    word_total_size = total_size[wordSet]
    total_inject_size, inj_size, inj_M, inj_size_set, total_inject_length, inj, inject_length = Violin_inject(wordSet, word_total_size)
    
    queryAccess = pd.concat([wordAccess.loc[query],inj.loc[query]],axis=1)
    M = co_o(wordAccess)
    N = co_o(queryAccess)

    query_total_size = inj_size[query] + word_total_size[query]
    
    length_after_inj_and_pad = inject_length[query] + real_length[query]
    volumeToken = length_after_inj_and_pad
    volumeKeyword = real_length

    
    inj_volume = pd.Series(np.diag(inj_M),index=M.index)
    
    query_size_set = {i:word_size_set[i]+inj_size_set[i] for i in query}
    
    s = time.time()
    K = {}
    partial_function = partial(searchPattern, wordSet, word_size_set, query_size_set)
    sub_query = Splits(query, NUM_CORES)
    with Pool(processes = NUM_CORES) as pool:
        for sub_K in pool.map(partial_function, sub_query):
            K.update(sub_K)
    print(time.time()-s)
    
    begin =time.time()
    result = {}
    for key in K.keys():
        CS = []
        for i in range(len(inj_volume)):
            if inj_volume[i] == volumeToken[key] - volumeKeyword[K[key]]:
                CS.append(wordSet[i])
        sub_size = inj_size[CS]
        unique_sub_size = sub_size.value_counts()
        for w in CS:
            if unique_sub_size[sub_size[w]] == 1 and sub_size[w] == query_total_size[key] - word_total_size[K[key]]:
                result[key] = w
    
    
    keys_of_K, values_of_K = K.keys(),K.values()

    sub_M = N.loc[keys_of_K][keys_of_K] - M.loc[values_of_K][values_of_K]
    
    if len(sub_M.index) != len(keys_of_K):
        sub_M = sub_M[~sub_M.index.duplicated(keep='first')]
        sub_M = sub_M.T[~sub_M.T.index.duplicated(keep='first')]
        sub_M = sub_M.T
    
    
    sub_inj_M = inj_M[result.values()]
    sub_sub_M = sub_M[result.keys()]
    U = set(K.keys()) - set(result.keys())
    while len(U) != 0:
        flag=len(U)
        for q in U:
            d = np.linalg.norm(sub_sub_M.loc[q] - sub_inj_M, ord=1, axis=1)
            s = (query_total_size[q] - word_total_size[K[q]]) - inj_size
            d = s - d
            if Counter(d)[0] == 1: result[q] = wordSet[list(d).index(0)] 

        U = set(K.keys()) - set(result.keys())
        sub_inj_M = inj_M[result.values()]
        sub_sub_M = sub_M[result.keys()]
        if flag==len(U): break

    print("Attack time: ",time.time() - begin)
    
    acc = recovery_acc(result)
    
    return acc, result, total_inject_size, total_inject_length


def Violin_main(wordSet, query, wordAccess, total_size, word_size_set, real_length, x, min_file_size, max_file_size):
    
    total_size, real_length, word_size_set = get_size_and_length_after_setup_padding(total_size, real_length, x, min_file_size, max_file_size,word_size_set)

    for kw in wordSet:
        wordAccess.loc[kw, list(set(word_size_set[kw]))] = 1
    
    wordAccess.fillna(0, inplace=True)
    queryAccess = wordAccess.loc[query]
    
    
    s = time.time()
    acc, result, total_inject_size, total_inject_length = Violin_recover(wordSet, query, wordAccess, queryAccess, total_size, word_size_set,real_length)
    runtime = time.time() - s
    return acc, result, total_inject_size, total_inject_length, runtime
    
    

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
    acc = recovery_acc(result)
    return acc, result, total_inject_size, total_inject_length


def Violin_set_main(wordSet, query, wordAccess, total_size, word_size_set):
    queryAccess = wordAccess.loc[query]
    s = time.time()
    acc, result, total_inject_size, total_inject_length = Violin_set_recover(wordSet, query, wordAccess, queryAccess, total_size, word_size_set)
    runtime = time.time() - s
    return acc, result, total_inject_size, total_inject_length,runtime


    