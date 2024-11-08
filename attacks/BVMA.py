# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:30:44 2023

@author: SU
"""
import math
import pandas as pd
import numpy as np
import time

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
        for pad in range(int(word_len/2) - len(f) + 2**i):
            f.append('not_in_chosen_kws')
        F.append(f)
        total_inject_size += len(f)
        
    inj = pd.DataFrame(0,index = kws,columns = [ 'inj'+str(i) for i in range(len(F)) ])
    inj_size = pd.Series(0,index = kws)
    #inj_size_set = { word:[] for word in kws}
    for w in kws:
        for i,d in enumerate(F):
            if w in d:
                inj.loc[w]['inj'+str(i)] = 1
                inj_size.loc[w] += len(d)
                #inj_size_set[w].append(len(d)) 

    return total_inject_size, inj_size, total_inject_length, inj



def BVMA_NoSP_recover(Q, chosen_kws, observed_size_in_baseline, observed_length_in_baseline, real_size_after_injection, real_length_after_injection):
    real_tag = {}
    recover_tag = {}
    recover_queries_num = 0
    for query in Q:
        real_tag[query] = query
        recover_tag[query] = -1
        #findflag = False
        for kw_id in observed_size_in_baseline.keys():
            if query in real_size_after_injection.keys() and real_size_after_injection[query] - observed_size_in_baseline[kw_id] >= 0 and (real_size_after_injection[query] - observed_size_in_baseline[kw_id]) - (len(chosen_kws)/2)*math.log2(len(chosen_kws)) < len(chosen_kws):                           
                diffReBa = (int) ((real_size_after_injection[query] - observed_size_in_baseline[kw_id]))
                t_counteae = 0
                num_tF = 0
                while(diffReBa>=0):
                    diffReBa -= (int) (len(chosen_kws)/2)
                    if diffReBa<0:
                        break
                    t_counteae += 1
                    if diffReBa<len(chosen_kws):
                        re_kw_id = diffReBa
                        tmp_kw_id = re_kw_id #recovery kw_id
                        num_tF = 0 # theoretical injection length
                        while tmp_kw_id!=0:
                            if tmp_kw_id&1==1:
                                num_tF += 1
                            tmp_kw_id >>= 1
                        if t_counteae!=num_tF:
                            continue
                        if real_length_after_injection[query] - observed_length_in_baseline[kw_id] == num_tF:
                            #recover_tag[query] = re_kw_id
                            recover_tag[query] = chosen_kws[re_kw_id]

        if recover_tag[query] == real_tag[query]:
            recover_queries_num += 1
        #total_queries_num += 1
    acc = recover_queries_num / len(Q)
    return recover_tag,real_tag,acc



def BVMA_NoSP_main(wordSet, query, wordAccess, total_size):
    word_total_size = total_size[wordSet]
    s = time.time()
    total_inject_size, inj_size, total_inject_length, inj = Violin_inject(wordSet, word_total_size)
    
    query_total_size = inj_size[query] + word_total_size[query]
    queryAccess=pd.concat([wordAccess.loc[query],inj.loc[query]],axis=1)
    volumeToken = queryAccess.apply(lambda x:sum(x),axis=1) 
    volumeKeyword = wordAccess.apply(lambda x:sum(x),axis=1) 
    
    
    recover_tag, real_tag, acc = BVMA_NoSP_recover(query,wordSet,word_total_size,volumeKeyword,query_total_size,volumeToken)
    runtime = time.time() - s
    return acc, total_inject_size, total_inject_length, runtime
