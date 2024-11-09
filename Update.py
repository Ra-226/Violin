# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 20:04:38 2024

@author: Ra
"""


import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import attacks_Update.Violin as Violin
import attacks_Update.BVMA as BVMA
import attacks_Update.BVA as BVA

import os
import sys


def polt_figure(df):
    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": True,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size":20})
    plt.figure(figsize=(10, 6))
    #ax=plt.subplot()
    fig, ax = plt.subplots(figsize=(8, 4)) 
    color_def = sns.color_palette('Set1')
    sns.barplot(data=df,x="Percentage", y="acc",hue='attack',hue_order=["BVMA","BVA","Violin"],legend=True, palette=[color_def[0],color_def[3],color_def[4]])
    plt.legend(loc='upper right',ncol=3)
    ax.set_xticks([0,1,2,3,4,5,6], [20,50,100,200,500,1000,2000])
    ax.set_yticks([0,0.25,0.5,0.75,1,1.3], [0,0.25,0.5,0.75,1,None])
    plt.grid(False)
    plt.grid(zorder=0,axis='y')
    plt.xlabel('Update Percentage (\%)')
    plt.ylabel('Recovery rate')
    plt.tick_params()
    plt.savefig("./pic/" + 'Update.pdf', bbox_inches = 'tight', dpi = 600)#.pdf
    plt.show()

if __name__=='__main__':
    file = "./pkl/Update.pkl"
    Update = [0.1, 0.2, 0.5, 1, 2, 5]
    Update = [20, 10, 5, 2, 1, 0.5, 0.2, 0.1]
    if os.path.exists(file):
        with open(file,'rb') as f:
            df=pickle.load(f)
        Overhead = pd.DataFrame(columns = ["attack", "Percentage", "type", "Overhead"])
        polt_figure(df)
        sys.exit(0) 

    with open('Datasets/enron_half_access.pkl', 'rb') as f:
        pkl_doc=pickle.load(f)
    doc_size = pd.Series(pkl_doc['doc_size'])
    min_file_size, max_file_size, avg_file_size = doc_size.min(),doc_size.max(),doc_size.mean() 
    if min_file_size == 0: min_file_size = 1
    
    queryRate = 0.25

    df = pd.DataFrame(0.0,range(2*len(Update)),['count',"Percentage","attack","acc","len","size","time"])
    flag = 0
    delta = 3000
    word_len = 1000
    run_times = 10
    
    update_dataset, client_dataset = pkl_doc["update_dataset"], pkl_doc["client_dataset"]
    doc = pkl_doc["doc"]
    
    selectivity = int(1*(delta-word_len))
    query_len=int(word_len*queryRate)
    
    kws = pkl_doc['word_size_set'].keys()
    
    wordSet=list(kws)[selectivity : selectivity + word_len]
    word_size_set = {i : pkl_doc['word_size_set'][i] for i in wordSet}
    total_size = pd.Series(0, kws)[wordSet]
    real_length = pd.Series(0,word_size_set.keys())
    for i in wordSet:
        total_size[i] = sum(word_size_set[i])
        real_length[i] = len(word_size_set[i])

    for v_x in Update:
        for times in range(run_times):
            query=[ wordSet[i] for i in np.random.permutation(word_len)[:query_len] ]
            percentage = v_x

            Violin_acc, Violin_result, Violin_total_inject_size, Violin_total_inject_length,Violin_time = Violin.Violin_set_main(
                wordSet, query, total_size, real_length, update_dataset, client_dataset, 
                min_file_size, max_file_size, word_size_set, percentage, doc)
            
            df.loc[flag] = [times, v_x, "Violin", Violin_acc,Violin_total_inject_length,Violin_total_inject_size,Violin_time]
            print(df.loc[flag])
            flag += 1

            BVMA_acc, BVMA_total_inject_size, BVMA_total_inject_length,BVMA_time = BVMA.BVMA_NoSP_main(
                wordSet, query, total_size, real_length, update_dataset, client_dataset, 
                min_file_size, max_file_size, word_size_set, percentage, doc)
            df.loc[flag] = [times, v_x, "BVMA", BVMA_acc,BVMA_total_inject_length,BVMA_total_inject_size,BVMA_time]
            print(df.loc[flag])
            flag += 1
            
            gamma = (int) (avg_file_size*np.ceil(3000/(2*avg_file_size)))
            
            BVA_acc, BVA_total_inject_size, BVA_total_inject_length,BVA_time = BVA.BVA_main(
                wordSet, query, total_size, real_length, update_dataset, client_dataset, 
                min_file_size, max_file_size, word_size_set, percentage, doc, gamma)
            df.loc[flag] = [times, v_x, "BVA", BVA_acc,BVA_total_inject_length,BVA_total_inject_size,BVA_time]
            flag += 1

    with open("./pkl/Update.pkl","wb") as f:
        pickle.dump(df,f)
    
    polt_figure(df)
    