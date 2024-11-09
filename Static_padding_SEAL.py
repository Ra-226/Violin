# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 10:00:49 2023

@author: Ra
"""




import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import attacks_seal.Violin as Violin
import attacks_seal.BVMA as BVMA
import attacks_seal.Decoding as Decoding
import attacks_seal.BVA as BVA

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
    plt.figure()
    ax=plt.subplot()
    color_def = sns.color_palette('Set1')
    bar = sns.barplot(data=df,x="x", y="acc",hue='attack',hue_order=["BVMA","BVA","Violin"],
                legend=False,edgecolor='white',palette=[color_def[0],color_def[3],color_def[4]])
    
    
    ax.set_xticks([0,1,2,3], ["No defense","$y=2$","$y=4$","$y=16$"])
    plt.grid(False)
    plt.grid(zorder=0,axis='y')
    plt.xlabel('Padding')
    plt.ylabel('Recovery rate')
    plt.tick_params()
    plt.savefig("./pic/" + 'Static_padding_SEAL.pdf', bbox_inches = 'tight', dpi = 600)#.pdf
    plt.show()
    
def polt_figure_size(df,str):
    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": True,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],  
    "font.size":20})
    plt.figure()
    
    color_def = sns.color_palette('Set1')
    if str == "len":
        sns.barplot(data=df,x="attack", y=str,hue='attack',hue_order=["BVMA","BVA","Violin"],
                    legend=True,palette=[color_def[0],color_def[3],color_def[4]])
        plt.legend(loc='upper right',fontsize = 16)
    else:
        sns.barplot(data=df,x="attack", y=str,hue='attack',hue_order=["BVMA","BVA","Violin"],
                    legend=False,palette=[color_def[0],color_def[3],color_def[4]])
    plt.yscale('log')
    plt.grid(False)
    plt.grid(zorder=0,axis='y')
    plt.xlabel('Attacks')
    if str=="size":
        plt.ylabel('injected size')
    else:
        plt.ylabel('injected length')
    plt.tick_params()
    plt.savefig("./pic/" + 'Static_padding_SEAL_{}.pdf'.format(str), bbox_inches = 'tight', dpi = 600)#.pdf 
    
    plt.show()

if __name__=='__main__':
    file = "./pkl/Static_padding_SEAL.pkl"
    x = [0,2,4,16]
    if os.path.exists(file):
        with open(file,'rb') as f:
            df=pickle.load(f)
        polt_figure(df)
        polt_figure_size(df.loc[1:4],"size")
        polt_figure_size(df.loc[1:4],"len")
        sys.exit(0) 
    

    with open('Datasets/enron_vol_access.pkl', 'rb') as f:
        pkl=pickle.load(f)
    with open('Datasets/doc_size_enron.pkl', 'rb') as f:
        pkl_doc=pickle.load(f)
    doc_size = pd.Series(pkl_doc['doc_size'])
    min_file_size, max_file_size, avg_file_size = doc_size.min(),doc_size.max(),doc_size.mean() 
    if min_file_size == 0: min_file_size = 1
    
    queryRate = 0.25

    df = pd.DataFrame(0.0,range(2*len(x)),['count',"x","attack","acc","len","size","time"])
    flag = 0
    
    delta = 3000
    
    word_len = 1000

    run_times = 10
    
    for v_x in x:
        selectivity = int(1*(delta-word_len))
        query_len=int(word_len*queryRate)
        
        wordSet=list(pkl[0].keys())[selectivity:selectivity+word_len]
        
        for times in range(run_times):
            query=[ wordSet[i] for i in np.random.permutation(word_len)[:query_len] ]
            
            total_size = pd.Series(0,pkl[0].keys())[wordSet]
            for i in wordSet:
                total_size[i] = sum(pkl[0][i])
            
            word_size_set = {i:pkl[0][i] for i in wordSet}
            real_length = pd.Series(0,word_size_set.keys())
            for i in wordSet:
                real_length[i] = len(word_size_set[i])
            
            wordAccess=pkl[1].loc[wordSet]
            seal_x = v_x

            Violin_acc, Violin_result, Violin_total_inject_size, Violin_total_inject_length,Violin_time = Violin.Violin_set_main(wordSet, query, wordAccess, total_size, word_size_set,real_length, seal_x, min_file_size, max_file_size)
            df.loc[flag] = [times, v_x, "Violin", Violin_acc,Violin_total_inject_length,Violin_total_inject_size,Violin_time]
            flag += 1
            
            BVMA_acc, BVMA_total_inject_size, BVMA_total_inject_length,BVMA_time = BVMA.BVMA_NoSP_main(wordSet, query, wordAccess, total_size, real_length, seal_x, min_file_size, max_file_size)
            df.loc[flag] = [times, v_x, "BVMA", BVMA_acc,BVMA_total_inject_length,BVMA_total_inject_size,BVMA_time]
            flag += 1
            
            gamma = (int) (avg_file_size*np.ceil(3000/(2*avg_file_size)))
            
            BVA_acc, BVA_total_inject_size, BVA_total_inject_length,BVA_time = BVA.BVA_main(wordSet, query, wordAccess, total_size, real_length, seal_x, gamma, min_file_size, max_file_size)
            df.loc[flag] = [times, v_x, "BVA", BVA_acc,BVA_total_inject_length,BVA_total_inject_size,BVA_time]
            flag += 1
    
    with open("./pkl/Static_padding_SEAL.pkl","wb") as f:
        pickle.dump(df,f)
    
    polt_figure(df)
    