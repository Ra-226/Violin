# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:26:20 2023

@author: Ra
"""


import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import attacks.Violin as Violin
import attacks.BVMA as BVMA
import attacks.Decoding as Decoding
import attacks.BVA as BVA

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
    
    color_def = sns.color_palette('Set1')
    bar = sns.barplot(data=df,x="x", y="acc",hue='attack',hue_order=["BVMA","BVA-l","BVA-h","Decoding","Violin"],
                legend=False,edgecolor='white',palette=[color_def[0],color_def[3],color_def[2],color_def[1],color_def[4]])

    plt.grid(False)
    plt.grid(zorder=0,axis='y')
    plt.xlabel('No. queries')
    plt.ylabel('Recovery rate')
    plt.tick_params()

    plt.savefig("./pic/" + 'm_enron_bar.pdf', bbox_inches = 'tight', dpi = 600)#.pdf 
    
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
    ax=plt.subplot()
    
    color_def = sns.color_palette('Set1')

    if str == "len":
        sns.barplot(data=df,x="attack", y=str,hue='attack',hue_order=["BVMA","BVA-l","BVA-h","Decoding","Violin"],
                    legend=True,palette=[color_def[0],color_def[3],color_def[2],color_def[1],color_def[4]])
        plt.legend(loc='upper left',fontsize = 16)
    else:
        sns.barplot(data=df,x="attack", y=str,hue='attack',hue_order=["BVMA","BVA-l","BVA-h","Decoding","Violin"],
                    legend=False,palette=[color_def[0],color_def[3],color_def[2],color_def[1],color_def[4]])
    

    plt.yscale('log')
    plt.grid(False)
    plt.grid(zorder=0,axis='y')
    plt.xlabel('Attacks')
    if str=="size":
        plt.ylabel('injected size')
    else:
        plt.ylabel('injected length')
    plt.tick_params()

    ax.set_xticks([0,1,2,3,4], ["BVMA","BVA-l","BVA-h","Violin","Decoding"])
    plt.xticks(rotation=30)
    plt.savefig("./pic/" + 'm_enron_bar_{}.pdf'.format(str), bbox_inches = 'tight', dpi = 600)#.pdf 
    
    plt.show()

if __name__=='__main__':
    file = "./pkl/m_enron.pkl"
    x = [0.25,0.5,0.75,1]
    if os.path.exists(file):
        with open(file,'rb') as f:
            df=pickle.load(f)
        polt_figure(df)
        df['attack'].replace("Violin","Violin",inplace = True)
        polt_figure_size(df.loc[2:6],"size")
        polt_figure_size(df.loc[2:6],"len")
        sys.exit(0) 

    with open('Datasets/enron_vol_access.pkl', 'rb') as f:
        pkl=pickle.load(f)

    df = pd.DataFrame(0.0,range(2*len(x)),['count',"x","attack","acc","len","size","time"])
    flag = 0
    delta = 3000
    word_len = 500
    run_times = 10
    
    for v_x in x:
        selectivity = int(1*(delta-word_len))
        query_len=int(word_len*v_x)
        wordSet=list(pkl[0].keys())[selectivity:selectivity+word_len]
        if v_x == 0:
            wordSet=[list(pkl[0].keys())[i] for i in np.random.permutation(3000)[:word_len]]
        for times in range(run_times):
            query=[ wordSet[i] for i in np.random.permutation(word_len)[:query_len] ]
            
            total_size = pd.Series(0,pkl[0].keys())[wordSet]
            for i in wordSet:
                total_size[i] = sum(pkl[0][i])
            
            word_size_set = {i:pkl[0][i] for i in wordSet}
            wordAccess=pkl[1].loc[wordSet]
        
        
            Violin_acc, Violin_result, Violin_total_inject_size, Violin_total_inject_length,Violin_time = Violin.Violin_main(wordSet, query, wordAccess, total_size, word_size_set)
            
            df.loc[flag] = [times, v_x, "Violin", Violin_acc,Violin_total_inject_length,Violin_total_inject_size,Violin_time]
            flag += 1
            
            
            Decoding_acc, Decoding_total_inject_size, Decoding_total_inject_length, offset, Decoding_time = Decoding.Decoding_main(wordSet, query, wordAccess, total_size)
            df.loc[flag] = [times, v_x, "Decoding", Decoding_acc,Decoding_total_inject_length,Decoding_total_inject_size,Decoding_time]
            flag += 1
            
            BVMA_acc_temp, BVMA_total_inject_size, BVMA_total_inject_length,BVMA_time = BVMA.BVMA_NoSP_main(wordSet, query, wordAccess, total_size)
            df.loc[flag] = [times, v_x, "BVMA", BVMA_acc_temp,BVMA_total_inject_length,BVMA_total_inject_size,BVMA_time]
            flag += 1
            
            gamma = len(wordSet)//2
            low_BVA_acc_temp, low_BVA_total_inject_size, low_BVA_total_inject_length,low_BVA_time = BVA.BVA_main(wordSet, query, wordAccess, total_size, gamma)
            df.loc[flag] = [times, v_x, "BVA-l", low_BVA_acc_temp,low_BVA_total_inject_length,low_BVA_total_inject_size,low_BVA_time]
            flag += 1
            
            gamma = offset // 4
            high_BVA_acc_temp, high_BVA_total_inject_size, high_BVA_total_inject_length,high_BVA_time = BVA.BVA_main(wordSet, query, wordAccess, total_size, gamma)
            df.loc[flag] = [times, v_x, "BVA-h", high_BVA_acc_temp,high_BVA_total_inject_length,high_BVA_total_inject_size,high_BVA_time]
            flag += 1
            
        
    
    
    with open("./pkl/m_enron_4.pkl","wb") as f:
        pickle.dump(df,f)
    
    polt_figure(df)


