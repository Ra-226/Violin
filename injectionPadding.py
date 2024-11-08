# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:40:26 2024

@author: Ra
"""


import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import attacks_ShieldDB.Violin_large_padding as Violin
import attacks_ShieldDB.BVMA as BVMA
import attacks_ShieldDB.BVA as BVA

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
    n = 25
    fig, ax = plt.subplots(figsize=(8, 4)) 
    color_def = sns.color_palette('Oranges', 12)
    sns.barplot(data=df,x="r", y="acc",legend=False, hue='r',palette=color_def[1:])

    ax.set_xticks([0,1,2,3,4,5,6,7], [1, 3, 5, 7, 9, 11, 13, 15], fontsize = n)
    ax.set_yticks([0,0.25,0.5,0.75,1], [0,0.25,0.5,0.75,1], fontsize = n)
    plt.grid(False)
    plt.grid(zorder=0,axis='y')
    plt.xlabel(r'$\mu$', fontsize = n)
    plt.ylabel('Recovery rate',fontsize = n)
    plt.tick_params()

    plt.savefig("./pic/" + 'padding_ShileldDB_inj_pad.pdf', bbox_inches = 'tight', dpi = 600)#.pdf
    plt.show()
    
def polt_figure_size(df):
    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": True,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],  
    "font.size":20})
    plt.figure()
    ax=plt.subplot()
    color_def = sns.color_palette('Paired')
    sns.barplot(data=df,x="ShieldAlpha", y = "Overhead", hue="type", legend=True, palette = [color_def[0], color_def[1]])
    
    plt.legend(loc='upper left',fontsize = 16)
    ax.set_xticks([0,1,2,3,4], [2,8,32,128,256])
    plt.grid(False)
    plt.grid(zorder=0,axis='y')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Overhead (\%)')

    plt.tick_params()
    plt.savefig("./pic/" + 'padding_ShileldDB_inj_pad_{}.pdf'.format("overhead"), bbox_inches = 'tight', dpi = 600)#.pdf 
    
    plt.show()

if __name__=='__main__':
    file = "./pkl/padding_ShileldDB_inj_pad.pkl"
    r = [1, 3, 5, 7, 9, 11, 13, 15] 
    if os.path.exists(file):
        with open(file,'rb') as f:
            df=pickle.load(f)
        polt_figure(df)
        sys.exit(0) 
    

    with open('Datasets/enron_vol_access.pkl', 'rb') as f:
        pkl=pickle.load(f)
    with open('../pickles/access_enron.pkl','rb') as f:
        pkl_doc=pickle.load(f)
    doc_size = pd.Series(pkl_doc['doc_size'])
    min_file_size, max_file_size, avg_file_size = doc_size.min(),doc_size.max(),doc_size.mean() 
    if min_file_size == 0: min_file_size = 1
    
    queryRate = 0.25
    df = pd.DataFrame(0.0,range(2*len(r)),['count', "r", "ShieldAlpha","attack","acc","len","size","time", "setup_Overhead", "inj_Overhead"])
    flag = 0
    delta = 3000
    word_len = 1000
    run_times = 20
    
    for v_x in r:
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
            Alpha = 256
            
            group = Violin.Group_cluster(Alpha)
            
            Violin_acc, Violin_result, Violin_total_inject_size, Violin_total_inject_length,Violin_time, setup_Overhead, inj_Overhead = Violin.Violin_set_main(
                wordSet, list(pkl[0].keys()), query, total_size, real_length, min_file_size, max_file_size, word_size_set, group, v_x)
            
            df.loc[flag] = [times, str(int(v_x)), Alpha, "Violin", Violin_acc,Violin_total_inject_length,Violin_total_inject_size,Violin_time,setup_Overhead, inj_Overhead]
            flag += 1

    
    
    with open("./pkl/padding_ShileldDB_inj_pad.pkl","wb") as f:
        pickle.dump(df,f)
    
    polt_figure(df)
    