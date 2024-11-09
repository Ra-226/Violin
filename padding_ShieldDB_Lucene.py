# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 21:03:28 2024

@author: Ra
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import attacks_ShieldDB.Violin as Violin
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
        "font.size": 20})
    plt.figure()
    ax = plt.subplot()
    color_def = sns.color_palette('Set1')
    sns.barplot(data=df, x="ShieldAlpha", y="acc", hue='attack', hue_order=["BVMA", "BVA", "Violin"], legend=True,
                palette=[color_def[0], color_def[3], color_def[4]])

    plt.legend(loc='upper right', fontsize=16)
    ax.set_xticks([0, 1, 2, 3, 4], [2, 8, 32, 128, 256])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1])
    plt.grid(False)
    plt.grid(zorder=0, axis='y')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Recovery rate')
    plt.tick_params()

    plt.savefig("./pic/" + 'padding_ShileldDB_Lucene.pdf', bbox_inches='tight', dpi=600)  # .pdf
    plt.show()


def polt_figure_size(df):
    plt.rcParams.update({
        "legend.fancybox": False,
        "legend.frameon": True,
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": 20})
    plt.figure()
    ax = plt.subplot()
    color_def = sns.color_palette('RdYlGn')
    sns.barplot(data=df, x="ShieldAlpha", y="Overhead", hue="type", legend=True, palette=[color_def[0], color_def[1]])

    plt.legend(loc='upper left', fontsize=16)
    ax.set_xticks([0, 1, 2, 3, 4], [2, 8, 32, 128, 256])
    plt.grid(False)
    plt.grid(zorder=0, axis='y')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Overhead (\%)')
    plt.tick_params()
    plt.savefig("./pic/" + 'padding_ShileldDB_{}_Lucene.pdf'.format("overhead"), bbox_inches='tight', dpi=600)  # .pdf

    plt.show()


if __name__ == '__main__':
    file = "./pkl/padding_ShileldDB_Lucene.pkl"
    ShieldAlpha = [256, 128, 32, 8, 2]
    if os.path.exists(file):
        with open(file, 'rb') as f:
            df = pickle.load(f)
        Overhead = pd.DataFrame(columns=["attack", "ShieldAlpha", "type", "Overhead"])
        cnt = 0
        for i in df.index:
            Overhead.loc[cnt] = [df.loc[i]["attack"], df.loc[i]["ShieldAlpha"], "Setup\&Fill",
                                 (df.loc[i]["setup_Overhead"] - 1) * 100]
            Overhead.loc[cnt + 1] = [df.loc[i]["attack"], df.loc[i]["ShieldAlpha"], "Inj\&Fill",
                                     (df.loc[i]["inj_Overhead"] - 1) * 100]
            cnt += 2
        polt_figure(df)
        polt_figure_size(Overhead)
        sys.exit(0)

    with open('./Datasets/Lucene_vol_access.pkl', 'rb') as f:
        pkl = pickle.load(f)
    with open('Datasets/doc_size_Lucene.pkl', 'rb') as f:
        pkl_doc = pickle.load(f)
    doc_size = pd.Series(pkl_doc['doc_size'])
    min_file_size, max_file_size, avg_file_size = doc_size.min(), doc_size.max(), doc_size.mean()
    if min_file_size == 0: min_file_size = 1

    queryRate = 0.25
    df = pd.DataFrame(0.0, range(2 * len(ShieldAlpha)),
                      ['count', "ShieldAlpha", "attack", "acc", "len", "size", "time", "setup_Overhead",
                       "inj_Overhead"])
    flag = 0
    delta = 3000
    word_len = 1000
    run_times = 10

    for v_x in ShieldAlpha:
        selectivity = int(1 * (delta - word_len))
        query_len = int(word_len * queryRate)
        wordSet = list(pkl[0].keys())[selectivity:selectivity + word_len]

        for times in range(run_times):
            query = [wordSet[i] for i in np.random.permutation(word_len)[:query_len]]

            total_size = pd.Series(0, pkl[0].keys())[wordSet]
            for i in wordSet:
                total_size[i] = sum(pkl[0][i])

            word_size_set = {i: pkl[0][i] for i in wordSet}
            real_length = pd.Series(0, word_size_set.keys())
            for i in wordSet:
                real_length[i] = len(word_size_set[i])

            wordAccess = pkl[1].loc[wordSet]
            Alpha = v_x

            group = Violin.Group_cluster(Alpha)

            Violin_acc, Violin_result, Violin_total_inject_size, Violin_total_inject_length, Violin_time, setup_Overhead, inj_Overhead = Violin.Violin_set_main(
                wordSet, list(pkl[0].keys()), query, total_size, real_length, min_file_size, max_file_size,
                word_size_set, group)

            df.loc[flag] = [times, v_x, "Violin", Violin_acc, Violin_total_inject_length, Violin_total_inject_size,
                            Violin_time, setup_Overhead, inj_Overhead]
            flag += 1

            BVMA_acc, BVMA_total_inject_size, BVMA_total_inject_length, BVMA_time, setup_Overhead, inj_Overhead = BVMA.BVMA_NoSP_main(
                wordSet, list(pkl[0].keys()), query, total_size, real_length, min_file_size, max_file_size, group)
            df.loc[flag] = [times, v_x, "BVMA", BVMA_acc, BVMA_total_inject_length, BVMA_total_inject_size, BVMA_time,
                            setup_Overhead, inj_Overhead]
            flag += 1

            gamma = (int)(avg_file_size * np.ceil(3000 / (2 * avg_file_size)))

            BVA_acc, BVA_total_inject_size, BVA_total_inject_length, BVA_time, setup_Overhead, inj_Overhead = BVA.BVA_main(
                wordSet, list(pkl[0].keys()), query, total_size, real_length, gamma, min_file_size, max_file_size,
                group)
            df.loc[flag] = [times, v_x, "BVA", BVA_acc, BVA_total_inject_length, BVA_total_inject_size, BVA_time,
                            setup_Overhead, inj_Overhead]
            flag += 1

    with open("./pkl/padding_ShileldDB_Lucene.pkl", "wb") as f:
        pickle.dump(df, f)

    polt_figure(df)
