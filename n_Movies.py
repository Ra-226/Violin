# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:33:54 2023

@author: Ra
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

import attacks.Violin as Violin
import attacks.BVMA as BVMA
import attacks.Decoding as Decoding
import attacks.BVA as BVA

import os
import sys


def polt_figure(x, y1, y2, y3, y4, y5):
    plt.rcParams.update({
        "legend.fancybox": False,
        "legend.frameon": True,
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": 20})
    plt.figure()
    ax = plt.subplot()

    y3_avg = np.mean(y3, axis=1)
    y3_min = y3_avg - np.min(y3, axis=1)
    y3_max = np.max(y3, axis=1) - y3_avg

    y4_avg = np.mean(y4, axis=1)
    y4_min = y4_avg - np.min(y4, axis=1)
    y4_max = np.max(y4, axis=1) - y4_avg

    y5_avg = np.mean(y5, axis=1)
    y5_min = y5_avg - np.min(y5, axis=1)
    y5_max = np.max(y5, axis=1) - y5_avg
    color_def = sns.color_palette('Set1')
    ax.errorbar(x, y3_avg, color=color_def[0], marker='o', yerr=[y3_min, y3_max],
                markersize=12, markeredgewidth=0.8,
                linewidth=1.5, capsize=4, label='BVMA')
    ax.errorbar(x, y4_avg, color=color_def[3], marker='v', yerr=[y4_min, y4_max],
                markersize=12, markeredgewidth=.8,
                linewidth=1.5, capsize=4, label='BVA-l')
    ax.errorbar(x, y5_avg, color=color_def[2], marker='^', yerr=[y5_min, y5_max],
                markersize=12, markeredgewidth=.8,
                linewidth=1.5, capsize=4, label='BVA-h')
    ax.errorbar(x, y2, color=color_def[1], marker='s',
                markersize=12, markeredgewidth=.8,
                linewidth=1.5, capsize=4, label='Decoding')

    ax.errorbar(x, y1, color=color_def[4], marker='d',
                markersize=12, markeredgewidth=.8,
                linewidth=1.5, capsize=4, label='Violin')

    y_major_locator = MultipleLocator(0.2)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xticks([500, 1000, 1500, 2000, 2500], [500, 1000, 1500, 2000, 2500])
    plt.grid()
    plt.xlabel('Keyword sapce')
    plt.ylabel('Recovery rate')
    plt.tick_params()
    plt.legend(fontsize=16)

    plt.savefig("./pic/" + 'n_Movies.pdf', bbox_inches='tight', dpi=600)  # .pdf
    plt.show()


def plot_size_and_time(x, y, str):
    plt.figure()
    ax = plt.subplot()

    color_def = sns.color_palette('Set1')
    plt.plot(x, y[1], linestyle='-', color=color_def[0], marker='o',
             markersize=12, markeredgewidth=0.8, linewidth=1.5, label='BVMA')
    plt.plot(x, y[2], linestyle='-', color=color_def[3], marker='v',
             markersize=12, markeredgewidth=0.8, linewidth=1.5, label='BVA-l')
    plt.plot(x, y[3], linestyle='-', color=color_def[2], marker='^',
             markersize=12, markeredgewidth=0.8, linewidth=1.5, label='BVA-h')
    plt.plot(x, y[4], linestyle='-', color=color_def[1], marker='s',
             markersize=12, markeredgewidth=0.8, linewidth=1.5, label='Decoding')
    plt.plot(x, y[0], linestyle='-', color=color_def[4], marker='d',
             markersize=12, markeredgewidth=0.8, linewidth=1.5, label='Violin')

    ax.set_xticks([500, 1000, 1500, 2000, 2500], [500, 1000, 1500, 2000, 2500])
    plt.yscale('log')
    plt.xlabel('Keyword sapce')
    if str == 'size':
        plt.ylabel('Injection size')
    else:
        plt.ylabel('Runing time(s)')
    plt.grid()
    plt.legend(fontsize=16)
    plt.savefig("./pic/" + 'n_{}_Movies.pdf'.format(str), bbox_inches='tight', dpi=600)


if __name__ == '__main__':
    file = "./pkl/n_Movies.pkl"
    x = [500, 1000, 1500, 2000, 2500]
    if os.path.exists(file):
        with open(file, 'rb') as f:
            pkl = pickle.load(f)
        acc = pkl["acc"]
        length = pkl["length"]
        size = pkl['size']
        time = pkl['time']
        polt_figure(x, acc[0], acc[4], acc[1], acc[2], acc[3])
        plot_size_and_time(x, pkl['size'], "size")
        plot_size_and_time(x, pkl['time'], "time")
        sys.exit(0)

    with open('./Datasets/Movies_vol_access.pkl', 'rb') as f:
        pkl = pickle.load(f)

    queryRate = 0.5
    df = pd.DataFrame(0.0, range(2 * len(x)), ["x", "dataset", "unique_size_num"])
    delta = 3000
    i_x = 0
    word_len = 500

    acc_Violin = []
    acc_BVMA = []
    acc_BVA_low = []
    acc_BVA_high = []
    acc_Decoding = []

    length_Violin = []
    length_BVMA = []
    length_BVA_low = []
    length_BVA_high = []
    length_Decoding = []

    size_Violin = []
    size_BVMA = []
    size_BVA_low = []
    size_BVA_high = []
    size_Decoding = []

    time_Violin = []
    time_BVMA = []
    time_BVA_low = []
    time_BVA_high = []
    time_Decoding = []

    run_times = 10

    for v_x in x:
        word_len = v_x
        query_len = int(word_len * queryRate)
        selectivity = int(1 * (delta - word_len))
        wordSet = list(pkl[0].keys())[selectivity:selectivity + word_len]

        Violin_acc = []
        Decoding_acc = []
        BVMA_acc = []
        low_BVA_acc = []
        high_BVA_acc = []

        for times in range(run_times):
            query = [wordSet[i] for i in np.random.permutation(word_len)[:query_len]]
            total_size = pd.Series(0, pkl[0].keys())[wordSet]
            for i in wordSet:
                total_size[i] = sum(pkl[0][i])

            word_size_set = {i: pkl[0][i] for i in wordSet}
            wordAccess = pkl[1].loc[wordSet]

            Violin_acc_temp, Violin_result, Violin_total_inject_size, Violin_total_inject_length, Violin_time = Violin.Violin_set_main(
                wordSet, query, wordAccess, total_size, word_size_set)
            Decoding_acc_temp, Decoding_total_inject_size, Decoding_total_inject_length, offset, Decoding_time = Decoding.Decoding_main(
                wordSet, query, wordAccess, total_size)

            BVMA_acc_temp, BVMA_total_inject_size, BVMA_total_inject_length, BVMA_time = BVMA.BVMA_NoSP_main(wordSet,
                                                                                                             query,
                                                                                                             wordAccess,
                                                                                                             total_size)

            gamma = len(wordSet) // 2
            low_BVA_acc_temp, low_BVA_total_inject_size, low_BVA_total_inject_length, low_BVA_time = BVA.BVA_main(
                wordSet, query, wordAccess, total_size, gamma)

            gamma = offset // 4
            high_BVA_acc_temp, high_BVA_total_inject_size, high_BVA_total_inject_length, high_BVA_time = BVA.BVA_main(
                wordSet, query, wordAccess, total_size, gamma)

            Violin_acc.append(Violin_acc_temp)
            Decoding_acc.append(Decoding_acc_temp)
            BVMA_acc.append(BVMA_acc_temp)
            low_BVA_acc.append(low_BVA_acc_temp)
            high_BVA_acc.append(high_BVA_acc_temp)

        acc_Violin.append(Violin_acc)
        acc_BVMA.append(BVMA_acc)
        acc_BVA_low.append(low_BVA_acc)
        acc_BVA_high.append(high_BVA_acc)
        acc_Decoding.append(Decoding_acc)

        length_Violin.append(Violin_total_inject_length)
        length_BVMA.append(BVMA_total_inject_length)
        length_BVA_low.append(low_BVA_total_inject_length)
        length_BVA_high.append(high_BVA_total_inject_length)
        length_Decoding.append(Decoding_total_inject_length)

        size_Violin.append(Violin_total_inject_size)
        size_BVMA.append(BVMA_total_inject_size)
        size_BVA_low.append(low_BVA_total_inject_size)
        size_BVA_high.append(high_BVA_total_inject_size)
        size_Decoding.append(Decoding_total_inject_size)

        time_Violin.append(Violin_time)
        time_BVMA.append(BVMA_time)
        time_BVA_low.append(low_BVA_time)
        time_BVA_high.append(high_BVA_time)
        time_Decoding.append(Decoding_time)

    D = {"acc": [acc_Violin, acc_BVMA, acc_BVA_low, acc_BVA_high, acc_Decoding],
         "length": [length_Violin, length_BVMA, length_BVA_low, length_BVA_high, length_Decoding],
         "size": [size_Violin, size_BVMA, size_BVA_low, size_BVA_high, size_Decoding],
         "time": [time_Violin, time_BVMA, time_BVA_low, time_BVA_high, time_Decoding]}
    with open("./pkl/n_Movies.pkl", "wb") as f:
        pickle.dump(D, f)

    polt_figure(x, acc_Violin, acc_Decoding, acc_BVMA, acc_BVA_low, acc_BVA_high)
