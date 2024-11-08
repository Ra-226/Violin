# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:04:56 2024

@author: Ra
"""


import math
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import os,sys

import attacks.Decoding as Decoding


def sumComb(n, m):
    sum = 0
    for i in range(m + 1):
        sum += math.comb(n, i)
    return sum

def getLengthAndx(n, t):
    minL = math.ceil(math.log(n, 2)) + 1
    flag = 0
    while(flag == 0):
        for x in range(1, minL):
            if (math.comb(minL - 1, x) >= n + 2 * minL - 2 * t) and (sumComb(minL - 1, x - 1) <= t - minL):
                flag = 1
                break
        minL += 1
    print("l : " + str(minL - 1) + "   x: " + str(x + 1))
    minL -= 1
    for x in range(math.ceil(math.log(n, 2)), 0, -1):
        if sumComb(math.ceil(math.log(n, 2)) - 1, x - 1) <= t:
            break
        
    return minL, x + 1

class Threshold: 
    
    def __init__(self, chosen_kws, T, gamma, offset_of_Decoding):
        self.chosen_kws = chosen_kws


        self.total_inject_length = 0
        self.total_inject_size = 0
        self.T = T
        self.gamma = gamma
        self.offset = offset_of_Decoding 
       
    def Decoding_inject(self):
        self.total_inject_length = 0
        self.total_inject_size = 0
        for kw_id in range(len(self.chosen_kws)):
            self.total_inject_length += math.ceil(kw_id*self.offset/self.T)
        self.total_inject_size = self.total_inject_length*self.T

    def BVA_inject(self):   
        self.total_inject_length = 0
        self.total_inject_size = 0
        kws_each_doc = math.ceil(len(self.chosen_kws)/2)
        if kws_each_doc==0:
            num_injection_doc=0
        else:
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
        """
        generate injected doc
        """
        size_each_doc = []
        if num_injection_doc >= 1:
            size_each_doc.append(self.gamma)
        if num_injection_doc >= 2:
            for i in range(1, num_injection_doc):
                size_each_doc.append(size_each_doc[i-1] + size_each_doc[i-1])

        for s in size_each_doc:
            if self.T>=s:
                self.total_inject_length += 1
            else:
                last_part_size = 0 #Ref_F2
                if s%self.T!=0:
                    last_part_size = math.ceil(self.T/(s%self.T))
                self.total_inject_length += math.ceil(len(self.chosen_kws)*0.5/self.T)*(math.floor(s/self.T)+last_part_size)    
        # 分割重填前后size不变
        F = []
        for i in range(num_injection_doc):
            f=[]
            for j in range(len(self.chosen_kws)):
                if bin(j)[2:].rjust(32,'0')[-(i+1)] == '1':
                    f.append(self.chosen_kws[j])
            for pad in range( (2**i)*self.gamma - len(f) ):
                f.append('not_in_chosen_kws')
            F.append(f)
            self.total_inject_size += len(f)
        
        
    def BVMA_inject(self):
        
        self.total_inject_length = 0
        self.total_inject_size = 0

        kws_each_doc = math.ceil(len(self.chosen_kws)/2)
        if kws_each_doc==0:
            num_injection_doc=0
        else:
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
        """
        generate doc.
        """
        size_each_doc = [] 
        if num_injection_doc >= 1:
            size_each_doc.append(1 + (int) (len(self.chosen_kws)/2))
        if num_injection_doc >= 2:
            for i in range(1, num_injection_doc):
                size_each_doc.append(size_each_doc[i-1] + size_each_doc[i-1] - (int) (len(self.chosen_kws)/2))

        for s in size_each_doc:
            if self.T>=s:
                self.total_inject_length += 1
            else:
                last_part_size = 0
                if s%self.T!=0:
                    last_part_size = math.ceil(self.T/(s%self.T))
                self.total_inject_length += math.ceil(len(self.chosen_kws)*0.5/self.T)*(math.floor(s/self.T)+last_part_size) 
        
        F = []
        for i in range(num_injection_doc):
            f=[]
            for j in range(len(self.chosen_kws)):
                if bin(j)[2:].rjust(32,'0')[-(i+1)] == '1':
                    f.append(self.chosen_kws[j])
            for pad in range(int(word_len/2) - len(f) + 2**i):
                f.append('not_in_chosen_kws')
            F.append(f)
            self.total_inject_size += len(f)

        
    def ZKP_inject(self):
        self.total_inject_length = 0
        self.total_inject_size = 0
        number_kws = len(self.chosen_kws)
        if self.T>=number_kws/2:
            self.total_inject_length = math.ceil(math.log2(number_kws))
        else:
            self.total_inject_length = math.ceil(number_kws*0.5/self.T)*(math.ceil(math.log2(2*self.T))+1)-1
        self.total_inject_size = self.total_inject_length*self.T
    
    
    def Violin_inject(self):
        self.total_inject_length = 0
        self.total_inject_size = 0
        number_kws = len(self.chosen_kws)
        self.total_inject_length, x = getLengthAndx(number_kws, self.T)
        self.total_inject_size = (self.T - self.total_inject_length) * self.total_inject_length + (self.total_inject_length - 1) * self.total_inject_length / 2

        
if __name__=='__main__':
    T = [200,400,600,800,1000,1200,1400]
    with open('./Datasets/enron_vol_access_2.pkl','rb') as f:
        pkl=pickle.load(f)

    queryRate = 0.5
    word_len = 3000
    wordSet=list(pkl[0].keys())

    offset = 117244  #Enron的3000个关键字计算得，计算的很慢
    gamma_of_BVA_low = len(wordSet)//2
    gamma_of_BVA_high = offset//4
    
    Decoding = []
    BVA_l = []
    BVA_h = []
    BVMA = []
    Violin = []
    ZKP = []
    
    Decoding_s = []
    BVA_l_s = []
    BVA_h_s = []
    BVMA_s = []
    Violin_s = []
    ZKP_s = []
    
    for t in tqdm.tqdm(iterable=T,desc="……"):
        attack1 = Threshold(wordSet, t, gamma_of_BVA_low, offset)
        attack2 = Threshold(wordSet, t, gamma_of_BVA_high, offset)
        
        attack1.Decoding_inject()
        Decoding.append(attack1.total_inject_length)
        Decoding_s.append(attack1.total_inject_size)
        attack1.BVA_inject()
        BVA_l.append(attack1.total_inject_length)
        BVA_l_s.append(attack1.total_inject_size)
        attack2.BVA_inject()
        BVA_h.append(attack2.total_inject_length)
        BVA_h_s.append(attack2.total_inject_size)
        attack1.BVMA_inject()
        BVMA.append(attack1.total_inject_length)
        BVMA_s.append(attack1.total_inject_size)
        attack1.ZKP_inject()
        ZKP.append(attack1.total_inject_length)
        ZKP_s.append(attack1.total_inject_size)
        attack1.Violin_inject()
        Violin.append(attack1.total_inject_length)
        Violin_s.append(attack1.total_inject_size)
    
    D = {
         "length":[BVMA,BVA_l,BVA_h,Violin,Decoding,ZKP],
         "size":[BVMA_s,BVA_l_s,BVA_h_s,Violin_s,Decoding_s,ZKP_s],
         }
    with open("./pkl/T_enron.pkl","wb") as f:
        pickle.dump(D,f)


    # plt.figure()
    # ax=plt.subplot()
    # plt.rcParams.update({
    # "legend.fancybox": False,
    # "legend.frameon": True,
    # "text.usetex": True,
    # "font.family": "serif",
    # "font.serif": ["Times"],
    # "font.size":20})
    #
    # color_def = sns.color_palette('Set1')
    #
    #
    # plt.plot(T, BVMA, linestyle = '-', color = color_def[0], marker = 'o',
    #           markersize = 12, linewidth=1.5, markeredgewidth=0.8, label = 'BVMA')
    # plt.plot(T, BVA_l, linestyle = '-', color = color_def[3], marker = 'v',
    #          markersize = 12, linewidth=1.5, markeredgewidth=0.8,  label = 'BVA-l')
    # plt.plot(T, BVA_h, linestyle = '-', color = color_def[2], marker = '^',
    #          markersize = 12, linewidth=1.5, markeredgewidth=0.8,  label = 'BVA-h')
    #
    # plt.plot(T, Decoding, linestyle = '-', color = color_def[1], marker = 's',
    #          markersize = 12, linewidth=1.5, markeredgewidth=0.8, label = 'Decoding')
    # plt.plot(T, Violin, linestyle = '-', color = color_def[4], marker = 'd',
    #          markersize = 12, linewidth=1.5, markeredgewidth=0.8,  label = 'Violin')
    # plt.plot(T, ZKP, linestyle = '-', color = 'cyan', marker = 'X',
    #          markersize = 12, linewidth=1.5, markeredgewidth=0.8, label = 'FIA')
    #
    # plt.yscale('log')
    # ax.set_xticks([200,600,1000,1400], [200,600,1000,1400])
    # plt.xlabel("Threshold $\\tau$")
    # plt.ylabel("injected length")
    # plt.grid()
    # plt.legend(fontsize = 14,ncol=2)
    # plt.tight_layout()
    # plt.savefig("./pic/" + 'T_length_enron2.pdf', bbox_inches = 'tight', dpi = 600)
    # plt.show()
    #
    #
    # plt.figure()
    # ax=plt.subplot()
    # plt.rcParams.update({
    # "legend.fancybox": False,
    # "legend.frameon": True,
    # "text.usetex": True,
    # "font.family": "serif",
    # "font.serif": ["Times"],
    # "font.size":20})
    # Decoding_s1=[80000]*len(T)
    # BVA_l_s1=[50000]*len(T)
    # BVA_h_s1=[60000]*len(T)
    #
    # plt.plot(T, BVMA_s, linestyle = '-', color = color_def[0], marker = 'o',
    #           markersize = 12, linewidth=1.5, markeredgewidth=0.8)
    # plt.plot(T, BVA_l_s1, linestyle = '-', color = color_def[3], marker = 'v',
    #          markersize = 12, linewidth=1.5, markeredgewidth=0.8)
    # plt.plot(T, BVA_h_s1, linestyle = '-', color = color_def[2], marker = '^',
    #          markersize = 12, linewidth=1.5, markeredgewidth=0.8)
    #
    # plt.plot(T, Decoding_s1, linestyle = '-', color = color_def[1], marker = 's',
    #          markersize = 12, linewidth=1.5, markeredgewidth=0.8)
    # plt.plot(T, Violin_s, linestyle = '-', color = color_def[4], marker = 'd',
    #          markersize = 12, linewidth=1.5, markeredgewidth=0.8)
    # plt.plot(T, ZKP_s, linestyle = '-', color = 'cyan', marker = 'X',
    #          markersize = 12, linewidth=1.5, markeredgewidth=0.8)
    #
    # plt.yscale('log')
    # ax.set_xticks([200,600,1000,1400], [200,600,1000,1400])
    # ax.set_yticks([8000,10000,20000,30000,40000,50000,80000], [None,"$1 \\times 10^4$","$2 \\times 10^4$","$3 \\times 10^4$","$4 \\times 10^4$","$6 \\times 10^6$","$5 \\times 10^{11}$"])
    # plt.xlabel("Threshold $\\tau$")
    # plt.ylabel("injected size")
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig("./pic/" + 'T_size_enron.pdf', bbox_inches = 'tight', dpi = 600)
    # plt.show()
    #
    
    