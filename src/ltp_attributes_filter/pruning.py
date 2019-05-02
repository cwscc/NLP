#!/usr/bin/env python
# encoding: utf-8
'''
@author: jgy

@software: Pycharm
@file: pruning.py
@time: 2019/3/30 10:50
@desc:
'''
import data_processing as dp
import copy

'''采用剪枝策略前对原始数据进行筛选和存放'''
def orignal_data(dbName, tables):
    ori_data = []
    for table in tables:
        ori_data += dp.select_data(dbName, table, wordType='')  # 原始数据
    freqItem = []
    for i in range(len(ori_data)):
        temp = ori_data[i]
        k = 1
        data = {}
        for j in range(len(temp)):
            if temp[j] != '':
                data[k] = temp[j]  # 把每条筛选后的评论放进data字典里
                k += 1
        freqItem.append(data)  # 每条评论被放进字典后，将其存进列表
    return freqItem

'''二维频繁项集邻近规则剪枝'''
def distance_2(data, dbName, tables):
    '''将原来的二维频繁项集中的名词词组放进原始数据进行比对，剪掉不符合条件的词组'''
    freqItem_2 = orignal_data(dbName, tables)     #原始数据
    newFreq2 = []
    for word in data:
        value=list(word)
        count=0;
        for item in freqItem_2:
            k1=0
            k2=0
            for key1 in item:
                if item[key1]==value[0]:k1=key1
            for key2 in item:
                if item[key2]==value[1]:k2=key2
            if k1==0 or k2==0:continue
            diff=abs(k1-k2)
            if diff<=1:count+=1
        if count>=2:newFreq2.append(word)
    return newFreq2
'''三维频繁项集邻近规则剪枝'''
def distance_3(data, dbName, tables):
    '''将原来的二维频繁项集中的名词词组放进原始数据进行比对，剪掉不符合条件的词组'''
    freqItem_3 = orignal_data(dbName, tables)     #原始数据
    newFreq3 = []
    for word in data:
        value=list(word)
        count=0;
        for item in freqItem_3:
            k1=0;k2=0;k3=0
            for key1 in item:
                if item[key1]==value[0]:k1=key1
            for key2 in item:
                if item[key2]==value[1]:k2=key2
            for key3 in item:
                if item[key3]==value[2]:k3=key3
            if k1==0 or k2==0 or k3==0:continue
            diff1=abs(k1-k2)
            diff2=abs(k2-k3)
            if diff1<=1 or diff2<=1:count+=1
        if count>=2:newFreq3.append(word)
    return newFreq3

''' 一维频繁项集独立支持度剪枝'''
def support(dataset,data_1D,data_2D): #输入原始数据、一维频繁项集、邻近规则剪枝后的二维频繁项集
    data_before=[]

    for word in data_2D:
        data_before.append(word[0])
        data_before.append(word[1])
    key_for_prune=list(set(data_before)) #去掉列表中重复的词
    key_after_count={}    #存储每个词出现的句子数
    for each_key in key_for_prune:
        cnt_1 = 0
        for data in dataset:
            if each_key in data:
                cnt_1+=1
        key_after_count[each_key]=cnt_1

    D2_Count={} #存储二维项集出现的次数
    for each in data_2D:
        cnt_2=0
        for data in dataset:
            if each[0] in data and each[1] in data:
                cnt_2+=1
        D2_Count[frozenset(each)]=cnt_2

    in_support=key_after_count
    for key1 in in_support.keys():
        mark=0
        for key2 in D2_Count.keys():
            if key1 in key2:mark+=D2_Count.get(key2)
        in_support[key1]=in_support.get(key1)-mark

    for key1 in list(in_support.keys()):
        if in_support[key1]<=4: #删除不满足支持度的
            for data in data_1D:
                if key1 in data:
                    data_1D.remove(data)
    ndata_1D = []
    for each in data_1D:
        ndata_1D.append(list(each)[0])
    return ndata_1D

'''对二维频繁项集进行语序调整'''
def injust_order_2(data, dbName, tables):
    freqItem_2 = orignal_data(dbName, tables)  # 原始数据
    injust_freq_2 = []

    for word in data:
        value = list(word)
        f1, f2 = 0, 0      #f1(w1,w2); f2(w2,w1)
        for item in freqItem_2:
            k1, k2 = 0, 0
            for key1 in item:
                if item[key1] == value[0]: k1=key1;
            for key2 in item:
                if item[key2] == value[1]: k2=key2;
            if k1 == 0 or k2 == 0: continue
            diff = k1-k2
            if diff<0: f1+=1;
            if diff>0: f2+=1
        if f1 > f2:injust_freq_2.append(value);
        else:
            new_value = []
            new_value = copy.deepcopy(value)
            temp = new_value[0];new_value[0] = new_value[1]; new_value[1] = temp
            injust_freq_2.append(new_value)

    return injust_freq_2