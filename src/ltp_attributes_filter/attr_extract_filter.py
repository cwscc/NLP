#!/usr/bin/env python
# encoding: utf-8
'''
@author: jgy

@software: Pycharm
@file: attr_extract_filter.py
@time: 2019/3/30 09:27
@desc:
'''
import data_processing as dp
import FP_Growth as fp
import pruning as pr
import extract_fragments as ef

_dbName = 'postagprecessed'

def aef(brand, wordType):
    tables = ['pos_'+brand+'scorepos', 'pos_'+brand+'scoreneg']

    print('**************************** '+ brand +' *****************************')
    simpDat = dp.loadSimpDat(_dbName, tables, wordType)
    print('原始数据：', simpDat)
    initSet = fp.createInitSet(simpDat)
    minSupport = len(initSet) * 0.01

    myFPtree, myHeaderTab = fp.createTree(initSet, minSupport)
    myFreqItems = []
    fp.mineTree(myFPtree, myHeaderTab, minSupport, set([]), myFreqItems)
    print('频繁项： ', myFreqItems)
    count1, count2, count3 = [], [], []
    for value in myFreqItems:
        if len(value) == 1: count1.append(value);
        if len(value) == 2: count2.append(value)
        if len(value) == 3: count3.append(value)
    print('一维频繁项集：', len(count1))
    print('二维频繁项集：', len(count2), count2)
    print('三维频繁项集：', len(count3), count3)

    print('------------------------------二、三维剪枝------------------------------')
    freq_2 = pr.distance_2(count2,_dbName, tables)
    freq_3 = pr.distance_3(count3,_dbName, tables)
    print('二维频繁项集: total:', len(freq_2), '\n', freq_2)
    print('三维频繁项集: total:', len(freq_3), '\n', freq_3)

    print('-----------------------------调整语序后的二维---------------------------')
    freq_2 = pr.injust_order_2(freq_2,_dbName, tables)
    print('二维频繁项集: total:', len(freq_2))
    for ee in freq_2:
        print(ee, end=', ')

    print('\n-----------------------一维频繁项集独立支持度剪枝----------------------')
    freq_1 = pr.support(simpDat, count1, freq_2)
    print('一维频繁项集: total:', len(freq_1), freq_1)

    print('--------------------------------片段抽取---------------------------------')
    fragmentList = ef.wFile(brand, wordType, freq_1, freq_2)
    print('最终抽取片段：total:', len(fragmentList), fragmentList)

    print('**************************** ' + brand + ' *****************************')

