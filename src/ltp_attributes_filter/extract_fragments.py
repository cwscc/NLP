#!/usr/bin/env python
# encoding: utf-8
'''
@author: jgy

@software: Pycharm
@file: extract_fragments.py
@time: 2019/3/30 12:13
@desc:
'''
import data_processing as dp
'''
    提取片段的词距离为3，
    根据用户的语言习惯先进行右遍历再进行左遍历
'''

_dbName = 'fragmentextraction'
_hpTable = ['hpscorepos_punc_pos', 'hpscoreneg_punc_pos']
_lenovoTable = ['lenovoscorepos_punc_pos', 'lenovoscoreneg_punc_pos']

'''抽取不同品牌的片段并写入txt文件'''
def wFile(brand, wordType, freq_1, freq_2):
    filename = brand + 'fragments(noun).txt' if wordType=='n' else brand + 'fragments(verb).txt'
    f = open('./files/'+filename, 'w', encoding='utf-8')
    comments = []
    if brand=='hp':
        for table in _hpTable:
            comments += getComments(_dbName, table)
    else:
        for table in _lenovoTable:
            comments += getComments(_dbName, table)
    fragmentList = getFragments(freq_1, freq_2, comments)
    for fragment in fragmentList:
        f.write(str(fragment) + '\n')
    print('已写入文件', filename)
    return fragmentList

'''获取词性标注后的评论数据'''
def getComments(dbName, table):
    try:
        conn, cursor1, cursor2, count = dp.connect_database(dbName, table)
        comments = []
        for i in range(0, count):
            commentstr = str(cursor2.fetchone()).replace("ufeff",'').replace("hellip",'')
            commentstr = commentstr.strip("('").strip("',)")
            comments.append(commentstr)
        conn.commit()
        cursor2.close()
        cursor1.close()
        conn.close()
        print("SUCCESS: 评论获取成功！")
        return comments
    except:
        print("ERROR: 评论获取失败！")
        return None

'''根据一、二维候选频繁项集抽取片段'''
def getFragments(freq_1, freq_2, comments):
    # 将每句话转换成列表元素存进片段列表
    fragments = []
    for fragItem in comments:
        for each in fragItem.split('，'):
            temp = []
            for item in each.split(' '):
                if item!='':
                    temp.append(item)
            fragments.append(temp)
    fragmentList = []
    # 利用二维候选频繁项集提取、过滤片段
    for item in freq_2:
        for fragItem in fragments:
            frag = extract_2(item[0], item[1], fragItem)
            if frag!=None:
                fragmentList += frag
                del fragments[fragments.index(fragItem)]    #过滤掉在候选二维频繁项集条件下的片段，以便下一步一维筛选
    # 利用一维候选频繁项集提取片段
    for item in freq_1:
        for fragItem in fragments:
            frag = extract_1(item, fragItem)
            if frag!=None:
                fragmentList += frag
    return list(set(fragmentList))

'''根据二维频繁项集进行片段提取'''
def extract_2(w1, w2, fragItem):
    fragment = []
    for el in fragItem:
        index = fragItem.index(el)
        if w1==el.split('/')[0] and index!=(len(fragItem)-1):
            if w2==fragItem[index+1].split('/')[0]:
                # 对符合条件的片段按照“右遍历-->左遍历”的顺序搜索形容词
                t_right = _right(fragItem, index, 2)
                if t_right == None:
                    t_left = _left(fragItem, index, 2)
                    if t_left!=None: fragment.append(tuple(t_left))
                else:   fragment.append(tuple(t_right))
    if len(fragment)!=0: return list(set(fragment))
    return None

'''根据一维频繁项集进行片段提取'''
def extract_1(word, fragItem):
    fragment = []
    for el in fragItem:
        if word == el.split('/')[0]:
            index = fragItem.index(el)
            t_right = _right(fragItem, index, 1)
            if t_right == None:
                t_left = _left(fragItem, index, 1)
                if t_left != None:   fragment.append(tuple(t_left))
            else:
                fragment.append(tuple(t_right))
    if len(fragment) != 0: return list(set(fragment))
    return None

'''左遍历寻找'''
def _left(fragment, index, flag):
    temp = []
    if flag==1:
        if index >= 3:
            # 向左遍历三个元素，抓取距离最近的片段
            for i in range(index - 1, index - 4, -1):
                if fragment[i][-1] == 'a':
                    for j in range(0, index - i + 1):
                        temp.append(fragment[j + i])
                    return temp
        else:
            for i in range(index - 1, 0):
                if fragment[i][-1] == 'a':
                    for j in range(0, index - i + 1):
                        temp.append(fragment[j + i])
                    return temp
    else:
        if index+1 >= 4:
            for i in range(index - 1, index - 4, -1):
                if fragment[i][-1] == 'a':
                    for j in range(0, index - i):
                        temp.append(fragment[j + i])
                    temp.append(fragment[index]+(fragment[index+1]))
                    #print(temp)
                    return temp
        else:
            for i in range(index - 1, 0):
                if fragment[i][-1] == 'a':
                    for j in range(0, index - i):
                        temp.append(fragment[j + i])
                    temp.append(fragment[index]+(fragment[index + 1]))
                    return temp
    return None

'''右遍历寻找'''
def _right(fragment, index, flag):
    temp = []
    if flag==1:
        if len(fragment) - (index + 1) > 2:
            # 向右遍历三个元素，抓取距离最近的片段
            for i in range(index+1, index+4):
                if fragment[i][-1] == 'a':
                    for j in range(0, i-index+1):
                        temp.append(fragment[j+index])
                        # temp[j] = fragment[j+index]
                    return temp
        else:
            for i in range(index+1, len(fragment)-1):
                if fragment[i][-1] == 'a':
                    for j in range(0, i-index+1):
                        temp.append(fragment[j+index])
                    return temp
    else:
        if len(fragment) - (index + 1) - 1 > 2:
            # 向右遍历三个元素，抓取距离最近的片段
            for i in range(index + 1, index + 4):
                if fragment[i][-1] == 'a':
                    for j in range(0, i - index):
                        temp.append(fragment[j + index])
                    temp.append(fragment[index]+(fragment[index + 1]))
                    return temp
        else:
            for i in range(index + 1, len(fragment) - 1):
                if fragment[i][-1] == 'a':
                    for j in range(0, i - index):
                        temp.append(fragment[j + index])
                    temp.append(fragment[index]+(fragment[index + 1]))
                    return temp
    return None

'''提取片段'''
def extract_frag(item, fragment):
    _fragment = []
    for el in fragment:
        if item == el.split('/')[0]:
            index = fragment.index(el)
            t_left = _left(fragment, index) #先进行左遍历
            if t_left==None:
                t_right = _right(fragment, index)   #左遍历无值再进行右遍历
                if t_right!=None:   _fragment.append(tuple(t_right))
            else: _fragment.append(tuple(t_left))
    if len(_fragment)!=0: return list(set(_fragment))
    return None
