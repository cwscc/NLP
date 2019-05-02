#!/usr/bin/env python
# encoding: utf-8
'''
@author: jgy

@software: Pycharm
@file: prefixspan.py
@time: 2019/3/30 13:15
@desc:
'''
import sys

PLACE_HOLDER = '_'

def read(filename):
    S = []
    with open(filename, 'r', encoding='utf-8') as input:
        for line in input.readlines():
            line = line.strip('(').strip(')\n')
            elements = line.split(',')
            s = []
            for e in elements:
                s.append(e.split())
            S.append(s)
    return S

class SquencePattern:
    def __init__(self, squence, support):
        self.squence = []
        for s in squence:
            self.squence.append(list(s))
        self.support = support

    def append(self, p):
        if p.squence[0][0] == PLACE_HOLDER:
            first_e = p.squence[0]
            first_e.remove(PLACE_HOLDER)
            self.squence[-1].extend(first_e)
            self.squence.extend(p.squence[1:])
        else:
            self.squence.extend(p.squence)
        self.support = min(self.support, p.support)


def prefixSpan(pattern, S, threshold):
    patterns = []
    f_list = frequent_items(S, pattern, threshold)

    for i in f_list:
        p = SquencePattern(pattern.squence, pattern.support)
        p.append(i)
        patterns.append(p)

        p_S = build_projected_database(S, p)
        p_patterns = prefixSpan(p, p_S, threshold)
        patterns.extend(p_patterns)

    return patterns


def frequent_items(S, pattern, threshold):
    items = {}
    _items = {}
    f_list = []
    if S is None or len(S) == 0:
        return []

    if len(pattern.squence) != 0:
        last_e = pattern.squence[-1]
    else:
        last_e = []
    for s in S:

        # class 1
        is_prefix = True
        for item in last_e:
            if item not in s[0]:
                is_prefix = False
                break
        if is_prefix and len(last_e) > 0:
            index = s[0].index(last_e[-1])
            if index < len(s[0]) - 1:
                for item in s[0][index + 1:]:
                    if item in _items:
                        _items[item] += 1
                    else:
                        _items[item] = 1

        if PLACE_HOLDER in s[0]:
            for item in s[0][1:]:
                if item in _items:
                    _items[item] += 1
                else:
                    _items[item] = 1
            s = s[1:]

        counted = []
        for element in s:
            for item in element:
                if item not in counted:
                    counted.append(item)
                    if item in items:
                        items[item] += 1
                    else:
                        items[item] = 1

    f_list.extend([SquencePattern([[PLACE_HOLDER, k]], v)
                   for k, v in _items.items()
                   if v >= threshold])
    f_list.extend([SquencePattern([[k]], v)
                   for k, v in items.items()
                   if v >= threshold])
    sorted_list = sorted(f_list, key=lambda p: p.support)
    return sorted_list


def build_projected_database(S, pattern):
    """ 
    suppose S is projected database base on pattern's prefix, 
    so we only need to use the last element in pattern to 
    build projected database 
    """
    p_S = []
    last_e = pattern.squence[-1]
    last_item = last_e[-1]
    for s in S:
        p_s = []
        for element in s:
            is_prefix = False
            if PLACE_HOLDER in element:
                if last_item in element and len(pattern.squence[-1]) > 1:
                    is_prefix = True
            else:
                is_prefix = True
                for item in last_e:
                    if item not in element:
                        is_prefix = False
                        break

            if is_prefix:
                e_index = s.index(element)
                i_index = element.index(last_item)
                if i_index == len(element) - 1:
                    p_s = s[e_index + 1:]
                else:
                    p_s = s[e_index:]
                    index = element.index(last_item)
                    e = element[i_index:]
                    e[0] = PLACE_HOLDER
                    p_s[0] = e
                break
        if len(p_s) != 0:
            p_S.append(p_s)

    return p_S

def select_patterns(wordType, patterns):
    patternsList = []
    count = 0
    for p in patterns:
        item = p.squence[0][0]
        if wordType=='noun':
            if len(p.squence)==2 or item[-2] == 'n' or (item[-2]=='s' and item[-3]=='w'):
                if item[2] != '/':  # 过滤单字词
                    count += 1
                    patternsList.append(item.strip('\'').strip('/n'))
        else:
            if len(p.squence) == 2 or item[-2] == 'v' :
                if item[2]!='/':    # 过滤单字词
                    count += 1
                    patternsList.append(item.strip('\'').strip('/v'))
    print('total: ',count)
    return patternsList

def getDict():
    scw = open('./files/sub_cognition_words.txt', 'r', encoding='utf-8')
    tq = open('./files/time_quantifier.txt', 'r', encoding='utf-8')
    ps = open('./files/product_self.txt', 'r', encoding='utf-8')
    dic_scw, dic_tq, dic_ps = [], [], []
    for each in scw.readlines():
        dic_scw.append(each.strip('\n').strip('\ufeff'))
    for each in tq.readlines():
        dic_tq.append(each.strip('\n').strip('\ufeff'))
    for each in ps.readlines():
        dic_ps.append(each.strip('\n').strip('\ufeff'))
    return dic_scw, dic_tq, dic_ps

def filter(dataSet, dic):
    for each in dic:
        dataSet.discard(each)
    return dataSet

def wFile(filename, words):
    f = open('./files/' + filename, 'w', encoding='utf-8')
    for word in words:
        f.write(str(word) + '\n')
    print('已写入文件', filename)

if __name__ == '__main__':
    brand = ['hp', 'lenovo']
    wordType = ['noun', 'verb']
    attrList = []
    for wt in wordType:
        for b in brand:
            S = read('./files/'+b+'fragments('+wt+').txt')
            minSupp = 0.01 if wt == 'noun' else 0.015
            patterns = prefixSpan(SquencePattern([], sys.maxsize), S, len(S) * minSupp)
            patternsList = select_patterns(wt, patterns)
            attrList.append(patternsList)
            print(b, wt, ': ',patternsList)
    # 不同品牌的动词/名词频繁项集取并
    nounSet = set(attrList[0]) | set(attrList[1])
    verbSet = set(attrList[2]) | set(attrList[3])
    # 获取表示主观认识类动词和时间长短的数量词词典
    dic_scw, dic_tq, dic_ps = getDict()
    # 在主观认识类词词典中的“感受”一词在词性标注时既可以作为动词也可以作为名词
    # 考虑到“感受”一词在名词频繁项集中属于抽象类名词，也无法作为有用的一种具体属性，故对其一次过滤
    print('主观认识类词汇过滤...');nouns = filter(nounSet, dic_scw)
    print('时间量词过滤...');nouns = filter(nouns, dic_tq)
    print('产品本身属性过滤...');nouns = filter(nouns, dic_ps)
    print('所有候选名词：', len(nouns), nouns)
    print('主观认识类动词过滤...');verbs = filter(verbSet, dic_scw)
    print('所有候选动词：', len(verbs), verbs)
    wFile('CNoun.txt', nouns)
    wFile('CVerb.txt', verbs)