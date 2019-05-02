#!/usr/bin/env python
# encoding: utf-8
'''
@author: jgy

@software: Pycharm
@file: FP_Growth.py
@time: 2019/3/30 09:28
@desc:
'''
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue       #存放结点名字
        self.count = numOccur       #计数
        self.nodeLink = None        #连接相似结点
        self.parent = parentNode    #存放父结点
        self.children = {}             #存放子结点

    def inc(self, numOccur):    #增加结点出现次数值
        self.count += numOccur

    def disp(self, ind=1):      #输出FP树
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

'''构建FP树(每次构建出来的树可能不同但是等价的)'''
def createTree(dataSet, minSupport):
    #第一次遍历数据集，创建头指针
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):
        if headerTable[k] < minSupport:
            del (headerTable[k])

    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0: return None, None     #如果没有元素项满足要求，则退出
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null Set', 1, None)     #根结点

    #第二次遍历数据集，创建FP树
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]     #根据全局频率对每个事务中的元素进行排序
        if len(localD) > 0:
            #字典根据值排序sorted(iterable,key,reverse)，（迭代对象，选取参与比较的函数，倒/顺序，默认顺序false）
            orderedItems = [v[0] for v in sorted(localD.items(),
                                                 key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)   #使用排序后的频率项集对树进行填充

    return retTree, headerTable

'''FP_Tree Growth'''
def updateTree(items, inTree, headerTable, count):
    if (items[0] in inTree.children):
        inTree.children[items[0]].inc(count)
    else:   #创建一个新的树节点作为子节点添加到树中
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    #对剩下的元素项迭代调用updateTree函数
    if len(items)>1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)
'''更新头指针'''
def updateHeader(nodeToTest, targetNode):
    #获取头指针表中该元素项对应的单链表的尾节点，然后将其指向新节点targetNode
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

'''初始化数据集'''
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
         retDict[frozenset(trans)] = 1
    return retDict

'''发现以给定元素项结尾的所有路径的函数，生成条件模式基'''
#迭代上溯整棵树
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
       # print(leafNode.parent)
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

'''寻找前缀路径'''
def findPrefixPath(basePat, treeNode):
    #创建前缀路径
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        #print(treeNode.name)
        if len(prefixPath)>1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

'''挖掘条件树'''
def mineTree(inTree, headerTable, minSupport, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    #从条件模式基来构建条件FP树
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])    #条件模式基
        myCondTree, myHead = createTree(condPattBases, minSupport)
        #挖掘条件FP树
        if myHead != None:
            mineTree(myCondTree, myHead, minSupport, newFreqSet, freqItemList)