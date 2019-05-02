#!/usr/bin/env python
# encoding: utf-8
'''
@author: jgy

@software: Pycharm
@file: data_processing.py
@time: 2019/3/30 09:35
@desc:
'''
import pymysql

'''连接数据库'''
def connect_database(dbName, table):
    try:
        conn = pymysql.connect(host='127.0.0.1', user='root', password='123456', db=dbName, charset='utf8')
        cursor1 = conn.cursor()  # 用于获取数据总条数
        cursor2 = conn.cursor()  # 用于获取数据所有内容
        cursor1.execute("select count(*) from " + table)
        count = int(str(cursor1.fetchone()).strip("('").strip("',)"))  # 读取总条数查询结果并转换为整数类型
        cursor2.execute('select comment from ' + table + ' where id between 1 and ' + str(count))
        print('SUCCESS: 数据库连接成功！')
        return conn, cursor1, cursor2, count
    except:
        print('ERROR: 数据库连接失败！')
        return None

'''获取原始数据集'''
def loadSimpDat(dbName, tables, wordType):
    simpDat = []
    for table in tables:
        simpDat += select_data(dbName, table, wordType)
    return simpDat

'''查找/筛选原始数据'''
def select_data(dbName, table, wordType):
    try:
        conn, cursor1, cursor2, count = connect_database(dbName, table)
        ddata=[]
        for i in range(0, count):
            commentstr=str(cursor2.fetchone())
            commentstr=commentstr.strip("('").strip("',)")
            if commentstr == '': continue
            templist=[]
            for word in commentstr.split(" "):
                tempstr=word.split("/")
                # 抽取所需名词
                if wordType=='n':
                    if tempstr[1]=="n" or tempstr[1]=="nz" or tempstr[1]=="j" or tempstr[1]=="ws":
                        templist.append(tempstr[0])
                # 抽取动词
                elif wordType=='v':
                    if tempstr[1]=="v":
                        templist.append(tempstr[0])
                # 抽取原始评论
                else:
                    templist.append(tempstr[0])
            ddata.append(templist)
        # 去掉列表里的空列表
        while [] in ddata:
            ddata.remove([])
        conn.commit()
        cursor2.close()
        cursor1.close()
        conn.close()
        print("SUCCESS: 数据获取成功！")
        return ddata
    except:
        print("ERROR: 数据查找错误！")
        return None

'''根据一、二维候选频繁项集抽取片段'''
def getFragments(freq_1, freq_2, comments):
    # 将每句话转换成列表元素存进片段列表
    fragments = []
    # for each in comments:
    #     fragItem = []
    #     fragItem.append(each.split(' '))
    #     fragments.append(fragItem[0])
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
