# import sys,os
# path1=os.path.abspath('..')
# path2 = os.path.dirname(path1)
# path3 = os.path.abspath(path2 + '/连接词/connecting_words_find.txt')
# f = open(path3, encoding='gbk')
# adjust_list = f.readlines()
# for i in adjust_list:
#     print(i)
#     break
# # print(path1)
# print(path3)
# import pymysql
# # 测试集验证（test set）
# conn = pymysql.connect("localhost", "root", "123456", "crawl")
# cursor = conn.cursor()
# sqlHpPos = "select * from hpscorepos1"
# sqlHpNeg = "select * from hpscoreneg1"
# sqlLenovoPos = "select * from lenovoscorepos1"
# sqlLenovoNeg = "select * from lenovoscoreneg1"
# 
# sql = "select * from hpscorepos1 union select * from hpscoreneg1 union \
# select * from lenovoscorepos1 union select * from lenovoscoreneg1"
# cursor.execute(sql)
# testSet = cursor.fetchall()
# print(type(testSet[0][1]))
# # 关闭数据库连接
# cursor.close()
# conn.close()
