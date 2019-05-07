import pandas as pd
import pymysql
from sqlalchemy import create_engine

'''
预处理
去重、删除此用户未填写评价内容、短句过滤
'''


# 在数据库中创建表
def create_table():
    db = pymysql.connect("localhost", "root", "123456", "preprocessing")
    cursor = db.cursor()
    
    # 使用 execute() 方法执行 SQL，如果表存在则删除
    cursor.execute("DROP TABLE IF EXISTS allposcomment")
    cursor.execute("DROP TABLE IF EXISTS allnegcomment")

    sql1 = """CREATE TABLE allposcomment (
             `id` INT NOT NULL AUTO_INCREMENT,
             `comment` VARCHAR(5000) NOT NULL,
              PRIMARY KEY (`id`))"""
    sql2 = """CREATE TABLE allnegcomment (
             `id` INT NOT NULL AUTO_INCREMENT,
             `comment` VARCHAR(5000) NOT NULL,
              PRIMARY KEY (`id`))"""
    
    cursor.execute(sql1)
    cursor.execute(sql2)
    
    print("CREATE TABLE OK\n")
    # 关闭数据库连接
    cursor.close()
    db.close()
    

'''连接数据库，读取数据'''
if __name__ == '__main__':
    create_table()
    
    print("开始去重、删除此用户未填写评价内容、短句过滤...")
    
    conn = pymysql.connect(host="127.0.0.1", user="root", passwd="123456", db="crawl2", charset='utf8')
    
    sqlPos = "select comment from allposcomment"
    sqlNeg = "select comment from allnegcomment"
    
#     sql1 = "select comment FROM crawl.hpscoreneg where score=1 union select comment from crawl.lenovoscoreneg union select comment from crawl.negcomment where score=1"
#     sql2 = "select comment from poscomment"
    
#     sql1 = "select comment from hpscoreneg1"
#     sql2 = "select comment from hpscorepos1"
#     sql3 = "select comment from lenovoscoreneg1"
#     sql4 = "select comment from lenovoscorepos1"
    
    data1 = pd.read_sql(sqlPos, conn)
    data2 = pd.read_sql(sqlNeg, conn)
#     data3 = pd.read_sql(sql3, conn)
#     data4 = pd.read_sql(sql4, conn)
    
#     print("处理前：\ndata1")
#     print(data1.info())
#     print("\ndata2")
#     print(data2.info())
#     print("\ndata3")
#     print(data3.info())
#     print("\ndata4")
#     print(data4.info())
    
    '''去重'''
    data1 = data1.drop_duplicates()
    data2 = data2.drop_duplicates()
#     data3 = data3.drop_duplicates()
#     data4 = data4.drop_duplicates()
    # print(data.duplicated().value_counts() + "\n") # 对返回结果进行计数
    # data.drop_duplicates(subset='content', keep='first', inplace=True)
    #     print("去重后：")
    #     data.info()
    
    '''删去默认评论'''
    print("默认评论共有：\n")
    print("data1\n" + data1.loc[data1["comment"] == "此用户未填写评价内容"])
    print("data2\n" + data2.loc[data2["comment"] == "此用户未填写评价内容"])
#     print("data3\n" + data3.loc[data3["comment"] == "此用户未填写评价内容"])
#     print("data4\n" + data4.loc[data4["comment"] == "此用户未填写评价内容"])
    
    data1.drop(data1.loc[data1["comment"] == "此用户未填写评价内容"].index, axis=0, inplace=True)
    data1['comment'] = data1['comment'].str.replace('此用户未填写评价内容', '')  # 查漏
    
    data2.drop(data2.loc[data2["comment"] == "此用户未填写评价内容"].index, axis=0, inplace=True)
    data2['comment'] = data2['comment'].str.replace('此用户未填写评价内容', '')
    
#     data3.drop(data3.loc[data3["comment"] == "此用户未填写评价内容"].index, axis=0, inplace=True)
#     data3['comment'] = data3['comment'].str.replace('此用户未填写评价内容', '')
#     
#     data4.drop(data4.loc[data4["comment"] == "此用户未填写评价内容"].index, axis=0, inplace=True)
#     data4['comment'] = data4['comment'].str.replace('此用户未填写评价内容', '')
    # print(data.iloc[0:81])  # 查看是否删除成功
    #     print("\n默认评论删除后（短句过滤前）：")
    #     data.info()
    
    '''短句过滤'''
    data1 = data1.iloc[:, 0]  # 定位，取评论那部分数据
    data1 = data1[data1.apply(len) >= 5]  # 至少5个字的评论才会被保留
    
    data2 = data2.iloc[:, 0]
    data2 = data2[data2.apply(len) >= 5]
    
#     data3 = data3.iloc[:, 0]
#     data3 = data3[data3.apply(len) >= 5]
#     
#     data4 = data4.iloc[:, 0]
#     data4 = data4[data4.apply(len) >= 5]
    
    '''写进数据库'''
    # 范例：engine = create_engine("mysql+pymysql://user:password@host:port/databasename?charset=utf8",echo=False)
    engine_loacl = create_engine('mysql+pymysql://root:123456@127.0.0.1:3306/preprocessing?charset=utf8', echo=False)
    data1.to_sql(name='allposcomment', con=engine_loacl, if_exists="append", index=False)
    data2.to_sql(name='allnegcomment', con=engine_loacl, if_exists="append", index=False)
#     data3.to_sql(name='lenovoscoreneg1', con=engine_loacl, if_exists="append", index=False)
#     data4.to_sql(name='lenovoscorepos1', con=engine_loacl, if_exists="append", index=False)
    
    conn.close()
    print("\nfinish!")
