'''
Created on 2018年10月2日

@author: cws
'''

import jieba
import jieba.posseg as pseg
import pymysql

'''
词性标注及分词，去停用词

'''


# 创建停用词list  
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='gbk').readlines()]  
    return stopwords


# 调整词典，可调节单个词语的词频，使其能（或不能）被分出来，调整分词效果
def adjust_jieba_dict(adjust_word_file):
    f = open(adjust_word_file, encoding='gbk')
    adjust_list = f.readlines()
    for i in adjust_list:
        jieba.suggest_freq(i.strip(), True)
    f.close()


# 词性标注及分词，去停用词
def pos_segment(sentence):
    stopwords = stopwordslist('D:/大三/大创/停用词/stop_word_new（修改后）.txt')  # 这里加载停用词的路径  
    stopwords1 = stopwordslist('D:/大三/大创/停用词/stop_words（官方）.txt')
    for i in stopwords1:
        stopwords.append(i)
    # 词性标注和分词
    sentence_pos_seged = pseg.cut(sentence.strip())
    outstr = ''
    for x in sentence_pos_seged:
        if x.word not in stopwords:
            if x.word != '\t':
                outstr += "{}/{}  ".format(x.word, x.flag)
#         outstr+="{}/{}  ".format(x.word,x.flag)
    return outstr


# 在数据库中创建表
def create_table():
    db = pymysql.connect("localhost", "root", "123456", "preprocessed")
    cursor = db.cursor()
    # 使用 execute() 方法执行 SQL，如果表存在则删除
    cursor.execute("DROP TABLE IF EXISTS hpscoreneg2")
    cursor.execute("DROP TABLE IF EXISTS hpscorepos2")
    cursor.execute("DROP TABLE IF EXISTS lenovoscoreneg2")
    cursor.execute("DROP TABLE IF EXISTS lenovoscorepos2")

    sql1 = """CREATE TABLE hpscoreneg2 (
             `id` INT NOT NULL AUTO_INCREMENT,
             `comment` VARCHAR(5000) NOT NULL,
              PRIMARY KEY (`id`))"""
    sql2 = """CREATE TABLE hpscorepos2 (
             `id` INT NOT NULL AUTO_INCREMENT,
             `comment` VARCHAR(5000) NOT NULL,
              PRIMARY KEY (`id`))"""
    sql3 = """CREATE TABLE lenovoscoreneg2 (
             `id` INT NOT NULL AUTO_INCREMENT,
             `comment` VARCHAR(5000) NOT NULL,
              PRIMARY KEY (`id`))"""
    sql4 = """CREATE TABLE lenovoscorepos2 (
             `id` INT NOT NULL AUTO_INCREMENT,
             `comment` VARCHAR(5000) NOT NULL,
              PRIMARY KEY (`id`))"""
    
    cursor.execute(sql1)
    cursor.execute(sql2)
    cursor.execute(sql3)
    cursor.execute(sql4)
    
    print("CREATE TABLE OK")
    # 关闭数据库连接
    cursor.close()
    db.close()
    
    
    
if __name__ == '__main__':
    create_table()
    
    # 建立两个数据库连接，第一个存的是去重、无效评论和短文本的数据，第二个是词性标注分词去停用词后存储的
    conn1 = pymysql.connect(host='127.0.0.1', user='root', passwd='123456', db='preprocessing', charset='utf8')
    cur1 = conn1.cursor()
    conn2 = pymysql.connect(host='127.0.0.1', user='root', passwd='123456', db='preprocessed', charset='utf8')
    cur2 = conn2.cursor()
    
    select_sql1 = "select `comment` from hpscoreneg1"
    insert_sql1 = "insert into `hpscoreneg2` (`comment`) values(%s)"
    
    select_sql2 = "select comment from hpscorepos1"
    insert_sql2 = "insert into `hpscorepos2` (`comment`) values(%s)"
    
    select_sql3 = "select comment from lenovoscoreneg1"
    insert_sql3 = "insert into `lenovoscoreneg2` (`comment`) values(%s)"
    
    select_sql4 = "select comment from lenovoscorepos1"
    insert_sql4 = "insert into `lenovoscorepos2` (`comment`) values(%s)"
    
    adjust_jieba_dict("D:/大三/大创/连接词/connecting_words_find.txt")
    
    # 开始处理
    cur1.execute(select_sql1)
    result1 = cur1.fetchall()
    for r in result1:
        r = "".join(r)
        line_seg = pos_segment(r)  # 这里的返回值是字符串 
        cur2.execute(insert_sql1, line_seg)
    
    cur1.execute(select_sql2)
    result2 = cur1.fetchall()
    for r in result2:
        r = "".join(r)
        line_seg = pos_segment(r)  # 这里的返回值是字符串 
        cur2.execute(insert_sql2, line_seg)
        
    cur1.execute(select_sql3)
    result3 = cur1.fetchall()
    for r in result3:
        r = "".join(r)
        line_seg = pos_segment(r)  # 这里的返回值是字符串 
        cur2.execute(insert_sql3, line_seg)
        
    cur1.execute(select_sql4)
    result4 = cur1.fetchall()
    for r in result4:
        r = "".join(r)
        line_seg = pos_segment(r)  # 这里的返回值是字符串 
        cur2.execute(insert_sql4, line_seg)

    conn1.commit()
    conn2.commit()
    cur1.close()
    cur2.close()
    conn1.close()
    conn2.close()
    
    print("\nfinish!")
    
