'''
Created on 2018年9月16日

@author: 84468
'''
import jieba
import pymysql.cursors

'''
分词、去停用词

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


# 对句子进行分词  
def seg_sentence(sentence):  
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('D:/大三/大创/停用词/stop_word_new（修改后）.txt')  # 这里加载停用词的路径  
    stopwords1 = stopwordslist('D:/大三/大创/停用词/stop_words（官方）.txt')
    for i in stopwords1:
        stopwords.append(i)
    outstr = ''  
    for word in sentence_seged:  
        if word not in stopwords:  
            if word != '\t':  
                outstr += word  
                outstr += " "  
    return outstr  


if __name__ == '__main__':
    # 建立两个数据库连接，第一个存的是去重、无效评论和短文本的数据，第二个是分词去停用词后存储的
    conn1 = pymysql.connect(host='127.0.0.1', user='root', passwd='123456', db='preprocessing', charset='utf8')
    cur1 = conn1.cursor()
    conn2 = pymysql.connect(host='127.0.0.1', user='root', passwd='123456', db='preprocessed', charset='utf8')
    cur2 = conn2.cursor()
    
    select_sql1 = "select `comment` from hpscoreneg"
    insert_sql1 = "insert into `hpscoreneg` (`comment`) values(%s)"
    
    select_sql2 = "select comment from hpscorepos"
    insert_sql2 = "insert into `hpscorepos` (`comment`) values(%s)"
    
    select_sql3 = "select comment from lenovoscoreneg"
    insert_sql3 = "insert into `lenovoscoreneg` (`comment`) values(%s)"
    
    select_sql4 = "select comment from lenovoscorepos"
    insert_sql4 = "insert into `lenovoscorepos` (`comment`) values(%s)"
    
    adjust_jieba_dict("D:/大三/大创/连接词/connecting_words_find.txt")
    
    # 开始处理
    cur1.execute(select_sql1)
    result1 = cur1.fetchall()
    for r in result1:
        r = "".join(r)
        line_seg = seg_sentence(r)  # 这里的返回值是字符串 
        cur2.execute(insert_sql1, line_seg)
    
    cur1.execute(select_sql2)
    result2 = cur1.fetchall()
    for r in result2:
        r = "".join(r)
        line_seg = seg_sentence(r)  # 这里的返回值是字符串 
        cur2.execute(insert_sql2, line_seg)
        
    cur1.execute(select_sql3)
    result3 = cur1.fetchall()
    for r in result3:
        r = "".join(r)
        line_seg = seg_sentence(r)  # 这里的返回值是字符串 
        cur2.execute(insert_sql3, line_seg)
        
    cur1.execute(select_sql4)
    result4 = cur1.fetchall()
    for r in result4:
        r = "".join(r)
        line_seg = seg_sentence(r)  # 这里的返回值是字符串 
        cur2.execute(insert_sql4, line_seg)

    conn1.commit()
    conn2.commit()
    cur1.close()
    cur2.close()
    conn1.close()
    conn2.close()
    
    print("finish!")
  
