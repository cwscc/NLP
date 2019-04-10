'''
Created on 2018年9月16日

@author: 84468
'''
import jieba
import pymysql.cursors
import chardet

'''
分词、去停用词

'''


# 创建停用词list  
def stopwordslist(filepath, encode="utf-8"):
    # stopwords = [line.strip() for line in open(filepath, 'rb',).readlines()]
    stopwords = [line.strip() for line in open(filepath, 'r', encoding=encode).readlines()]

    return stopwords



# 调整词典，可调节单个词语的词频，使其能（或不能）被分出来，调整分词效果
def adjust_jieba_dict(adjust_word_file, encode="utf-8"):
    f = open(adjust_word_file, encoding=encode)
    adjust_list = f.readlines()
    for i in adjust_list:
        jieba.suggest_freq(i.strip(), True)
    f.close()


# 对句子进行分词  
# def seg_sentence(sentence):  
#     sentence_seged = jieba.cut(sentence.strip())
#     stopwords = stopwordslist('')  # 这里加载停用词的路径
#     stopwords1 = stopwordslist('D:/大三/大创/停用词/stop_words（官方）.txt')
#     for i in stopwords1:
#         stopwords.append(i)
#     outstr = ''  
#     for word in sentence_seged:  
#         if word not in stopwords:  
#             if word != '\t':
#                 outstr += word  
#                 outstr += " "  
#     return outstr

# 对句子进行分词,此步做好了去停用词和分词，并且只处理一个句子
# 这个方法是2019/4/10更改
def seg_sentence_new(sentence):
    sentence_seged = jieba.cut(sentence.strip())  #生成一个生成器

    stopwords = stopwordslist('/Users/chenjiaqi/Documents/mylearn/git_resposity/othergit/NLP/停用词/stop_words_符号.txt',encode='utf-8')  # 这里加载停用词的路径
    stopwords1 = stopwordslist('/Users/chenjiaqi/Documents/mylearn/git_resposity/othergit/NLP/停用词/stop_word_官方改new.txt', encode='utf-8')
    stopword_replace = stopwordslist("/Users/chenjiaqi/Documents/mylearn/git_resposity/othergit/NLP/停用词/stop_words_官方抽取替换.txt", encode='utf-8')
    # 停用词合并
    for i in stopwords1:
        stopwords.append(i)
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t' and word != "\n" and word != "\r" and word!="\b" and word!=" ":
                if word in stopword_replace:
                    outstr += "， "
                else:
                    outstr += word
                    outstr += " "
    outstr = remove_duplicate_dot(outstr)  #去掉标点
    return outstr.strip()

# 移除标点符号，删除并列的重复标点符号，头尾的标点符号，栈的思想
def remove_duplicate_dot(sentence):
    sentence_list = sentence.split(" ") #以空格分隔
    no_dupli_list = []
    no_val_num = [0,1,2,3,4,5,6,7,8,9,"一","二","三","四","五","六","七","八","九","十","零"]
    for word_index,word in enumerate(sentence_list):
        if word.strip() != "":
            if word != "，":
                no_dupli_list.append(word)
            elif word == "，":
                if len(no_dupli_list)!=0 and no_dupli_list[-1] != "，" and no_dupli_list[-1] not in no_val_num and no_dupli_list[-1].isdigit() == False:
                    no_dupli_list.append("，")
                elif len(no_dupli_list)!=0 and (no_dupli_list[-1] in no_val_num or no_dupli_list[-1].isdigit() == False):
                    no_dupli_list.pop()

    if len(no_dupli_list)!=0 and no_dupli_list[-1] == "，":
        no_dupli_list.pop()

    no_dupli_str = ' '.join(no_dupli_list)  #空格分隔

    return no_dupli_str





def create_table():
    db = pymysql.connect("localhost", "root", "Xg123456", "fragmentExtraction")
    cursor = db.cursor()
    # 使用 execute() 方法执行 SQL，如果表存在则删除
    cursor.execute("DROP TABLE IF EXISTS hpscoreneg_punc")
    cursor.execute("DROP TABLE IF EXISTS hpscorepos_punc")
    cursor.execute("DROP TABLE IF EXISTS lenovoscoreneg_punc")
    cursor.execute("DROP TABLE IF EXISTS lenovoscorepos_punc")

    sql1 = """CREATE TABLE hpscoreneg_punc (         
             `id` INT NOT NULL AUTO_INCREMENT,
             `comment` VARCHAR(1000) NOT NULL,
              PRIMARY KEY (`id`))"""
    sql2 = """CREATE TABLE hpscorepos_punc (         
             `id` INT NOT NULL AUTO_INCREMENT,
             `comment` VARCHAR(1000) NOT NULL,
              PRIMARY KEY (`id`))"""
    sql3 = """CREATE TABLE lenovoscoreneg_punc (         
             `id` INT NOT NULL AUTO_INCREMENT,
             `comment` VARCHAR(1000) NOT NULL,
              PRIMARY KEY (`id`))"""
    sql4 = """CREATE TABLE lenovoscorepos_punc (         
             `id` INT NOT NULL AUTO_INCREMENT,
             `comment` VARCHAR(1000) NOT NULL,
              PRIMARY KEY (`id`))"""
    
    cursor.execute(sql1)
    cursor.execute(sql2)
    cursor.execute(sql3)
    cursor.execute(sql4)
    
    print("CREATE TABLE OK")
    # 关闭数据库连接
    cursor.close()
    db.close()


def main():
    #创建四个表，供处理后存储
    create_table()
    
    # 建立两个数据库连接，第一个存的是去重、无效评论和短文本的数据，第二个是分词去停用词后存储的
    conn1 = pymysql.connect(host='127.0.0.1', user='root', passwd='Xg123456', db='preprocessing', charset='utf8')
    cur1 = conn1.cursor()
    conn2 = pymysql.connect(host='127.0.0.1', user='root', passwd='Xg123456', db='fragmentExtraction', charset='utf8')
    cur2 = conn2.cursor()
    
    select_sql1 = "select `comment` from hpscoreneg"
    insert_sql1 = "insert into `hpscoreneg_punc` (`comment`) values(%s)"
    
    select_sql2 = "select comment from hpscorepos"
    insert_sql2 = "insert into `hpscorepos_punc` (`comment`) values(%s)"
    
    select_sql3 = "select comment from lenovoscoreneg"
    insert_sql3 = "insert into `lenovoscoreneg_punc` (`comment`) values(%s)"
    
    select_sql4 = "select comment from lenovoscorepos"
    insert_sql4 = "insert into `lenovoscorepos_punc` (`comment`) values(%s)"
    
    adjust_jieba_dict(adjust_word_file="/Users/chenjiaqi/Documents/mylearn/git_resposity/othergit/NLP/连接词/connect_word_zhuang.txt", encode="utf-8")
    
    # 开始处理
    cur1.execute(select_sql1)
    result1 = cur1.fetchall()
    for r in result1:
        r = "".join(r)
        line_seg = seg_sentence_new(r)  # 这里的返回值是字符串
        cur2.execute(insert_sql1, line_seg)
    
    cur1.execute(select_sql2)
    result2 = cur1.fetchall()
    for r in result2:
        r = "".join(r)
        line_seg = seg_sentence_new(r)  # 这里的返回值是字符串
        cur2.execute(insert_sql2, line_seg)
        
    cur1.execute(select_sql3)
    result3 = cur1.fetchall()
    for r in result3:
        r = "".join(r)
        line_seg = seg_sentence_new(r)  # 这里的返回值是字符串
        cur2.execute(insert_sql3, line_seg)
        
    cur1.execute(select_sql4)
    result4 = cur1.fetchall()
    for r in result4:
        r = "".join(r)
        line_seg = seg_sentence_new(r)  # 这里的返回值是字符串
        cur2.execute(insert_sql4, line_seg)

    conn1.commit()
    conn2.commit()
    cur1.close()
    cur2.close()
    conn1.close()
    conn2.close()
    
    print("\nfinish!")

if __name__ == "__main__":
    # 此处进行数据的预处理（19.3.11）
    # 统一编码utf-8
    # stop_word = stopwordslist("/Users/chenjiaqi/Documents/mylearn/git_resposity/othergit/NLP/停用词/stop_word_官方改new.txt",encode='utf-8')
    # stop_replace_word = stopwordslist("/Users/chenjiaqi/Documents/mylearn/git_resposity/othergit/NLP/停用词/stop_words_官方抽取替换.txt",encode='utf-8')
    # stop_word2 = stopwordslist("/Users/chenjiaqi/Documents/mylearn/git_resposity/othergit/NLP/停用词/stop_words_符号.txt",encode="utf-8")
    #
    # # 把停用词合并成一个列表
    # for line in stop_word2:
    #     stop_word.append(line)
    #
    # # 添加连接词
    adjust_jieba_dict(adjust_word_file="/Users/chenjiaqi/Documents/mylearn/git_resposity/othergit/NLP/连接词/connect_word_zhuang.txt", encode="utf-8")
    main()







