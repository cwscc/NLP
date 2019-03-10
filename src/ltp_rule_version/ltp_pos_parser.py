# -*- coding: utf-8 -*-
import pymysql
import sys
import os
from pyltp import Postagger
from pyltp import Parser



# LTP nltk 词性标注，nltk句法分析，ltp nltk依存句法分析,结合数据库

# 连接数据库
# db_name = "preprocessed"
user = "root"
password = "Xg123456" ###注意这里填写自己数据库的代码

# LTP模型路径
LTP_DATA_DIR = '/Users/chenjiaqi/Downloads/addition-jar/ltp_data_v3.4.0/'  # ltp模型目录的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`



# 新建数据库
def create_db(db_name, user, password, host="127.0.0.1", charset='utf8'):
    sql1 = "SHOW DATABASES"
    # sql2 = "DROP DATABASES IF EXISTS " + db_name
    sql2 = "CREATE DATABASE IF NOT EXISTS " + db_name
    try:
        # 连接数据库
        db = pymysql.connect(host,user,password,charset=charset)
        # 创建游标，通过连接与数据通信
        cursor = db.cursor()
        # 执行sql语句
        cursor.execute(sql1)
        cursor.execute(sql2)
        db.commit()
        print("Creating db successfully")
    except:
        print("Having error in creating db !")
    finally:
        db.close()


# 连接数据库
def connect_db(db_name, user, password, host="127.0.0.1",port=3306,charset="utf8"):
    # 连接数据库的配置项
    config = {
        'host': host,
        'port': port,  # MySQL默认端口
        'user': user,  # mysql默认用户名
        'password': password,
        'db': db_name,  # 数据库
        'charset': charset,
    }
    # 连接数据库
    db = pymysql.connect(**config)
    return db

# 创建comment,默认列名
def create_table(db, table):
    cursor = db.cursor()    #打开游标
    # 创建sql语句
    sql = "CREATE TABLE " + table + " " + """(
                `id` INT NOT NULL AUTO_INCREMENT,
                `comment` VARCHAR(3000) NOT NULL,
                PRIMARY KEY (`id`))
                """
    try:
        # 使用 execute() 方法执行 SQL ，如果表存在则删除
        cursor.execute("DROP TABLE IF EXISTS " + table)
        # 创建数据库表格
        cursor.execute(sql)
        print("Create table " + table + " is finished!")
        db.commit()
    except:
        print("Having errors in creating table . . . ")
        return False
    finally:
        # 关闭数据库连接
        cursor.close()
        # db.close()
    return True

# 向数据库中插入表格数据
# 前提是先创建好表格
def insert_table(db, table, sentence_list):
    cursor = db.cursor()
    insert_sql = "INSERT INTO `" + table + "` (`comment`) values(%s)"
    #开始处理
    try:
        for sentence in sentence_list:      #把每一条句子插入到表格中
            cursor.execute(insert_sql, str(sentence))
        # 提交到数据库执行
        db.commit()
        print("Insert to db finish!")
    except:
        #如果发生错误则回滚
        db.rollback()
        return False
    finally:
        cursor.close()
        # db.close()
    return True


# 从数据库中读取数据并进行处理, 处理的数据是用LTP还是NLTK,  NLTK-->True   LTP-->False
# table里的数据必须是分好词的
def data_processing(db, table, is_Nltk=False):
    # 使用cursor()方法创建一个游标对象 cursor，操作游标
    cursor = db.cursor()
    select_sql = "SELECT comment FROM `" + table + "`"
    all_sentence = []
    try:
        cursor.execute(select_sql)
        all_results = cursor.fetchall()

        for sentence in all_results:
            new_str = ""
            list_str = []
            for char in sentence[0].split():
                if (char+"").strip() != "":
                    list_str.append((char+"").strip())
            for char in list_str:
                new_str += char + " "
            if not is_Nltk:
                all_sentence.append(new_str.strip().split())
            else:
                all_sentence.append(new_str.strip())

    except:
        print("Having errors in processing data . . . ")
        return None
    finally:
        cursor.close()
        # db.close()
    return all_sentence



def select_all_from_pos_table(db, table):
    cursor = db.cursor()
    select_sql = "SELECT comment FROM `" + table + "`"
    all_sentence = []
    try:
        cursor.execute(select_sql)
        all_results = cursor.fetchall()

        for sentence in all_results:
            # print(sentence[0])
            child_list = eval(sentence[0])
            all_sentence.append(child_list)

    except:
        print("Having errors in selecting data . . . ")
        return None
    finally:
        cursor.close()
        # db.close()
    return all_sentence



# word_list是嵌套列表，包含每一个干净的句子的列表
# LTP词性标注
def postagger_ltp(word_lists):
    postagger = Postagger()     #初始化实例
    postagger.load(pos_model_path)  #加载模型
    all_postags = []
    all_tag = []
    flag = 1
    for one_list in word_lists: #每个句子读取
        tag_str = ""
        postags = postagger.postag(one_list)    #词性标注
        tag_list = [tag for tag in postags]
        for one_word, one_tag in zip(one_list, postags):
            tag_str += one_word + "/" + one_tag + " "
        all_postags.append(tag_str.strip())
        all_tag.append(tag_list)
        print("######LTP POSTagger finished "+ str(flag)+" sentences")
        flag += 1
    postagger.release()                         #释放模型
    print("LTP POSTagger finish!")
    return all_postags, all_tag                 #返回所有句子以及每个句子的词性列表

# arc.head 表示依存弧的父节点词的索引，arc.relation 表示依存弧的关系
#  LTP依存句法分析,返回列表
def parser_ltp(word_list, tag_list):
    parser = Parser()   #初始化实例
    parser.load(par_model_path)
    all_parser_list = []
    flag = 1
    for words, tags in zip(word_list, tag_list):
        arcs_str = ""
        arcs = parser.parse(words, tags)     #依存句法分析，一条评论的依存句法分析
        for arc in arcs:
            arcs_str += str(arc.head) + "/" + arc.relation + " "
        all_parser_list.append(arcs_str)
        print("######LTP DependencyParse finished " + str(flag) + " sentences")
        flag += 1
    parser.release()
    return all_parser_list


#  LTP依存句法分析,返回列表
def parser_ltp_arc(word_list, tag_list):
    parser = Parser()   #初始化实例
    parser.load(par_model_path)
    all_parser_list = []
    for words, tags in zip(word_list, tag_list):
        arcs = parser.parse(words, tags)     #依存句法分析，一条评论的依存句法分析
        all_parser_list.append(arcs)
    parser.release()
    return all_parser_list


# 主要抽取名词以及名词对
# Rule1: ATT(n + n) 提取词对
# Rule2: SBV(n + v) 提取n
# Rule3: ATT(d + nd) 提取nd
# Rule4: VOB + ATT(v + v + n) 提取n==>换成 VOB(v+n/ws)
# Rule5: SBV(n + a) 提取n
# Rule6: COO + LAD(n + c + n) 提取n和n==>  换成 COO（n+n）分别提取n和n
# Rule7: ATT(v + n) 提取v + n
# Rule8: ATT(nt+n)
# 基于依存句法规则提取名词词组，以此提高精度
# 传递原生的arcs列表
def extract_noun_base_rule(sentence_list, parser_arcs_list, sentence_pos_list):
    all_noun_list = [] #所有评论的名词
    freq_noun = ["n","nz","ws"] #经常出现的名词词性
    # rule_dict = {
    #     "ATT" : {
    #        [ {"n" : "n"},
    #          {"d" : "nd"},
    #          {"v" : "n",}]
    #     },
    #     "SBV" : {
    #        [ {"n" : "v"},
    #          {"n" : "a",}]
    #     },
    #     "VOB" : {
    #         "ATT" : [{
    #             "v":{
    #                 {"v" : "n"},
    #             }
    #         }]
    #     },
    #     "COO" : {
    #         "LAD" : [{
    #             "n" : {
    #                 "c" : "n"
    #             }
    #         }]
    #     }
    # }

    for one_sentence, one_arcs, one_pos in zip(sentence_list, parser_arcs_list, sentence_pos_list):
        one_noun_list = []
        for flag in range(len(one_sentence)):
            if one_arcs[flag].relation == "ATT":
                noun_phrase = ""
                if one_pos[flag] == "n" and one_pos[one_arcs[flag].head-1] == "n" and (one_arcs[flag].head-1-flag) == 1: #两个相近的名词，且距离为1
                    noun_phrase += one_sentence[flag] + one_sentence[one_arcs[flag].head-1]
                    one_noun_list.append(noun_phrase)
                elif (one_pos[flag] == "d" and one_pos[one_arcs[flag].head-1] == "nd") or (one_pos[flag] == "v" and one_pos[one_arcs[flag].head-1] == "n") or (one_pos[flag] == "nt" and one_pos[one_arcs[flag].head-1] == "n"):
                    one_noun_list.append(one_sentence[one_arcs[flag].head-1])
            elif one_arcs[flag].relation == "SBV":
                if ((one_pos[flag] == "n" and one_pos[one_arcs[flag].head-1] == "v") or (one_pos[flag] == "n" and one_pos[one_arcs[flag].head-1] == "a")) and (one_arcs[flag].head-1-flag) == 1:
                    one_noun_list.append(one_sentence[flag])
            elif one_arcs[flag].relation == "VOB":
                if one_pos[flag] == "v" and (one_pos[one_arcs[flag].head-1] == "n" or one_pos[one_arcs[flag].head-1] == "ws"):
                    one_noun_list.append(one_sentence[one_arcs[flag].head-1])
            elif one_arcs[flag].relation == "COO":
                if one_pos[flag] in freq_noun and one_pos[one_arcs[flag].head-1] in freq_noun:
                    one_noun_list.append(one_sentence[flag])
                    one_noun_list.append(one_sentence[one_arcs[flag].head-1])
        one_noun_list = list(set(one_noun_list)) #去重
        all_noun_list.append(one_noun_list)

    return all_noun_list




# 依据论文的方法进行抽取，进行语法规则的抽取
# n+n ATT (n+n作为整体抽取）
# n+v ATT （n+v作为整体抽取）
# v+n ATT （v+n作为整体抽取）
# v+a SBV (抽取v作为特征词)
# v+a CMP (抽取v作为特征词)
# v+a VOB (抽取v作为特征词)
def extract_noun_base_pager_rule(sentence_list, parser_arcs_list, sentence_pos_list):
    all_noun_list = [] #所有评论的名词

    count = 0
    for one_sentence, one_arcs, one_pos in zip(sentence_list, parser_arcs_list, sentence_pos_list):
        one_noun_list = []
        print("提取名词第"+str(count)+"句" )
        count += 1

        # 此条件临时加上去
        if len(one_sentence) != len(one_arcs) or len(one_sentence)!=len(one_pos) or len(one_arcs)!=len(one_pos):
            all_noun_list.append("[]")
        else:

            for flag in range(len(one_sentence)):
                if one_arcs[flag].relation == "ATT":
                    noun_phrase = ""
                    if one_pos[flag] == "n" and one_pos[one_arcs[flag].head-1] == "n" and (one_arcs[flag].head-1-flag) == 1: #两个相近的名词，且距离为1
                        noun_phrase += one_sentence[flag] + one_sentence[one_arcs[flag].head-1]
                        one_noun_list.append(noun_phrase)
                    elif (one_pos[flag] == "v" and one_pos[one_arcs[flag].head-1] == "n") and (one_arcs[flag].head-1-flag) == 1:
                        noun_phrase += one_sentence[flag] + one_sentence[one_arcs[flag].head - 1]
                        one_noun_list.append(noun_phrase)
                    elif (one_pos[flag] == "n" and one_pos[one_arcs[flag].head-1] == "v") and (one_arcs[flag].head-1-flag) == 1:
                        noun_phrase += one_sentence[flag] + one_sentence[one_arcs[flag].head - 1]
                        one_noun_list.append(noun_phrase)
                elif one_arcs[flag].relation == "SBV":
                    if (one_pos[flag] == "v" and one_pos[one_arcs[flag].head-1] == "a") and (one_arcs[flag].head-1-flag) == 1:
                        one_noun_list.append(one_sentence[flag])
                elif one_arcs[flag].relation == "VOB":
                    if one_pos[flag] == "a" and one_pos[one_arcs[flag].head-1] == "v" and (flag - one_arcs[flag].head-1) == 1: #反过来
                        one_noun_list.append(one_sentence[one_arcs[flag].head-1])
                elif one_arcs[flag].relation == "CMP":
                    if one_pos[flag] == "a" and one_pos[one_arcs[flag].head-1] == "v" and (flag - one_arcs[flag].head-1) == 1: #反过来减
                        one_noun_list.append(one_sentence[one_arcs[flag].head-1])

            one_noun_list = list(set(one_noun_list)) #去重
            all_noun_list.append(one_noun_list)

    return all_noun_list



if __name__ == "__main__":
    # 分好词的原数据库名
    db_sentence_name = "preprocessed"
    # 词性标注后的数据库名
    db_pos_tag_name = "postagprecessed"

    #  原数据句子的数据库表
    hpN_table_sentence = "hpscoreneg"
    hpY_table_sentence = "hpscorepos"

    lenoN_table_sentence = "lenovoscoreneg"
    lenoY_table_sentence = "lenovoscorepos"

    # ltp词性标注后的数据库表
    hpN_table_pos = "pos_hpscoreneg_tag"
    hpY_table_pos = "pos_hpscorepos_tag"

    lenoN_table_pos = "pos_lenovoscoreneg_tag"
    lenoY_table_pos = "pos_lenovoscorepos_tag"


    # 连接数据库
    # 原数据的数据库
    db_one = connect_db(db_name=db_sentence_name, user=user, password=password)

    # 词性标注后的数据库
    db_two = connect_db(db_name=db_pos_tag_name, user=user, password=password)

    # 原数据列表处理好的
    hpN_sentence_list = data_processing(db=db_one, table=hpN_table_sentence, is_Nltk=False)
    hpY_sentence_list = data_processing(db=db_one, table=hpY_table_sentence, is_Nltk=False)

    lenoN_sentence_list = data_processing(db=db_one, table=lenoN_table_sentence, is_Nltk=False)
    lenoY_sentence_list = data_processing(db=db_one, table=lenoY_table_sentence, is_Nltk=False)



    #  词性标记后的列表
    hpN_pos_list = select_all_from_pos_table(db=db_two,table=hpN_table_pos)
    hpY_pos_list = select_all_from_pos_table(db=db_two,table=hpY_table_pos)
    lenoN_pos_list = select_all_from_pos_table(db=db_two,table=lenoN_table_pos)
    lenoY_pos_list = select_all_from_pos_table(db=db_two,table=lenoY_table_pos)

    a_String = "['n', 'e', 'd', 'v', 'i', 'n', 'n', 'v', 'v', 'n', 'v', 'm', 'n', 'v', 'n', 'n', 'd', 'nh', 'v', 'm', 'q', 'n', 'v', 'n', 'm', 'v', 'n', 'n', 'v', 'ws']"

    print(eval(a_String)[0])

    for i in hpN_pos_list:
        print(i)
        print(type(i))

    parser_list = parser_ltp(word_list=hpN_sentence_list,tag_list=hpN_pos_list)

    #  依存句法分析后的列表
    hpN_parser_list_arc = parser_ltp_arc(word_list=hpN_sentence_list, tag_list=hpN_pos_list)
    hpY_parser_list_arc = parser_ltp_arc(word_list=hpY_sentence_list, tag_list=hpY_pos_list)
    lenoN_parser_list_arc = parser_ltp_arc(word_list=lenoN_sentence_list, tag_list=lenoN_pos_list)
    lenoY_parser_list_arc = parser_ltp_arc(word_list=lenoY_sentence_list, tag_list=lenoY_pos_list)

    # print("########## 原生弧 ########")
    # for arcs in parser_list_arc:
    #     one_str = ""
    #     for i in range(len(arcs)):
    #         one_str += str(arcs[i].relation) + " "
    #     print(one_str)
    #
    # print("########## 非原生弧 #######")
    for arcs in parser_list:
        print(arcs)

    # for one_sentence in hpY_sentence_list:
    #     one_str = ""
    #     for i in one_sentence:
    #         one_str += i + " "
    #     print(one_str)

    # Rule1: ATT(n + n) 提取词对
    # Rule2: SBV(n + v) 提取n
    # Rule3: ATT(d + nd) 提取nd
    # Rule4: VOB + ATT(v + v + n) 提取n
    # Rule5: SBV(n + a) 提取n
    # Rule6: COO + LAD(n + c + n) 提取n和n
    # Rule7: ATT(v + n) 提取v + n

    rule_dict = {
        "ATT" : {

        }
    }

    # for one_sentence, arcs, one_sentence_pos in zip(hpN_sentence_list, parser_list_arc,hpN_pos_list):
    #     print
    for i, arcs in enumerate(parser_list):
        print("the " + str(i) + " sentence " + arcs)

    for flag in range(len(hpN_parser_list_arc[12])):
        print(" ".join("%d:%s" % (arc.head, arc.relation) for arc in hpN_parser_list_arc[12]))
        print(" ".join("%s") % (word) for word in hpN_sentence_list[12])
    for word in hpN_sentence_list[12]:
        print(word)
    print(hpN_sentence_list[12])
    # rule_dict = {
    #     "ATT":
    #         [{"n": "n"},
    #          {"d": "nd"},
    #          {"v": "n", }]
    #     ,
    #     "SBV":
    #         [{"n": "v"},
    #          {"n": "a", }]
    #     ,
    #     "VOB": {
    #         "ATT": [{
    #             "v":
    #                 [{"v": "n"}],
    #         }]
    #     },
    #     "COO": {
    #         "LAD": [{
    #             "n": {
    #                 "c": "n"
    #             }
    #         }]
    #     }
    # }

    all_noun_hpN = extract_noun_base_pager_rule(sentence_list=hpN_sentence_list, parser_arcs_list=hpN_parser_list_arc, sentence_pos_list=hpN_pos_list)
    all_noun_hpY = extract_noun_base_pager_rule(sentence_list=hpY_sentence_list, parser_arcs_list=hpY_parser_list_arc, sentence_pos_list=hpY_pos_list)
    all_noun_lenoN = extract_noun_base_pager_rule(sentence_list=lenoN_sentence_list, parser_arcs_list=lenoN_parser_list_arc, sentence_pos_list=lenoN_pos_list)
    all_noun_lenoY = extract_noun_base_pager_rule(sentence_list=lenoY_sentence_list, parser_arcs_list=lenoY_parser_list_arc, sentence_pos_list=lenoY_pos_list)

    # all_noun2 = []
    # for child_list in all_noun:
    #     for pharse in child_list:
    #         all_noun2.append(pharse)
    #
    # all_noun2 = list(set(all_noun2))
    #
    # for i in all_noun2:
    #     print(i)



    for i in all_noun_hpN:
        print(i)
    with open("hpN.txt",'w') as fp:
        for ch_list in all_noun_hpN:
            fp.write(str(ch_list) + "\n")

    with open("hpY.txt",'w') as fp:
        for ch_list in all_noun_hpY:
            fp.write(str(ch_list) + "\n")

    with open("lenoN.txt",'w') as fp:
        for ch_list in all_noun_lenoN:
            fp.write(str(ch_list) + "\n")

    with open("lenoY.txt", 'w') as fp:
        for ch_list in all_noun_lenoY:
            fp.write(str(ch_list) + "\n")