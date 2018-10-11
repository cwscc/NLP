# -*- coding: utf-8 -*-
import pymysql
import sys
import os
from pyltp import Postagger
from pyltp import Parser
from nltk.tag import StanfordPOSTagger
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordParser
from nltk.internals import find_jars_within_path
import shelve

# LTP nltk 词性标注，nltk句法分析，ltp nltk依存句法分析,结合数据库

# 连接数据库
db_name = "preprocessed"
user = "root"
password = "Xg123456"

# LTP模型路径
LTP_DATA_DIR = 'D:/Anaconda/ltp/3.3.1/ltp_data/'  # ltp模型目录的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

# NLTK配置路径
JAVA_PATH = "C:/Program Files/Java/jdk1.8.0_121/bin/java.exe" #Java路径
# NLTK词性标注路径
STANFORD_POSTAGGER_PATH = "D:/Anaconda/stanford-postagger-full-2018-02-27/stanford-postagger.jar" #词性标注模型路径
STANFORD_POSTAGGER_MODELS = "D:/Anaconda/stanford-postagger-full-2018-02-27/models" #词性标注模型路径
nltk_pos_model_filename = 'D:/Anaconda/stanford-postagger-full-2018-02-27/models/chinese-distsim.tagger' #词性标注语料路径
# NLTK句法分析/依存句法分析路径
STANFORD_PARSER_PATH = "D:/Anaconda/stanford-parser-full-2018-02-27/stanford-parser.jar" #句法分析/依存句法的jar包路径
STANFORD_PARSER_MODELS = "D:/Anaconda/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar" #句法分析/依存句法的模型路径
nltk_parse_model_path = 'D:/Anaconda/gz-file-2018/edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz'  #或者chineseFactored.ser.gz/chinesePCFG.ser.gz

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

# 2-返回跳板文件中的的句法树string文本
def return_str_tofile(sentence_parse, filename="temp.txt"):
    f = open(filename, encoding='utf-8', mode='w')
    f.write(sentence_parse + '\n')
    f.close()
    all_sentences = ""
    with open(filename, encoding='utf-8', mode='r') as f:
        sentences = f.readlines()
        for line in sentences:
            all_sentences += line.strip()
    return all_sentences


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
            arcs_str += arc.head + "/" + arc.relation + " "
        all_parser_list.append(arcs_str)
        print("######LTP DependencyParse finished " + str(flag) + " sentences")
        flag += 1
    parser.release()
    return all_parser_list


# 使用NLTK进行词性标注
def postagger_nltk(word_lists):
    os.environ['JAVAHOME'] = JAVA_PATH
    os.environ["STANFORD_PARSER"] = STANFORD_POSTAGGER_PATH
    os.environ["STANFORD_MODELS"] = STANFORD_POSTAGGER_MODELS
    chinese_tagger = StanfordPOSTagger(model_filename=nltk_pos_model_filename,
                                       path_to_jar=STANFORD_POSTAGGER_PATH)
    chinese_tagger.java_options = '-mx12000m'
    nltk_all_tag = []
    flag = 1
    for sentence in word_lists:
        analyse = chinese_tagger.tag(sentence.split())
        str_tag = ""
        for tag_tuple in analyse:
            tag_list = [tag_char for tag_char in tag_tuple[1].split("#")]
            str_tag += tag_list[0] + "/" + tag_list[1] + " "
        nltk_all_tag.append(str_tag.strip())
        print("######LTP POSTagger finished " + str(flag) + " sentences")
        flag += 1
    print("NLTK POATagger is finished!!")

    return nltk_all_tag

# nltk的句法分析,filename作为备份文件不存进数据库
# filename最好是一个绝对路径比如 f:/data_backup/hpscorepos/hpscorepos_parser_tree 最后的文件名不用加后缀
def parser_nltk(word_lists, filename):
    os.environ['JAVAHOME'] = JAVA_PATH
    os.environ["STANFORD_PARSER"] = STANFORD_PARSER_PATH
    os.environ["STANFORD_MODELS"] = STANFORD_PARSER_MODELS
    chinese_parser = StanfordParser(model_path=nltk_parse_model_path)
    STANFORD_DIR = chinese_parser._classpath[0].rpartition('/')[0]
    chinese_parser._classpath = tuple(find_jars_within_path(STANFORD_DIR))
    chinese_parser.java_options = '-mx15000m'
    all_parser_sentence = []
    file = shelve.open(filename)
    flag = 0

    for sentence in word_lists:
        if sentence.strip()!="":
            res = list(chinese_parser.parse((sentence.strip()).split()))
            new_str = return_str_tofile(sentence_parse=str(res[0]))
            file[str(flag)] = res
            all_parser_sentence.append(new_str)
            flag += 1
            print("###### NLTK Dependency Parser Have finished "+ str(flag)+" sentences ###")
    return all_parser_sentence


# nltk依存句法分析
# filename最好是一个绝对路径比如 f:/data_backup/hpscorepos/hpscorepos_denpendency_parser_tree 最后的文件名不用加后缀，用作备份文件
def dependency_parser_nltk(word_lists,filename):
    os.environ['JAVAHOME'] = JAVA_PATH
    os.environ["STANFORD_PARSER"] = STANFORD_PARSER_PATH
    os.environ["STANFORD_MODELS"] = STANFORD_PARSER_MODELS
    chinese_parser = StanfordDependencyParser(model_path=nltk_parse_model_path)
    STANFORD_DIR = chinese_parser._classpath[0].rpartition('/')[0]
    chinese_parser._classpath = tuple(find_jars_within_path(STANFORD_DIR))
    chinese_parser.java_options = '-mx15000m'

    node_file = shelve.open(filename)
    all_dependency_list = []
    for index,sentence in enumerate(word_lists):
        if sentence.strip()!="":
            # 存进all_dependency_list中，存储数据类型是列表
            res = list(chinese_parser.parse(sentence.strip().split()))
            print("######we have finished ",str(index+1), " sentence!!!####")

            list_file = [triple for triple in res[0].triples()]
            all_dependency_list.append(list_file)

            #存进node_file，存储数据类型是dict/defaultdict,用作备份文件
            node_dict = {}
            node = res[0].nodes
            for inner_index in range(len(node.items()) * 2):
                if node[inner_index]['word'] != None or node[inner_index]['ctag'] != None:
                    # print(node[inner_index])
                    node_dict[node[inner_index]["address"]] = node[inner_index]
                    # print(node[inner_index]["address"], type(node[inner_index]["address"]))
            node_file[str(index)] = node_dict

    node_file.close()
    return all_dependency_list


# 一下部分进行测试
# part one 这一部分用于LTP词性标注。速度很快
def demo_pos_ltp():
    # 若数据库还没有创建，调用create_db方法创建
    # 创建数据库连接
    db = connect_db(db_name="preprocessed",user=user,password=password)
    db_new = connect_db(db_name="postagprecessed",user=user, password=password)

    # 在新数据库中创建8个新表
    a_bool = create_table(db_new,table="POS_hpscoreneg")
    b_bool = create_table(db_new, table="POS_hpscorepos")
    c_bool = create_table(db_new, table="POS_lenovoscoreneg")
    d_bool = create_table(db_new, table="POS_lenovoscorepos")

    e_bool = create_table(db_new, table="POS_hpscoreneg_tag")
    f_bool = create_table(db_new, table="POS_hpscorepos_tag")
    g_bool = create_table(db_new, table="POS_lenovoscoreneg_tag")
    h_bool = create_table(db_new, table="POS_lenovoscorepos_tag")

    # 创建成功
    if a_bool and b_bool and c_bool and d_bool and e_bool and f_bool and g_bool and h_bool:
        hp_neg = data_processing(db, "hpscoreneg", is_Nltk=False)
        hp_pos = data_processing(db, "hpscorepos", is_Nltk=False)
        lenovo_neg = data_processing(db, "lenovoscoreneg", is_Nltk=False)
        lenovo_pos = data_processing(db, "lenovoscorepos", is_Nltk=False)
        print(hp_neg)
        print(hp_pos)

        # #进行LTP词性标注
        hp_neg_list, hp_neg_tag = postagger_ltp(hp_neg)
        hp_pos_list, hp_pos_tag = postagger_ltp(hp_pos)
        lenovo_neg_list,lenovo_neg_tag = postagger_ltp(lenovo_neg)
        lenovo_pos_list, lenovo_pos_tag = postagger_ltp(lenovo_pos)
        # print(hp_neg_list)
        # print(hp_pos_list)
        print(hp_pos_tag)
        print(hp_neg_tag)
        #
        # # 向8个表中插入数据
        insert_table(db_new,table="POS_hpscoreneg", sentence_list=hp_neg_list)
        insert_table(db_new, table="POS_hpscorepos", sentence_list=hp_pos_list)
        insert_table(db_new, table="POS_lenovoscoreneg", sentence_list=lenovo_neg_list)
        insert_table(db_new, table="POS_lenovoscorepos", sentence_list=lenovo_pos_list)

        insert_table(db_new, table="POS_hpscoreneg_tag", sentence_list=hp_neg_tag)
        insert_table(db_new, table="POS_hpscorepos_tag", sentence_list=hp_pos_tag)
        insert_table(db_new, table="POS_lenovoscoreneg_tag", sentence_list=lenovo_neg_tag)
        insert_table(db_new, table="POS_lenovoscorepos_tag", sentence_list=lenovo_pos_tag)

    db.close()
    db_new.close()
    print("完成所有LTP 词性标注")

# 测试nltk句法分析
def demo_nltk_parser():
    print("使用的时候再去掉注释")
    # part2 用nltk进行句法分析
    # db = connect_db(db_name="preprocessed",user=user, password=password)
    # db_parser = connect_db(db_name="parse_processed",user=user, password=password)
    # a_bool = create_table(db_parser, table="Parse_hpscoreneg")
    # b_bool = create_table(db_parser, table="Parse_hpscorepos")
    # c_bool = create_table(db_parser, table="Parse_lenovoscoreneg")
    # d_bool = create_table(db_parser, table="Parse_lenovoscorepos")
    #
    #
    #
    # if a_bool and b_bool and c_bool and d_bool:
    #     # 获取数据
    #     hp_neg_parse = data_processing(db, "hpscoreneg", is_Nltk=True)
    #     hp_pos_parse = data_processing(db, "hpscorepos", is_Nltk=True)
    #     lenovo_neg_parse = data_processing(db, "lenovoscoreneg", is_Nltk=True)
    #     lenovo_pos_parse = data_processing(db, "lenovoscorepos", is_Nltk=True)
    #
    #     # 对句子进行句法分析nltk方法
    #     hp_neg_parse_list = parser_nltk(word_lists=hp_neg_parse,filename="F:/pycharm project/mysecondproject/细粒度情感分析/data_backup/nltk_parser/hp_neg_parser/hp_neg_parser")
    #     insert_table(db_parser, table="Parse_hpscoreneg", sentence_list=hp_neg_parse_list)
    #
    #     hp_pos_parse_list = parser_nltk(word_lists=hp_pos_parse,filename="F:/pycharm project/mysecondproject/细粒度情感分析/data_backup/nltk_parser/hp_pos_parser/hp_pos_parser")
    #     insert_table(db_parser, table="Parse_hpscorepos", sentence_list=hp_pos_parse_list)
    #
    #     lenovo_neg_parse_list = parser_nltk(word_lists=lenovo_neg_parse,filename="F:/pycharm project/mysecondproject/细粒度情感分析/data_backup/nltk_parser/lenovo_neg_parser/lenovo_neg_parser")
    #     insert_table(db_parser, table="Parse_lenovoscoreneg", sentence_list=lenovo_neg_parse_list)
    #
    #     lenovo_pos_parse_list = parser_nltk(word_lists=lenovo_pos_parse,filename="F:/pycharm project/mysecondproject/细粒度情感分析/data_backup/nltk_parser/lenovo_pos_parser/lenovo_pos_parser")
    #     insert_table(db_parser, table="Parse_lenovoscorepos", sentence_list=lenovo_pos_parse_list)
    #
    #     print("完成所有的nltk句法分析！")
    #
    # db.close()
    # db_parser.close()

def main():

    # 使用Factorted包 -->  chineseFactored.ser.gz --->放弃了  太慢了
    db = connect_db(db_name="preprocessed",user=user, password=password)
    db_parser = connect_db(db_name="parse_processed",user=user, password=password)
    a_bool = create_table(db_parser, table="dep_Parse_hpscoreneg")
    b_bool = create_table(db_parser, table="dep_Parse_hpscorepos")
    c_bool = create_table(db_parser, table="dep_Parse_lenovoscoreneg")
    d_bool = create_table(db_parser, table="dep_Parse_lenovoscorepos")



    if a_bool and b_bool and c_bool and d_bool:
        # 获取数据
        hp_neg_parse = data_processing(db, "hpscoreneg", is_Nltk=True)
        hp_pos_parse = data_processing(db, "hpscorepos", is_Nltk=True)
        lenovo_neg_parse = data_processing(db, "lenovoscoreneg", is_Nltk=True)
        lenovo_pos_parse = data_processing(db, "lenovoscorepos", is_Nltk=True)

        # 对句子进行句法分析nltk方法
        hp_neg_dep_parse_list = dependency_parser_nltk(word_lists=hp_neg_parse,filename="F:/pycharm project/mysecondproject/细粒度情感分析/data_backup/nltk_depenency_parser/hp_neg_dep_parser/hp_neg_dep_parser")
        insert_table(db_parser, table="dep_Parse_hpscoreneg", sentence_list=hp_neg_dep_parse_list)

        hp_pos_dep_parse_list = dependency_parser_nltk(word_lists=hp_pos_parse,filename="F:/pycharm project/mysecondproject/细粒度情感分析/data_backup/nltk_depenency_parser/hp_pos_dep_parser/hp_pos_dep_parser")
        insert_table(db_parser, table="dep_Parse_hpscorepos", sentence_list=hp_pos_dep_parse_list)

        lenovo_neg_dep_parse_list = dependency_parser_nltk(word_lists=lenovo_neg_parse,filename="F:/pycharm project/mysecondproject/细粒度情感分析/data_backup/nltk_depenency_parser/lenovo_neg_dep_parser/lenovo_neg_dep_parser")
        insert_table(db_parser, table="dep_Parse_lenovoscoreneg", sentence_list=lenovo_neg_dep_parse_list)

        lenovo_pos_dep_parse_list = dependency_parser_nltk(word_lists=lenovo_pos_parse,filename="F:/pycharm project/mysecondproject/细粒度情感分析/data_backup/nltk_depenency_parser/lenovo_pos_dep_parser/lenovo_pos_dep_parser")
        insert_table(db_parser, table="dep_Parse_lenovoscorepos", sentence_list=lenovo_pos_dep_parse_list)

        print("完成所有的nltk依存句法分析！")

    db.close()
    db_parser.close()



if __name__ == "__main__":
    main()