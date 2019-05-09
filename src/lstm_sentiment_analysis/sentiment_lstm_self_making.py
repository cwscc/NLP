# -*- coding: utf-8 -*-

import yaml
import sys
import os

from sklearn.cross_validation import train_test_split  # 处理向量
import multiprocessing  # 多线程处理
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence  # 导入sequence模块，将用于截长补短让所有数字列表长度为100，tokenizer是用于建立字典的（英文）
from keras.models import Sequential  # 蛋糕架子
from keras.layers.embeddings import Embedding  # 词向量嵌入模块
from keras.layers.recurrent import LSTM  # 模块的LSTM层
from keras.layers.core import Dense, Dropout, Activation  # 用于添加层，dropout数据， 激活函数
from keras.models import model_from_yaml  # 用于存储模型数据

import jieba
import pandas as pd
import pymysql

np.random.seed(1337)  # 再现？？？？
sys.setrecursionlimit(1000000)  # 设置允许最大的迭代次数是 1千万次

# 设置参数
vocab_dim = 250  # 指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
maxlen = 100  # 字数的最大长度（原本100）
n_iterations = 15  # 迭代次数，默认为5
n_exposures = 5  # 对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
window_size = 5  # 表示当前词与预测词在一个句子中的最大距离是多少，默认为5
batch_size = 32  # 每批次运行的数据量大小，默认32
epochs = 10  # 默认10
input_length = 100  # 输入长度（原本100）
cpu_count = multiprocessing.cpu_count()  # cpu多线程个数


# 加载训练文件
def loadfile():
    """

    :return: 1. 数据语料集连接的向量  2.每条语料的标签数组，1的向量表示积极，0的向量表示消极
    """

    # 我们的资料，总共23183*2条数据
    conn = pymysql.connect(host="127.0.0.1", user="root", passwd="123456", db="preprocessing", charset='utf8')
    sqlneg = "select comment from allnegcomment where id between 1 and 16905 union \
    select comment from negcomment"
    sqlpos = "select comment from allposcomment union \
    select comment from poscomment where id between 1 and 6278"
    
    neg1 = pd.read_sql(sqlneg, conn)
    pos1 = pd.read_sql(sqlpos, conn)

    combined = np.concatenate((pos1.iloc[:, 0], neg1.iloc[:, 0]), axis=0)  # data
    y = np.concatenate((np.ones(len(pos1), dtype=int), np.zeros(len(neg1), dtype=int)))  # target 拼接数组，aixs参数0表示按照行向，1是纵向

    return combined, y

    # 原作者的资料
    # neg = pd.read_excel('F:/pycharm project/mysecondproject/Sentiment-Analysis-master/data/neg.xls', header=None, index=None)
    # pos = pd.read_excel('F:/pycharm project/mysecondproject/Sentiment-Analysis-master/data/pos.xls', header=None, index=None)
    #
    # combined = np.concatenate((pos[0], neg[0]))
    # y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))   #拼接数组，aixs参数0表示按照行向，1是纵向
    #
    # return combined, y


# 调整词典，可调节单个词语的词频，使其能（或不能）被分出来，调整分词效果
def adjust_jieba_dict(adjust_word_file):
    f = open(adjust_word_file, encoding='gbk')
    adjust_list = f.readlines()
    for i in adjust_list:
        jieba.suggest_freq(i.strip(), True)
    f.close()


# 先对句子进行分词， 并去掉换行符, 数据处理的不完整，后期再完善
def tokenizer(text):
    text = [jieba.lcut(document.replace("\n", '')) for document in text]
    return text


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None, combined=None):
    """


    :param model: 传进去的模型（词向量模型）
    :param combined: 数据集语料的对应向量
    :return: 1-创建的每个词对应的索引字典  2-创建一个词向量的字典  3-转换训练与测试数据集的字典
    """
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()  # 创建gensim字典
        # gensim_dict.doc2bow(model.vocab.keys(), allow_update=True)               #创建词袋,这是原来的代码有错，改成一下一行

#         for k,v in model.wv.vocab.items():
#             print(k, "------>", v)
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)

        w2_index = {v : k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
        w2_vec = {word : model[word] for word in w2_index.keys()}  # 所有频数超过10的词语的词向量

        def parse_dataset(combined):
            """词语变成了整数数字"""

            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2_index[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语索引为0
        combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 设置每个句子的最大长度为maxlen，低于的补0，多出的截断
        return w2_index, w2_vec, combined
    else:
        print('NO data provided....没有数据提供了，会发生错误')


# 创建词向量模型，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引，combined是分好词的列表或者矩阵
def word2vec_train(combined):
    model = Word2Vec(
                    size=vocab_dim,  # 指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百
#                     alpha=0.025,  # 学习速率
                    window=window_size,  # 表示当前词与预测词在一个句子中的最大距离是多少
                    min_count=n_exposures,  # 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
                    workers=cpu_count,  # workers参数控制训练的并行数，默认为3
#                     min_alpha=0.0001,
                    sg=1,  # sg=0 --> cbow算法(默认)   sg=1 --> skip-gram算法
                    iter=n_iterations,  # 迭代次数，默认为5
#                     sorted_vocab=1,  # 如果为1（defau·t），则在分配word index 的时候会先对单词基于频率降序排序
#                     batch_words=10000  # 每一批的传递给线程的单词的数量，默认为10000
                     )  
    model.build_vocab(combined)  # 创建字典
    model.train(combined, total_words=model.corpus_count, epochs=model.epochs)  #  jc此处改动，，训练模型
    path1 = os.path.abspath('.')  # 表示当前所处的文件夹的绝对路径
    model.save(path1 + '\\word2vec\\Word2vec_model_change1.pkl')  # 保存模型
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined


def get_data(index_dict, word_vectors, combined, y):

    n_symbols = len(index_dict) + 1  # 所有单词的缩阴数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词对应词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.1)
#     print(x_train.shape, y_train.shape)
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


# 定义网络结构，现在定义复杂一点的LSTM网络
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    # 隐藏层太多太慢了
    print("定义一个简单的Keras模型,后期再去搭建复杂的....")
    model = Sequential()  # 定义一个蛋糕架
    model.add(Embedding(output_dim=vocab_dim,  # 添加词向量嵌入层
                        input_dim=n_symbols,  # 词向量字典中的词数量
                        mask_zero=True,
                        weights=[embedding_weights],  # 词向量中的权重列表
                        input_length=input_length))
    model.add(Dropout(0.2))  # 原先0.2                     #在这一层dropout丢弃一些数据，避免模型过拟合
                                                            # 每次训练迭代时会随机地在神经网络中放弃20%的神经元
     
#     model.add(keras.layers.Flatten())
    # 建立第一层LSTM层
    model.add(LSTM(activation="tanh", return_sequences=True, units=256))  # 添加LSTM层
    model.add(Dropout(0.25))  # 原先0.25    # 在训练数据集过程中舍去50%的数据避免过拟合
                   # activation='sigmoid'))
                   # inner_activation='hard_sigmoid'))
     
#     model.add(LSTM(64, input_dim = 100, input_len = 100, return_sequences=True))
#     model.add(LSTM(50, activation='tanh',input_shape=(x_train.shape[0],y_train.shape[0]),return_sequences=True))
    # 建立第二层LSTM层
    model.add(LSTM(return_sequences=True, units=128))
    model.add(Dropout(0.25))  # 原先0.25
     
    # 建立第三层LSTM层
    model.add(LSTM(units=64))
     
#     print("no problems to solve!!!!!!! go on !")
#     # 建立第一层隐藏层
#     model.add(Dense(units=256,
#                     activation='relu'))
#     model.add(Dropout(0.1))  # 原先0.2
      
#     # 建立第二层隐藏层
#     model.add(Dense(units=128,
#                     activation='relu'))
#     model.add(Dropout(0.1))  # 原先0.2
#        
#     # 建立第三层隐藏层
#     model.add(Dense(units=64,
#                     activation='relu'))
#     model.add(Dropout(0.1))  # 原先0.3
#     
    #  建立输出层，Dense(1)表示只输出一种分类
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.add(Dense(units=1,
                    activation='sigmoid'))  # 输出多分类的时候可以改成 softmax（sigmoid一般是二分类多）
     
    print('开始编译compiling模型')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
     
    print('Training 开始训练...')
    # 开始传入要训练的数据，同时进行其他参数的设置
    # 用于训练一个固定迭代次数的模型。返回：记录字典，包括每一次迭代的训练误差率和验证误差率。
    model.fit(x_train,  # 训练数据
              y_train,  # 标签
              batch_size=batch_size,  # 每次训练和梯度更新块的大小（默认128）
              epochs=epochs,  # 迭代次数
              verbose=1,  # 进度表示方式。0表示不显示数据，1表示显示进度条，2表示只显示一个数据
            validation_split=0.1,
#             validation_data=(x_test, y_test),  # 被用来作为验证数据的(X, y)元组。会代替validation_split所划分的验证数据
            shuffle=True,
              )
     
    print('Evaluate...开始评估模型')
#     scores = cross_val_score(model, X=x_test, y=y_test, score='f1_macro', cv=10, verbose=1)
#     print("交叉验证结果（f1_macro）：" + scores.mean())
     
#     evaluate：展示模型在验证数据上的效果，和fit函数中的参数基本一致。返回：误差率或者是准确率元组（if show_accuracy=True）。
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
     
    yaml_string = model.to_yaml()
    with open("word2vec/lstm.yml", 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    # save_weights(fname, overwrite=False)：将所有层的权值保存为HDF5文件，fname:文件名；overwrite:如果已经存在，是否覆盖原文件。
    # 返回：如果overwrite=False并且fname已经存在，则会抛出异常。
    model.save_weights("word2vec/lstm.h5", overwrite=True)
    print('Test score:', score)

    # part one : 一个简单的lstm网络，只有一层隐藏层，每一层维数很低
#     print("定义一个简单的Keras模型,后期再去搭建复杂的....")
#     model = Sequential()    #定义一个蛋糕架
#     model.add(Embedding(output_dim=vocab_dim,               #添加词向量嵌入层
#                         input_dim=n_symbols,
#                         mask_zero=True,
#                         weights=[embedding_weights],
#                         input_length=input_length))
#     # model.add(keras.layers.Flatten())
#     model.add(LSTM(output_dim=50,                           #添加LSTM层
#                    activation="tanh",
#                    ))
#                    # activation='sigmoid'))
#                    #inner_activation='hard_sigmoid'))
#     print("no problems to solve!!!!!!! go on !")
#     model.add(Dropout(0.3))     #0.25                       #在训练数据集过程中舍去50%的数据避免过拟合
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#  
#     print('开始编译compiling模型')
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#  
#     print('Training 开始训练...')
#     #开始传入要训练的数据，同时进行其他参数的设置
#     model.fit(x_train,
#               y_train,
#               batch_size=batch_size,
#               nb_epoch=epochs,
#               verbose=1,
# #               validation_data=(x_test, y_test),
#               validation_split=0.1,
#               shuffle=True,
#               )
#  
#     print('Evaluate...开始评估模型')
#     score = model.evaluate(x_test, y_test, batch_size=batch_size)
#  
#     yaml_string = model.to_yaml()
#     with open("word2vec/lstm.yml", 'w') as outfile:
#         outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
#     model.save_weights( "word2vec/lstm.h5" )
#     print('Test score:', score)


# 训练模型，并进行保存
def train():
    print('loading data... 加载数据')
    combined, y = loadfile()
    print(len(combined), len(y))

    print('Tokenising...进行词语分词处理..')
    path1 = os.path.abspath('..')
    path2 = os.path.dirname(path1)
    path3 = os.path.abspath(path2 + '/连接词/connecting_words_find.txt')
    adjust_jieba_dict(path3)
    combined = tokenizer(combined)

    print("Training a word2vec model...开始训练词向量模型")
    index_dict, word_vectors, combined = word2vec_train(combined)  # 这个combined是分好词的列表/矩阵

    print("为词嵌入层设置好数组矩阵...")
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
#     print(x_train.shape, y_train.shape)

    print("开始训练lstm神经网络")
    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


# 往下都是预测的函数
def input_tranform(string):
    words = jieba.lcut(string)
    words = np.array(words).reshape(1, -1)
    model = Word2Vec.load("Word2vec/Word2vec_model_change1.pkl")
    _, _, combined = create_dictionaries(model, words)
    return combined


# 进行lstm预测
def lstm_sentence_predict(string):
    print("loading model加载模型...")
    with open('Word2vec/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)

    model = model_from_yaml(yaml_string)

    print("loading weights... 加载权重")
    model.load_weights("Word2vec/lstm.h5")
    model.compile(loss="binary_crossentropy",
                  optimizer='adam',
                  metrics=['accuracy'])
    data = input_tranform(string)
    data.reshape(1, -1)
    # predict_classes：用于对测试数据的分类预测，和evaluate函数中的参数一样。返回：对于测试数据的预测分类结果数组。
    result = model.predict_classes(data)
#     print(result)

    if result[0][0] == 1:
        return "positive"
    else:
        return "negative"

#     if result[0][0] == 1:
#         print(string, "positive")
#     else:
#         print(string, 'negative')


def lstm_batch_predict(db_user, db_password, db_name, sql):
    '''从数据库中读取数据进行批量预测
    
        Args:
            db_user: 数据库用户名
            db_password: 数据库密码
            db_name: 数据库名字
            sql: 查询语句
            
        Returns:
            predict_dict: 返回一个字典，predict_dict[0]为差评率，predict_dict[1]为好评率
        
    '''
    print("loading model加载模型...")
    with open('Word2vec/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)
    print("loading weights... 加载权重")
    model.load_weights("Word2vec/lstm.h5")
    model.compile(loss="binary_crossentropy",
                  optimizer='adam',
                  metrics=['accuracy'])
    
    predict_dict = {}
    predict_dict = predict_dict.fromkeys([0, 1])  # 0表示差评率，1表示好评率
    positiveCount = 0  # 记录好评数
    negativeCount = 0  # 记录差评数
    
    conn = pymysql.connect("localhost", db_user, db_password, db_name, charset='utf8')
    cursor = conn.cursor()
    cursor.execute(sql)
    data = cursor.fetchall()
    for row in data:  # row[0]为id，row[1]为comment
        sentence = input_tranform(row[1])
        sentence.reshape(1, -1)
        # predict_classes：用于对测试数据的分类预测，和evaluate函数中的参数一样。返回：对于测试数据的预测分类结果数组。
        result = model.predict_classes(sentence)
        if result[0][0] == 1:
            positiveCount += 1
#             print(row[1], "positive")
        if result[0][0] == 0:
            negativeCount += 1
#             print(row[1], "negative")
    predict_dict[0] = str(round(negativeCount / len(data), 2) * 100) + "%"  # 差评率
    predict_dict[1] = str(round(positiveCount / len(data), 2) * 100) + "%"  # 好评率
    return predict_dict


if __name__ == '__main__':

#     hpDict = lstm_batch_predict("root", "123456", "nlp", "select * from hpscorepos union \
#     select * from hpscoreneg")
#     lenovoDict = lstm_batch_predict("root", "123456", "nlp","select * from lenovoscorepos \
#     union select * from lenovoscoreneg")
#     print(hpDict)
#     print(lenovoDict)
#     fw = open("comment_tendency.txt", 'w+')
#     fw.write(str(hpDict))  # 把字典转化为str
#     fw.write(str(lenovoDict))
#     fw.close()
    
    # 训练模型
#     print("train...")
#     train()

#     string = '电脑质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'       #negative
#     string = "客服的态度真的很好，好的不得了，只是电脑的屏幕真的一般，运行内存16G但打起游戏来很卡，但是总体上说还不错"    #positive
#     string = '质量不错，是正品 ，安装师傅也很好，才要了83元材料费'      #positive
#     string = "显示器好评，键盘摸着舒服，但是客服态度真的有问题，问了几次都没有回我，老实说电脑是真心不错，给个好评吧"    #positive
#     string = "呵呵！这电脑真是服了，跑不动，性价比一般"     #negative
#     string = "满怀高兴回来的机子，拿到手不到24小时，彻底死机5回，就玩个dnf还捎带下点东西。。。这跑20w分是假的吧？我台式机没显卡也是i5都玩的杠杠的。各种多开。快递小哥是没得说。第一次一星，实话实说。"
    string = "对电脑本身还是很满意的，室友都说颜值不错，很好看，1.99mm，一开始以为很厚，可是拿到手的时候感觉还不错，问了客服很多问题，他也一一解答了。唯一不足的就是，物流，无缘无故的在长沙停留了那么久，毕竟五千多的东西，免不了担起心来，而且之所以这么急，也在于距离计算机考试就这么几天了。。。。。。。总体来说还不错，好评"
    print(lstm_sentence_predict(string))


############################################################
#     # 用人工标注的测试集测试模型准确率
#     print("loading model加载模型...")
#     with open('Word2vec/lstm.yml', 'r') as f:
#         yaml_string = yaml.load(f)
#                          
#     model = model_from_yaml(yaml_string)
#                          
#     print("loading weights... 加载权重")
#     model.load_weights("Word2vec/lstm.h5")
#     model.compile(loss="binary_crossentropy",
#                   optimizer='adam',
#                   metrics=['accuracy'])
#                  
#     # 测试集验证（test set）
#     print("开始验证测试集...")
#     conn = pymysql.connect("localhost", "root", "123456", "crawl", charset='utf8')
#     cursor = conn.cursor()
# #     sql = "select * from hpscorepos1"
#     sql = "select * from hpscorepos1 union select * from hpscoreneg1 union \
#     select * from lenovoscorepos1 union select * from lenovoscoreneg1"
#     cursor.execute(sql)
#     testSet = cursor.fetchall()
#     rightCount = 0
#     for row in testSet:  # row[0]为id，row[1]为comment，row[2]为人工标注的polarity
#         data = input_tranform(row[1])
#         data.reshape(1, -1)
#         # predict_classes：用于对测试数据的分类预测，和evaluate函数中的参数一样。返回：对于测试数据的预测分类结果数组。
#         result = model.predict_classes(data)
# #         print(result)
# #         result = lstm_predict(row[1])
#         if result[0][0] == 1 and row[2] == "P":
#             rightCount += 1
# #             print(row[1], "positive")
#         if result[0][0] == 0 and row[2] == "N":
#             rightCount += 1
# #             print(row[1], "negative")
#     print("The testSet's accuracy is:" + str(rightCount / len(testSet)))
#     print(rightCount)
#                               
#     # 关闭数据库连接
#     cursor.close()
#     conn.close()


####################### part one 原版分界线###################
# np.random.seed(1337)    #再现？？？？
# sys.setrecursionlimit(1000000)  #设置允许最大的迭代次数是 1千万次
#
# # 设置参数
# vocab_dim = 100     #字典的什么
# maxlen = 100        #字数的最大长度
# n_iterations = 1    #迭代次数？？？
# n_exposures = 10
# window_size = 7     #窗口大小？？
# batch_size = 32     #每批次运行的数据量大小
# n_epoch = 4
# input_length = 100  #输入长度
# cpu_count = multiprocessing.cpu_count()
#
#
# # 加载训练文件
# def loadfile():
#     """
#
#     :return: 1. 数据语料集连接的向量  2.每条语料的标签数组，1的向量表示积极，0的向量表示消极
#     """
#     neg = pd.read_excel('F:/pycharm project/mysecondproject/Sentiment-Analysis-master/data/neg.xls', header=None, index=None)
#     pos = pd.read_excel('F:/pycharm project/mysecondproject/Sentiment-Analysis-master/data/pos.xls', header=None, index=None)
#
#     combined = np.concatenate((pos[0], neg[0]))
#     y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))   #拼接数组，aixs参数0表示按照行向，1是纵向
#
#     return combined, y
#
# # 先对句子进行分词， 并去掉换行符, 数据处理的不完整，后期再完善
# def tokenizer(text):
#
#     text = [jieba.lcut(document.replace("\n", '')) for document in text]
#     return text
#
# # 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
# def create_dictionaries(model=None, combined=None):
#     """
#
#
#     :param model: 传进去的模型（词向量模型）
#     :param combined: 数据集语料的对应向量
#     :return: 1-创建的每个词对应的索引字典  2-创建一个词向量的字典  3-转换训练与测试数据集的字典
#     """
#     if (combined is not None) and (model is not None):
#         gensim_dict = Dictionary()          #创建gensim字典
#         gensim_dict.doc2bow(model.vocab.keys(), allow_update=True)               #创建词袋
#
#         w2_index = {v : k+1 for k,v in gensim_dict.items()}                     #所有频数超过10的词语的索引
#         w2_vec = {word : model[word] for word in w2_index.keys()}               #所有频数超过10的词语的词向量
#
#         def parse_dataset(combined):
#             """词语变成了整数数字"""
#
#             data = []
#             for sentence in combined:
#                 new_txt = []
#                 for word in sentence:
#                     try:
#                         new_txt.append(w2_index[word])
#                     except:
#                         new_txt.append(0)
#                 data.append(new_txt)
#             return data
#         combined = parse_dataset(combined)
#         #每个句子所含词语对应的索引，所以句子中含有频数小于10的词语索引为0
#         combined = sequence.pad_sequences(combined, maxlen=maxlen)  #设置每个句子的最大长度为maxlen，低于的补0，多出的截断
#         return w2_index, w2_vec, combined
#     else:
#         print('NO data provided....没有数据提供了，会发生错误')
#
#
# # 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
# def word2vec_train(combined):
#     model = Word2Vec(size=vocab_dim,        #创建词向量模型
#                      min_count=n_exposures,
#                      window=window_size,
#                      workers=cpu_count,
#                      iter=n_iterations)
#     model.build_vocab(combined)
#     model.train(combined)
#     model.save('F:/pycharm project/mysecondproject/Sentiment-Analysis-master/lstm_make_by_self_data/Word2vec_model.pkl')
#     index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
#     return index_dict, word_vectors, combined
#
# def get_data(index_dict, word_vectors, combined, y):
#
#     n_symbols = len(index_dict) + 1  #所有单词的缩阴数，频数小于10的词语索引为0，所以加1
#     embedding_weights = np.zeros((n_symbols, vocab_dim))    #索引为0的词语，词向量全为0
#     for word, index in index_dict.items():      #从索引为1的词语开始，对每个词对应词向量
#         embedding_weights[index, :] = word_vectors[word]
#     x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
#     print(x_train.shape, y_train.shape)
#     return n_symbols, embedding_weights, x_train, y_train, x_test, y_test
#
# # 定义网络结构
# def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
#     print("定义一个简单的Keras模型,后期再去搭建复杂的....")
#     model = Sequential()    #定义一个蛋糕架
#     model.add(Embedding(output_dim=vocab_dim,               #添加词向量嵌入层
#                         input_dim=n_symbols,
#                         mask_zero=True,
#                         weights=[embedding_weights],
#                         input_length=input_length))
#     model.add(LSTM(output_dim=50,                           #添加LSTM层
#                    activation='sigmoid',
#                    inner_activation='hard_sigmoid'))
#     model.add(Dropout(0.5))                                 #在训练数据集过程中舍去50%的数据避免过拟合
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#
#     print('开始编译compiling模型')
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#
#     print('Training 开始训练...')
#     #开始传入要训练的数据，同时进行其他参数的设置
#     model.fit(x_train,
#               y_train,
#               batch_size=batch_size,
#               nb_epoch=n_epoch,
#               verbose=1,
#               validation_data=(x_test, y_test),
#               show_accuracy=True)
#
#     print('Evaluate...开始评估模型')
#     score = model.evaluate(x_test, y_test, batch_size=batch_size)
#
#     yaml_string = model.to_yaml()
#     with open("F:/pycharm project/mysecondproject/Sentiment-Analysis-master/lstm_make_by_self_data/lstm.yml", 'w') as outfile:
#         outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
#     model.save_weights( "F:/pycharm project/mysecondproject/Sentiment-Analysis-master/lstm_make_by_self_data/lstm.h5" )
#     print('Test score:', score)
#
# # 训练模型，并进行保存
# def train():
#     print('loading data... 加载数据')
#     combined, y = loadfile()
#     print(len(combined), len(y))
#
#     print('Tokenising...进行词语分词处理..')
#     combined = tokenizer(combined)
#
#     print("Training a word2vec model...开始训练词向量模型")
#     index_dict, word_vectors, combined = word2vec_train(combined)
#
#     print("为词嵌入层设置好数组矩阵...")
#     n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
#     print(x_train.shape, y_train.shape)
#
#     print("开始训练lstm神经网络")
#     train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)
#
# def input_tranform(string):
#     words = jieba.lcut(string)
#     words = np.array(words).reshape(1,-1)
#     model = Word2Vec.load("F:/pycharm project/mysecondproject/Sentiment-Analysis-master/lstm_make_by_self_data/Word2vec_model.pkl")
#     _, _, combined = create_dictionaries(model, words)
#     return combined
#
# # 进行lstm预测
# def lstm_predict(string):
#     print("loading model加载模型...")
#     with open('F:/pycharm project/mysecondproject/Sentiment-Analysis-master/lstm_make_by_self_data/lstm.yml', 'r') as f:
#         yaml_string = yaml.load(f)
#
#     model = model_from_yaml(yaml_string)
#
#     print("loading weights... 加载权重")
#     model.load_weights("F:/pycharm project/mysecondproject/Sentiment-Analysis-master/lstm_make_by_self_data/lstm.h5")
#     model.compile(loss="binary_crossentropy",
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     data = input_tranform(string)
#     data.reshape(1, -1)
#     result = model.predict_classes(data)
#
#     if result[0][0] == 1:
#         print(string, "positive")
#     else:
#         print(string, 'negative')
#
# if __name__ == '__main__':
#     print
#     string = '手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'
#     lstm_predict(string)
