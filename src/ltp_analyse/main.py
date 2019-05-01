
import jieba
import re
import os
from pyltp import Postagger
from pyltp import Segmentor
from pyltp import Parser

'''
分词、去停用词

'''
# LTP模型路径
LTP_DATA_DIR = 'E:/ltp_data_v3.4.0/'  # ltp模型目录的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
seg_model_path = os.path.join(LTP_DATA_DIR,'cws.model') #分词模型

#处理空白字符和&;
def sentence_process(sentence):
    s1 = re.compile(r'(?<=&).+?(?=;)')
    #s2 = re.compile(r'\s')
    s3 = re.compile(r'[\s|\uff1f|\uff01|?!.]')
    s4 = re.compile(r'[\uff1b|,;]')
    sentence = s1.sub('', sentence)
    sentence = s3.sub('。',sentence)
    sentence = s4.sub('，',sentence)

    return sentence


# 移除标点符号，删除并列的重复标点符号，头尾的标点符号，栈的思想
def remove_duplicate_dot(sentence):
    sentence_list = sentence.split(" ") #以空格分隔
    #sentence_list = sentence
    no_dupli_list = []
    for word_index,word in enumerate(sentence_list):
        if word.strip() != "":
            if word != "，":
                no_dupli_list.append(word)
            elif word == "，":
                if len(no_dupli_list) != 0 and no_dupli_list[-1] != "，":
                    no_dupli_list.append("，")
                elif len(no_dupli_list) != 0 and no_dupli_list[-1] == "，":
                    no_dupli_list.append("，")
                    no_dupli_list.pop()

    if len(no_dupli_list)!=0 and (no_dupli_list[-1] == "，" or no_dupli_list[-1].isdigit()):
        no_dupli_list.pop()

    no_dupli_str = ' '.join(no_dupli_list)  #空格分隔

    return no_dupli_str


# 创建停用词list
def stopwordslist(filepath, encode="utf-8"):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding=encode).readlines()]
    return stopwords


# 调整词典，可调节单个词语的词频，使其能（或不能）被分出来，调整分词效果
def adjust_jieba_dict(adjust_word_file):
    f = open(adjust_word_file, encoding='utf-8')
    adjust_list = f.readlines()
    for i in adjust_list:
        jieba.suggest_freq(i.strip(), True)

    f.close()

def seg_sentence_new(sentence):

    adjust_jieba_dict("connect_word_zhuang.txt") #调整jieba词频
    adjust_jieba_dict("adv_dic.txt")

    sentence_seged = jieba.cut(sentence.strip())  # 生成一个生成器

    stopwords = stopwordslist('stop_words_main.txt',encode='utf-8')  # 主要的停用词（修改过的）
    stopwords_extra = stopwordslist('stop_words_punctuation.txt',encode='utf-8') #一些符号、颜表情
    stopword_replace = stopwordslist("stop_words_replace.txt",encode='utf-8') #用来替换的标点符号
    # 停用词合并
    for i in stopwords_extra:
        stopwords.append(i)
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\s' and word != '\b':
                if word in stopword_replace:
                    outstr += "， "
                else:
                    outstr += word
                    outstr += " "
    outstr = remove_duplicate_dot(outstr)  # 去掉标点
    word_list = outstr.strip().split(" ")
    return word_list


# word_list是嵌套列表，包含每一个干净的句子的列表
# LTP词性标注
def postagger_ltp(words):
    postagger = Postagger()     #初始化实例
    postagger.load(pos_model_path)  #加载模型
    postags = postagger.postag(words)
    tag_list = list(postags)
    postagger.release()    #释放模型
    out = ''
    for i in range(len(tag_list)):
        out += words[i]+"/"+tag_list[i]+" "
    print("词性标注结果:",out)
    return tag_list                #返回所有句子以及每个句子的词性列表


#获取属性词N列表
def get_dict(filename):
    f = open(filename, 'r', encoding='utf-8')
    dict = []
    for noun in f.readlines():
        noun = noun.strip()
        dict.append(noun)
    f.close()
    return dict

def ltp_seg_sentiment_word(one_list):
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(seg_model_path)  # 加载模型
    adv_dic = get_dict('adv_dic.txt')
    neg_dic = get_dict('neg_dic.txt')
    relist = []
    for each in one_list:
        wordlist = each.split("+")
        target = wordlist[-1]
        target = target.split('/')[0]
        wordlist = wordlist[0:-1]
        flag = len(target)
        for i in adv_dic:
            if (target.find(i.strip()) != -1):
                word_split = segmentor.segment(target)
                target = list(word_split)
                break;
        flag1 = len(target)
        if (flag1 == 3):
            if(target[0] in neg_dic):
                if(target[1] in neg_dic):
                    target[0] += '/neg'
                    target[1] += '/neg'
                else:
                    target[0] += '/neg'
                    target[1] += '/d'
            else:
                target[0] += '/d'
                target[1] += '/d'
            target[2] += '/a'
        elif (flag1 == 2):
            if(target[0] in neg_dic):
                target[0] += '/neg'
            else:
                target[0] +='/d'
            target[1] += '/a'
        else:
            target[0] += '/a'
        for index in range(flag1):
            wordlist.append(target[index])
        relist.append(wordlist)
    segmentor.release()
    return relist


#  LTP依存句法分析,返回列表(重写)

def parser_ltp_arc(word_list, tag_list):
    parser = Parser()   #初始化实例
    parser.load(par_model_path)
    arcs = parser.parse(word_list, tag_list)     #依存句法分析，一条评论的依存句法分析
    parser.release()
    return arcs


def extract_noun_base_pager_rule(sentence, arcs, pos):
    all_list = [] #所有评论的名词
    noun_list = get_dict('CNoun.txt')
    verb_list = get_dict('CVerb.txt')
    adv_dic = get_dict('adv_dic.txt')
    neg_dic = get_dict('neg_dic.txt')

    count = 0
    one_list = [] #存放每条评论抽出来的词对
    count += 1
    for flag in range(len(sentence)):
        mark4 = False  # 是否符合四维规则
        head = arcs[flag].head-1 #head表示依存弧的父节点词的数组下标
        if arcs[flag].relation == "SBV":
            # SBV关系弧中 (n ← a)
            phrase = ""
            if(pos[flag] == "n" and pos[head] == "a"
                                and abs(head-flag) <= 3 and sentence[flag] in noun_list):
                phrase += ('+').join((sentence[flag]+'/n',sentence[head]+'/a'))
                #################下一层 ###################
                for flag1 in range(len(sentence)):
                    head1 = arcs[flag1].head-1 #同上
                    #查找nna  (n1 ← n ← a)
                    if(arcs[flag1].relation == "ATT"  and pos[flag1] == "n"
                                                      and head1 == flag
                                                      and flag1<flag ):
                        phrase = ""
                        phrase += ('+').join((sentence[flag1]+'/n'+sentence[flag]+'/n',sentence[head]+'/a'))
                        #################下一层 ###################
                        for flag2 in range(len(sentence)):
                            head2 = arcs[flag2].head-1
                            #查找 nnda (n1 ← n ← a)&&(d ← a)
                            if(arcs[flag2].relation == "ADV" and pos[flag2] == "d"
                                                             and head2 == head
                                                             and flag2<head
                                                             and flag<flag2):
                                phrase = ""
                                if(sentence[flag2] in neg_dic):
                                    phrase += ('+').join(
                                        (sentence[flag1]+'/n'+sentence[flag]+'/n',sentence[flag2]+'/neg',sentence[head]+'/a'))
                                else:
                                    if(sentence[flag2] in adv_dic):
                                        phrase += ('+').join(
                                            (sentence[flag1]+'/n'+sentence[flag]+'/n', sentence[flag2]+'/d',sentence[head]+'/a'))
                                    else:
                                        phrase += ('+').join(
                                            (sentence[flag1] + '/n'+ sentence[flag] + '/n',sentence[head] + '/a'))
                                mark4 = True #符合四维规则，标记为真
                        one_list.append(phrase)
                        phrase = ""
                    #查找vna (v ← n ← a)
                    elif (arcs[flag1].relation == "ATT" and pos[flag1] == "v"
                                                        and head1 == flag
                                                        and flag1 < flag
                                                        and mark4 == False):
                        phrase = ""
                        phrase += ('+').join((sentence[flag1]+'/v'+sentence[flag]+'/n',sentence[head]+'/a'))
                        #################下一层 ###################
                        for flag2 in range(len(sentence)):
                            head2 = arcs[flag2].head - 1
                            #查找vnda (v ← n ← a)&&(d ← a)
                            if(arcs[flag2].relation == "ADV" and pos[flag2] == "d"
                                                             and head2 == head
                                                             and flag2 < head
                                                             and flag < flag2):
                                phrase = ""
                                if (sentence[flag2] in neg_dic):
                                    phrase += ('+').join(
                                        (sentence[flag1] + '/v'+sentence[flag]+'/n', sentence[flag2] + '/neg',
                                         sentence[head] + '/a'))
                                else:
                                    if(sentence[flag2] in adv_dic):
                                        phrase += ('+').join(
                                            (sentence[flag1] + '/v'+sentence[flag] + '/n', sentence[flag2] + '/d',
                                             sentence[head] + '/a'))
                                    else:
                                        phrase += ('+').join(
                                            (sentence[flag1] + '/v'+sentence[flag] + '/n',sentence[head] + '/a'))
                                mark4 = True #符合四维规则，标记为真
                        one_list.append(phrase)
                        phrase = ""
                    #查找nda (n ← a)&&(d ← a)
                    elif(arcs[flag1].relation == "ADV"  and pos[flag1] == "d"
                                                        and head1 == head
                                                        and flag < flag1
                                                        and mark4 == False):
                        phrase = ""
                        if(sentence[flag1] in neg_dic):
                            phrase += ('+').join((sentence[flag]+'/n', sentence[flag1]+'/neg',sentence[head]+'/a'))
                        else:
                            if(sentence[flag1] in adv_dic):
                                phrase += ('+').join((sentence[flag] + '/n', sentence[flag1] + '/d', sentence[head] + '/a'))
                            else:
                                phrase += ('+').join(
                                    (sentence[flag] + '/n',sentence[head] + '/a'))

                        #################下一层 ###################
                        for flag2 in range(len(sentence)):
                            head2 = arcs[flag2].head-1
                            #查找ndda (n ← a)&&(d1 ← a)&&(d2 ← a)
                            if (arcs[flag2].relation == "ADV" and pos[flag2] == "d"
                                                              and head2 == head
                                                              and flag1<flag2):
                                phrase = ""
                                if(sentence[flag1] in neg_dic):
                                    if(sentence[flag2] in neg_dic):
                                        phrase += ('+').join(
                                            (sentence[flag]+'/n',sentence[flag1]+'/neg',sentence[flag2]+'/neg',sentence[head]+'/a'))
                                    else:
                                        if(sentence[flag2] in adv_dic):
                                            phrase += ('+').join(
                                                (sentence[flag] + '/n', sentence[flag1] + '/neg',
                                                 sentence[flag2] + '/d', sentence[head] + '/a'))
                                        else:
                                            phrase += ('+').join(
                                                (sentence[flag] + '/n', sentence[flag1] + '/neg',sentence[head] + '/a'))
                                else:
                                    if(sentence[flag1] in adv_dic):
                                        if(sentence[flag2] in adv_dic):
                                            phrase += ('+').join(
                                                (sentence[flag] + '/n', sentence[flag1] + '/d', sentence[flag2] + '/d',
                                                 sentence[head] + '/a'))
                                        else:
                                            phrase += ('+').join(
                                                (sentence[flag] + '/n', sentence[flag1] + '/d',sentence[head] + '/a'))
                                    else:
                                        phrase += ('+').join(
                                            (sentence[flag] + '/n',sentence[head] + '/a'))
                                mark4 = True
                        one_list.append(phrase)
                        phrase = ""
                #没找到符合二维以上的规则时
                if(phrase!=""):one_list.append(phrase)
            #查找ni(n ← i)
            elif (pos[flag] == "n" and pos[head] == "i"
                                   and abs(head-flag) <= 3 and sentence[flag] in noun_list):
                phrase = ""
                phrase += ('+').join((sentence[flag]+'/n', sentence[head]+'/a'))
                #################下一层 ###################
                for flag1 in range(len(sentence)):
                    #查找 nni
                    if (arcs[flag1].relation == "ATT" and pos[flag1] == "n"
                                                      and head == flag
                                                      and flag1 < flag):
                        phrase = ""
                        phrase += ('+').join((sentence[flag1]+'/n', sentence[flag]+'/n',sentence[head]+'/a'))
                one_list.append(phrase)
            #查找nd (n ← d)
            elif (pos[flag] == "n" and pos[head] == "d"
                                   and abs(head-flag) <= 3 and sentence[flag] in noun_list):
                phrase = ""
                phrase += ('+').join((sentence[flag]+'/n',sentence[head]+'/a'))
                one_list.append(phrase)
            # 查找va (v ← a)
            elif (pos[flag] == "v" and pos[head] == "a"
                                   and abs(head-flag)<=3 and sentence[flag] in verb_list):
                phrase = ""
                phrase += ('+').join((sentence[flag]+'/v' ,sentence[head]+'/a'))
                #################下一层 ###################
                for flag1 in range(len(sentence)):
                    head1 = arcs[flag1].head - 1
                    #查找nva (n ← v ← a)
                    if(arcs[flag1].relation == "ATT"  and pos[flag1] == "n"
                                                      and head1 == flag
                                                      and flag1 < flag ):
                        phrase = ""
                        phrase += ('+').join((sentence[flag1]+'/n'+sentence[flag]+'/v',sentence[head]+'/a'))
                        ################下一层 ###################
                        for flag2 in range(len(sentence)):
                            head2 = arcs[flag2].head - 1
                            #查找nvda (n ← v ← a)&&(d ← a)
                            if(arcs[flag2].relation == "ADV" and pos[flag2] == "d"
                                                             and head2 == head):
                                phrase = ""
                                if(sentence[flag2] in neg_dic):
                                    phrase += ('+').join(
                                        (sentence[flag1]+'/n'+sentence[flag]+'/v',sentence[flag2]+'/neg',sentence[head]+'/a'))
                                else:
                                    if(sentence[flag2] in adv_dic):
                                        phrase += ('+').join(
                                            (sentence[flag1] + '/n'+sentence[flag] + '/v', sentence[flag2] + '/d',
                                             sentence[head] + '/a'))
                                    else:
                                        phrase += ('+').join(
                                            (sentence[flag1] + '/n'+sentence[flag] + '/v',sentence[head] + '/a'))
                                mark4 = True #符合四维规则，标记为真
                        ################下一层 ###################
                        phrase = ""
                    #查找vda(v ← a) && ( d ← a)
                    elif (arcs[flag1].relation == "ADV" and pos[flag1] == "d"
                                                        and head1 == head
                                                        and flag1 < head
                                                        and mark4 == False):
                        phrase = ""
                        if(sentence[flag1] in neg_dic):
                            phrase += ('+').join((sentence[flag]+'/v',sentence[flag1]+'/neg',sentence[head]+'/a'))
                        else:
                            if(sentence[flag1] in adv_dic):
                                phrase += ('+').join(
                                    (sentence[flag] + '/v', sentence[flag1] + '/d', sentence[head] + '/a'))
                            else:
                                phrase += ('+').join(
                                    (sentence[flag] + '/v', sentence[head] + '/a'))
                        for flag2 in range(len(sentence)):
                            head2 = arcs[flag2].head - 1
                            # 查找vdda (v ← a)&&(d1 ← a)&&(d2 ← a)
                            if (arcs[flag2].relation == "ADV" and pos[flag2] == "d"
                                    and head2 == head
                                    and flag1 < flag2):
                                phrase = ""
                                if(sentence[flag1] in neg_dic):
                                    if(sentence[flag2] in neg_dic):
                                        phrase += ('+').join(
                                            (sentence[flag]+'/v', sentence[flag1]+'/neg', sentence[flag2]+'/neg', sentence[head]+'/a'))
                                    else:
                                        if(sentence[flag2] in adv_dic):
                                            phrase += ('+').join(
                                                (sentence[flag] + '/v', sentence[flag1] + '/neg',
                                                 sentence[flag2] + '/d', sentence[head] + '/a'))
                                        else:
                                            phrase += ('+').join(
                                                (sentence[flag] + '/v', sentence[flag1] + '/neg',sentence[head] + '/a'))
                                else:
                                    if(sentence[flag1] in adv_dic):
                                        if(sentence[flag2] in adv_dic):
                                            phrase += ('+').join(
                                                (sentence[flag] + '/v', sentence[flag1] + '/d', sentence[flag2] + '/d',
                                                 sentence[head] + '/a'))
                                        else:
                                            phrase += ('+').join(
                                            (sentence[flag] + '/v', sentence[flag1] + '/d',sentence[head] + '/a'))
                                    else:
                                        phrase += ('+').join(
                                            (sentence[flag] + '/v',sentence[head] + '/a'))
                                mark4 = True
                        one_list.append(phrase)
                        phrase = ""
                # 没找到符合二维以上的规则时
                if (phrase != ""):one_list.append(phrase)

        elif arcs[flag].relation == "CMP":
            phrase = ""
            #查找CMP va (v → a)
            if (pos[flag] == "a" and pos[head] == "v"
                                 and abs(flag-head) <=3 and sentence[head] in verb_list):
                phrase +=('+').join((sentence[head]+'/v',sentence[flag]+'/a'))
                for flag1 in range(len(sentence)):
                    head1 = arcs[flag1].head - 1
                    # 查找vda(CMP) (v ← a)&&(d ← a)
                    if(arcs[flag1].relation == "ADV" and pos[flag1] == "d"
                                                       and head1 == flag):
                        phrase = ""
                        if(sentence[flag1] in neg_dic):
                            phrase += ('+').join((sentence[head]+'/v',sentence[flag1]+'/neg', sentence[flag]+'/a'))
                        else:
                            if(sentence[flag1] in adv_dic):
                                phrase += ('+').join(
                                    (sentence[head] + '/v', sentence[flag1] + '/d', sentence[flag] + '/a'))
                            else:
                                phrase += ('+').join((sentence[head] + '/v', sentence[flag] + '/a'))
                one_list.append(phrase)
            #查找vd(CMP) (v → d)
            elif(pos[flag] =="d" and pos[head] == "v"
                                 and abs(flag-head) <= 3 and sentence[head] in verb_list):
                phrase = ""
                phrase += ('+').join((sentence[head]+'/v',sentence[flag]+'/a'))
                one_list.append(phrase)

    one_list = list(set(one_list)) #去重

    one_list = ltp_seg_sentiment_word(one_list)

    return one_list

def sentence_analyse(sentence):
    sentence_after_process = sentence_process(sentence)
    print('预处理后的评论：',sentence_after_process)

    word_list = seg_sentence_new(sentence_after_process)
    print("分词列表：",word_list)

    tag_list = postagger_ltp(word_list)

    arcs = parser_ltp_arc(word_list, tag_list)

    result = extract_noun_base_pager_rule(word_list, arcs, tag_list)

    dic = dict()
    for each in result:
        opinion_target = each[0].split('/')[0]
        if (opinion_target not in dic.keys()):
            dic[opinion_target] = []
        dic[opinion_target].append(each[1:])

    return dic

if __name__ == '__main__':

    sentence = '外观确实非常漂亮，质感细腻，收感板扎。不足的是word打开后时常会闪退，不知道是什么原因...其它的都还好，同时打开CAD、和ps外加一个游戏和word毫无卡顿。发热一般，usb接口太少而且插u盘有点费力。'
    sentence = '质量非常不错，手感挺好，运行流畅，散热及时，好产品？？？？？'

    dict_for_score = sentence_analyse(sentence)

    print("可供计算得分的字典；",dict_for_score)



