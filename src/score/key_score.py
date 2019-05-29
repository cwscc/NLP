'''
Created on 2019年5月5日

@author: cws

处理单个标签得分
'''

from wordScore import word_score


def get_degree_adverb_words():
    '''读取程度副词评分词典
    
    Returns:
                    程度副词评分字典
    
    '''
    degree_adverb_words_file = open("sentiment_label_dicts/degree_adverb_words.txt", 'r')
    degree_adverb_words = eval(degree_adverb_words_file.read())
    return degree_adverb_words


def write_to_txt(dict, filePath):
    '''将字典写入txt
    
    Args:
        dict: 需要写入txt的字典
        filePath: txt的文件路径
    
    '''
    fw = open(filePath, 'w+')
    fw.write(str(dict))  # 把字典转化为str
    fw.close()

    
def key_score(dict):
    '''计算属性词得分
    
    Args:
        dict: 需要评分的情感标签字典
        
    Returns:
                    属性词得分字典

        属性词评分方法：计算当前属性词的所有标签得分，标签得分有正有负，最终得分的公式为：
                                     当前属性词得分   = （所有负得分标签之和  / 负得分标签个数）   + （所有正得分标签之和  / 正得分标签个数）
    
        标签评分方法：否定词为-1，
                                  程度副词分为四个等级，在程度副词词典（"sentiment_label_dicts/degree_adverb_words.txt"）中有说明，
                                  形容词由工具包（"wordScore.py"）计算得到。
                                 公式为：标签得分  = 否定词得分  * 程度副词得分  * 形容词得分
    '''
    degree_adverb_words = get_degree_adverb_words()
    key_score_dict = {}
    key_score_dict = key_score_dict.fromkeys(dict.keys())  # 以需要评分的属性词为key值构建评分字典
    for key in dict.keys():  # key即属性词
        current_key_score = 0  # 当前key的得分
        minus_score_list = []  # 存储负分值
        plus_score_list = []  # 存储正分值
        minus_score_sum = 0  # 所有负得分标签之和
        plus_score_sum = 0  # 所有正得分标签之和
        for label in dict[key]:  # label指一个标签
            current_label_score = 1  # 当前标签的分值，默认为1
            for value in label:
                sentimentLabel = value.split('/')
                if len(sentimentLabel) == 2:
                    word = value.split('/')[0]
                    wordLabel = value.split('/')[1]
                    if wordLabel == "neg":
                        current_label_score = current_label_score * (-1)
                    elif wordLabel == "d":
                        for scoreKey in degree_adverb_words.keys():
                            if word in degree_adverb_words[scoreKey]:
                                current_label_score = current_label_score * float(scoreKey)
                    elif wordLabel == "a":
                        current_label_score = current_label_score * word_score(word)
                else:
                    continue
            if current_label_score < 0:
                minus_score_list.append(current_label_score)
            elif current_label_score >= 0:
                plus_score_list.append(current_label_score)
        if len(minus_score_list) > 0:
            for minusScore in minus_score_list:
                minus_score_sum += float(minusScore)
            current_key_score += (minus_score_sum / len(minus_score_list))
        if len(plus_score_list) > 0:
            for plusScore in plus_score_list:
                plus_score_sum += float(plusScore)
            current_key_score += (plus_score_sum / len(plus_score_list))
        key_score_dict[key] = current_key_score
    return key_score_dict


if __name__ == '__main__':
    dict = {'内存': [['略/d', '小/a'], ['很/d', '给力/a']], '屏幕':[['一般/a']]}
    dict2 = {'内存': [['略/d', '小/a'], ['很/d', '给力/a']], '屏幕':[['清晰/a']]}
    result = key_score(dict)
    print(result)  # dict:{'内存': 0.12476103038762892, '屏幕': 0.006418219461697721}
                   # dict2:{'内存': 0.12476103038762892, '屏幕': 0.9751552795031055}
    
