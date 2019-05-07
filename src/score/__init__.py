# coding=utf-8

# from wordScore import word_score
# 
# degree_adverb_words_file = open("sentiment_label_dicts/degree_adverb_words.txt", 'r')
# degree_adverb_words = eval(degree_adverb_words_file.read())
# 
# 
# hp_sl_keys_list = []
# with open("sentiment_label/dict_merge_keys.txt", 'r') as fm:
#     for line in fm:
#         hp_sl_keys_list.append(line.strip('\n'))
#  
# hp_sl_scores_dict = {}
# hp_sl_scores_dict = hp_sl_scores_dict.fromkeys(hp_sl_keys_list)
#  
# hp_sl_file = open("sentiment_label/hp_dict_merge.txt", 'r', encoding='utf-8')
# hp_sl_dict = eval(hp_sl_file.read())
# dict = {}
# dict = dict.fromkeys(hp_sl_dict.keys())
# print(dict)
# for key in hp_sl_keys_list:
#     current_key_score = 0
#     minus_score_list = []
#     plus_score_list = []
#     minus_score_sum = 0
#     plus_score_sum = 0
#     for label in hp_sl_dict[key]:  # list指一个标签
#         current_label_score = 1
#         for value in label:
#             sentimentLabel = value.split('/')
#             if len(sentimentLabel) == 2:
#                 word = value.split('/')[0]
#                 wordLabel = value.split('/')[1]
#                 if wordLabel == "neg":
#                     current_label_score = current_label_score * (-1)
#                 elif wordLabel == "d":
#                     for scoreKey in degree_adverb_words.keys():
#                         if word in degree_adverb_words[scoreKey]:
#                             current_label_score = current_label_score * float(scoreKey)
#                 elif wordLabel == "a":
#                     current_label_score = current_label_score * word_score(word)
#             else:
#                 continue
#         if current_label_score < 0:
#             minus_score_list.append(current_label_score)
#         elif current_label_score >= 0:
#             plus_score_list.append(current_label_score)
#     if len(minus_score_list) > 0:
#         for minusScore in minus_score_list:
#             minus_score_sum += float(minusScore)
#         current_key_score += (minus_score_sum / len(minus_score_list))
#     if len(plus_score_list) > 0:
#         for plusScore in plus_score_list:
#             plus_score_sum += float(plusScore)
#         current_key_score += (plus_score_sum / len(plus_score_list))
#     hp_sl_scores_dict[key] = current_key_score
# 
# fw = open("keys_score/hp_keys_score.txt",'w+')
# fw.write(str(hp_sl_scores_dict))      #把字典转化为str
# fw.close()
# 
# print("\nfinish!!!")

# print(hp_sl_dict["内存"])
# lists = dic["内存"]
# print(hp_sl_dict["内存"][0][0].split('/')[1])
# print(lists[0][0].split('/')[0])
# file.close()


#     dict = {}
#     n = 0.5
#     with open("新建文本文档.txt",'r',encoding='gbk') as f:
#         for line in f:
#             line = line.strip('\n')
#             words = line.split(' ')
#             dict[str(n)] = words
#             n += 0.5
# #     print(dict)
#     fw = open("degree_adverb_words.txt",'w+')
#     fw.write(str(dict))      #把字典转化为str
#     fw.close()