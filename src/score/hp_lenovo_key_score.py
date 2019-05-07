'''
Created on 2019年5月3日

@author: cws

处理hp和lenovo的属性得分
'''

import key_score


if __name__ == "__main__":
    hp_sl_file = open("sentiment_label/hp_dict_merge.txt", 'r', encoding='utf-8')
    hp_sl_dict = eval(hp_sl_file.read())
    hp_key_score_dict = key_score.key_score(hp_sl_dict)
    key_score.write_to_txt(hp_key_score_dict, "keys_score/hp_keys_score.txt")
    
    lenovo_sl_file = open("sentiment_label/lenovo_dict_merge.txt", 'r', encoding='utf-8')
    lenovo_sl_dict = eval(lenovo_sl_file.read())
    lenovo_key_score_dict = key_score.key_score(lenovo_sl_dict)
    key_score.write_to_txt(lenovo_key_score_dict, "keys_score/lenovo_keys_score.txt")
    
    print("\nfinish!!!")
    