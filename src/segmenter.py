'''
Author: dogeee-debug gxyhome030404@gmail.com
Date: 2025-11-21 22:55:17
LastEditors: dogeee-debug gxyhome030404@gmail.com
LastEditTime: 2025-11-22 15:23:30
Description: 实现最大正向匹配分词，导入jieba接口
'''

import jieba

def fmm_segment(sentence, dictionary=None, max_len=5):
    result = []
    i = 0
    n = len(sentence)
    while i < n:
        matched = False
        for l in range(max_len, 0, -1):
            if i + l > n:
                continue
            piece = sentence[i:i+l]
            if dictionary and piece in dictionary:
                result.append(piece)
                i += l
                matched = True
                break
        if not matched:
            result.append(sentence[i])
            i += 1
    return result

def jieba_segment(sentence):
    return list(jieba.cut(sentence))
