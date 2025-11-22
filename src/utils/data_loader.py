'''
Author: dogeee-debug gxyhome030404@gmail.com
Date: 2025-11-21 22:55:17
LastEditors: dogeee-debug gxyhome030404@gmail.com
LastEditTime: 2025-11-22 16:34:18
Description: 解析处理好的两个json文件
'''

import json
import os
import random

def load_processed_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            tokens = item['tokens']
            labels = item['labels']
            sent = list(zip(tokens, labels)) 
            data.append(sent)
    return data

def process_raw_data(raw_path, save_train, save_dev, dev_ratio=0.1):
    """
    处理原始 train.txt 数据，生成 train.json 和 dev.json
    raw_path: 原始文本路径
    save_train/save_dev: 输出路径
    """
    random.seed(42)

    with open(raw_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    data = []
    for line in lines:
        tokens = []
        labels = []
        items = line.split() 
        for i in range(0, len(items), 2):
            token = items[i]
            label = items[i+1]
            tokens.append(token)
            labels.append(label)
        data.append({'tokens': tokens, 'labels': labels})

    # 随机划分
    random.shuffle(data)
    split_idx = int(len(data) * (1 - dev_ratio))
    train_data = data[:split_idx]
    dev_data = data[split_idx:]

    os.makedirs(os.path.dirname(save_train), exist_ok=True)
    os.makedirs(os.path.dirname(save_dev), exist_ok=True)

    with open(save_train, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(save_dev, 'w', encoding='utf-8') as f:
        for item in dev_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Process completed, train: {len(train_data)}, dev: {len(dev_data)}")
