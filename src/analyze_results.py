'''
Author: dogeee-debug gxyhome030404@gmail.com
Date: 2025-11-22 18:10:02
LastEditors: dogeee-debug gxyhome030404@gmail.com
LastEditTime: 2025-11-23 13:53:45
FilePath: \nlp_experiment\src\analyze_results.py
Description:分词/标注结果分析与可视化
'''


import os
import json
import jieba
import joblib
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from transformers import BertTokenizerFast, BertForTokenClassification

from src.segmenter import fmm_segment
from src.pos_tagger import CRFPOSTagger
from src.utils.data_loader import load_processed_data

dev_path = "data/processed/dev.json"
crf_model_path = "crf_model.pkl"
bert_checkpoint = "bert_model/checkpoint-3500"
label_list = ["B", "M", "E", "S"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 512
stride = 256

dev_sents = load_processed_data(dev_path)

def word_seq_from_labels(sent):
    words = []
    current_word = ""
    for char, label in sent:
        current_word += char
        if label in ["E","S"]:
            words.append(current_word)
            current_word = ""
    return words

def evaluate_method(pred_word_seqs, true_word_seqs):
    y_true, y_pred = [], []
    for pred, true in zip(pred_word_seqs, true_word_seqs):
        true_set = set(true)
        for w in pred:
            y_pred.append(1 if w in true_set else 0)
            y_true.append(1)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

true_word_seqs = [word_seq_from_labels(s) for s in dev_sents]

# --- FMM ---
fmm_word_seqs = [fmm_segment("".join([c for c,_ in s])) for s in dev_sents]
fmm_prec, fmm_rec, fmm_f1 = evaluate_method(fmm_word_seqs, true_word_seqs)

# --- Jieba ---
jieba_word_seqs = [list(jieba.cut("".join([c for c,_ in s]))) for s in dev_sents]
jieba_prec, jieba_rec, jieba_f1 = evaluate_method(jieba_word_seqs, true_word_seqs)

# --- CRF ---
crf_model = CRFPOSTagger()
crf_model.load(crf_model_path)
crf_word_seqs = []
for s in dev_sents:
    pred_labels = crf_model.predict(s)
    pred_sent = list(zip([c for c,_ in s], pred_labels))
    crf_word_seqs.append(word_seq_from_labels(pred_sent))
crf_prec, crf_rec, crf_f1 = evaluate_method(crf_word_seqs, true_word_seqs)

tokenizer = BertTokenizerFast.from_pretrained(bert_checkpoint)
model = BertForTokenClassification.from_pretrained(bert_checkpoint, num_labels=len(label_list))
model.to(device)
model.eval()

all_true_labels, all_pred_labels = [], []

for sent in dev_sents:
    chars = [c for c,_ in sent]
    labels = [label_list.index(l) for _, l in sent]
    pred_labels_full = [None] * len(chars)

    start = 0
    while start < len(chars):
        end = min(start + max_len, len(chars))
        chunk_chars = chars[start:end]

        encoding = tokenizer(
            chunk_chars,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True
        )
        word_ids = encoding.word_ids(batch_index=0) 

        inputs = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)

        for idx, widx in enumerate(word_ids):
            if widx is None:
                continue
            global_idx = start + widx
            if pred_labels_full[global_idx] is None:
                pred_labels_full[global_idx] = logits[idx].argmax().item()

        if end == len(chars):
            break
        start += stride

    # 补齐未预测的字符
    pred_labels_full = [p if p is not None else 0 for p in pred_labels_full]

    assert len(pred_labels_full) == len(labels)
    all_true_labels.extend(labels)
    all_pred_labels.extend(pred_labels_full)

bert_report = classification_report(all_true_labels, all_pred_labels, target_names=label_list)
bert_f1 = f1_score(all_true_labels, all_pred_labels, average='macro')

print("==== 词级别分词评估 FMM / Jieba / CRF ====")
print(f"FMM:    Precision={fmm_prec:.4f}, Recall={fmm_rec:.4f}, F1={fmm_f1:.4f}")
print(f"Jieba:  Precision={jieba_prec:.4f}, Recall={jieba_rec:.4f}, F1={jieba_f1:.4f}")
print(f"CRF:    Precision={crf_prec:.4f}, Recall={crf_rec:.4f}, F1={crf_f1:.4f}")
print("\n==== BERT 分类报告 ====")
print(bert_report)

methods = ["FMM", "Jieba", "CRF", "BERT"]
f1_scores = [fmm_f1, jieba_f1, crf_f1, bert_f1]

plt.figure(figsize=(7,4))
plt.bar(methods, f1_scores, color=['skyblue','lightgreen','salmon','gold'])
plt.ylim(0,1)
plt.ylabel("F1 Score")
plt.title("分词方法 F1 对比")
plt.show()

cm = confusion_matrix(all_true_labels, all_pred_labels, labels=list(range(len(label_list))))
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_list, yticklabels=label_list, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("BERT 混淆矩阵")
plt.show()
