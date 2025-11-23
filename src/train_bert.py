'''
Author: dogeee-debug gxyhome030404@gmail.com
Date: 2025-11-21 22:55:17
LastEditors: dogeee-debug gxyhome030404@gmail.com
LastEditTime: 2025-11-22 16:32:13
Description: BERT训练用于分词任务
'''


import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments

class TokenDataset(Dataset):
    def __init__(self, data_path, tokenizer, label2id, max_len=128):
        self.data = []
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        labels = [self.label2id[l] for l in item['labels']]

        tokens = tokens[:self.max_len]
        labels = labels[:self.max_len]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt' # issue: https://github.com/huggingface/transformers/issues/20638?utm_source=chatgpt.com
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        label_ids = [-100] * self.max_len
        label_ids[:len(labels)] = labels

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label_ids)
        }


def train_bert(train_file, dev_file, label_list, output_dir='./src/models/bert_model'):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    train_dataset = TokenDataset(train_file, tokenizer, label2id)
    dev_dataset = TokenDataset(dev_file, tokenizer, label2id)

    model = BertForTokenClassification.from_pretrained(
        "bert-base-chinese",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8, 
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./results/logs',
        save_total_limit=1,
        report_to="none"  # 禁用 W&B，每次都跳这个太烦了我真操了
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"BERT model saved to {output_dir}")
