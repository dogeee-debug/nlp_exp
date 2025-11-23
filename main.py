'''
Author: dogeee-debug gxyhome030404@gmail.com
Date: 2025-11-22 15:37:42
LastEditors: dogeee-debug gxyhome030404@gmail.com
LastEditTime: 2025-11-22 19:07:05
Description: 主函数
'''

import os
import json
from pprint import pprint

from src.segmenter import fmm_segment, jieba_segment
from src.pos_tagger import CRFPOSTagger
from src.train_bert import train_bert
from src.utils.data_loader import process_raw_data, load_processed_data

def main():
    print("=" * 80)
    print("Starting")
    print("=" * 80)


    if not os.path.exists("data/processed/train.json"):
        print("Processing...")
        process_raw_data(
            raw_path="data/raw/train.txt",
            save_train="data/processed/train.json",
            save_dev="data/processed/dev.json"
        )
    else:
        print("Skip processing...")

    train_sents = load_processed_data("data/processed/train.json")
    dev_sents = load_processed_data("data/processed/dev.json")

    print("\n=== Step 2: FMM===")
    sentence = "迈向充满希望的新世纪"
    print("输入句子:", sentence)
    print("FMM result:", fmm_segment(sentence))

    # === 
    print("\n=== Step 3: Jieba===")
    print("Jieba result:", jieba_segment(sentence))

    # === 
    print("\n=== Step 4: CRF training ===")
    crf = CRFPOSTagger()

    print("training samples:", len(train_sents))
    crf.train(train_sents)

    print("\nCRF model predict example:")
    crf_example = [("迈", "B"), ("向", "E")]
    pprint(crf.predict(crf_example))

    crf.save("crf_model.pkl")
    print("crf model saved")

    # === 
    print("\n=== Step 5: BERT training ===")
    label_list = ["B", "M", "E", "S"]

    train_bert(
        "data/processed/train.json",   
        "data/processed/dev.json",     
        label_list,                    
        "bert_model"        
    )

    print("\nBert model saved!")

    # === 
    print("Completed!")


if __name__ == "__main__":
    main()
