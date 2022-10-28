import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import AutoConfig, BertForSequenceClassification, AdamW, AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report

from config import CFG

class Collate:
    def __init__(self, tokenizer, is_test=False):
        self.tokenizer = tokenizer
        self.is_test = is_test
        # self.args = args

    def __call__(self, batch):
        texts = [item["text"] for item in batch]
        max_len = min(max([len(text)+2 for text in texts]), CFG.max_len)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens = True,
            return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        output = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        if not self.is_test:
            labels = [item["target"] for item in batch]
            output["targets"] = torch.tensor(labels)
        return output

class FeedBackDataset(Dataset):
    def __init__(self, data, tokenizer, is_test=False):
        self.data = data
        self.is_test = is_test
        self.tokenizer = tokenizer
        
    def __getitem__(self, idx):
        text = self.data['text'][idx]
        target_value = []
        if not self.is_test:
            target_value = self.data["label"][idx]
        return {"text":text, "target": target_value}
        
    def __len__(self):
        return len(self.data)

class Utils:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

    def create_prompt(self, text):
        return self.config.query + text

    def load_data(self, path, is_test=False):
        df = pd.read_csv(path)
        if self.config.prompt_type == "hard":
            df["text"] = [self.create_prompt(str(t)) for t in df["text"]]
        else:
            df["text"] = [str(t) for t in df["text"]]
        if not is_test:
            df["label"] = [CFG.label2id[l] for l in df["label"]]
            # pass
        return df
    
    def get_mask_index(self):
        mask_index = self.tokenizer.encode_plus(self.config.query,
                           add_special_tokens=True,
                           max_length=self.config.max_len,
                           truncation=True,
                           return_offsets_mapping=False)["input_ids"].index(self.tokenizer.mask_token_id)
        print("query为: %s mask的位置在：%s" % (self.config.query, mask_index))
        return mask_index


if __name__ == "__main__":
    u = Utils(CFG)
    print(u.config.model_path)
    print(u.config.label2id)
