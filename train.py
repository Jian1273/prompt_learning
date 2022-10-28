from asyncio import FastChildWatcher
from ctypes import util
import os
import sys
import random
from time import clock_settime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoConfig, BertForSequenceClassification, AdamW, AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import CFG
from model import FcPromptModel, FineTuneModel, FGM, SoftPromptModel
from utils import Utils, Collate, FeedBackDataset

os.environ['CUDA_VISIBLE_DEVICES'] = CFG.device
# TOKENIZERS_PARALLELISM=False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class BuildPiiModel:
    def __init__(self, config):
        self.config = config
        self.utils = Utils(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)  
        self.query = self.config.query
        self.train = self.utils.load_data(self.config.train_path)
        self.validation = self.utils.load_data(self.config.val_path) 
        # self.config = AutoConfig.from_pretrained(self.config.model_path)
        if self.config.prompt_type == "hard":
            self.mask_index = self.utils.get_mask_index()
            self.model = FcPromptModel(self.config.model_path, self.config.num_lables, self.mask_index)
        elif self.config.prompt_type == "soft":
            self.model = SoftPromptModel(self.config.model_path, self.config.num_lables, self.config.n_tokens)
        else:
            self.model = FineTuneModel(self.config.model_path, self.config.num_lables) 

    def train_model(self):
        # 训练模型
        if self.config.gpu:
            self.model.cuda()
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learn_rate, eps=self.config.eps, betas=self.config.betas)
        collate_fn = Collate(self.tokenizer, is_test=False)
        trainData = torch.utils.data.DataLoader(FeedBackDataset(self.train, self.tokenizer), batch_size=CFG.batch_size, shuffle=True, num_workers=4,collate_fn=collate_fn)
        valData = torch.utils.data.DataLoader(FeedBackDataset(self.validation, self.tokenizer), batch_size=CFG.batch_size, shuffle=False, num_workers=4,collate_fn=collate_fn)

        total_steps = int(self.config.epochs * len(trainData) / self.config.gradient_accumulation_steps)
        if self.config.schedule == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=self.config.num_warmup_steps, 
                    num_training_steps=total_steps, 
                    num_cycles=self.config.num_cycles
                )
        elif CFG.schedule == "CAWR":
            T_mult = 1
            rewarm_epoch_num = 2
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,len(trainData) * rewarm_epoch_num,T_mult)
        else:
            scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=self.config.num_warmup_steps, 
                    num_training_steps=total_steps
                )
        
        best_acc, best_f1, classification_report = 0, 0, ""
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = []
            tbar = tqdm(trainData, file=sys.stdout)
            # 训练
            for batch in tbar:
                self.model.zero_grad()
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                # print(input_ids.size())
                logits = self.model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, batch["targets"].cuda())
                train_loss.append(loss.item())
                loss.backward()

                if self.config.fgm:
                    # FGM对抗训练
                    fgm = FGM(self.model)
                    fgm.attack(emb_name="word_embeddings")
                    logits = self.model(input_ids, attention_mask)
                    loss_adv = F.cross_entropy(logits, batch["targets"].cuda())
                    loss_adv.backward()
                    fgm.restore(emb_name="word_embeddings")
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                tbar.set_description(f"Epoch {epoch + 1}/{self.config.epochs} Loss: {round(np.mean(train_loss), 5)} lr: {scheduler.get_last_lr()[0]}")
    
            report = self.eval_model(self.model, valData, is_test=False)
            best_acc = max(best_acc, report["acc"])
            best_f1 = max(best_f1, report["f1"])
            print(report["report"])
            print(report["confusion_matrix"])
            print("acc: %.4f,  f1: %.4f, best_f1: %.4f" % (report["acc"], report["f1"], best_f1))
            if best_f1 == report["f1"]:
                if not os.path.exists(self.config.checkpoint_path):
                    os.makedirs(self.config.checkpoint_path) 
                path = os.path.join(self.config.checkpoint_path, self.config.best_model_name)
                torch.save(self.model.state_dict(), path)
                classification_report = report
        return classification_report

    def predict_batch(self, model, test_dataloader, is_test=True):
        model.cuda()
        model.eval()
        preds = []
        labels = []
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                logits = model(input_ids, attention_mask)
                logits = logits.detach().cpu().numpy()
                preds += list(np.argmax(logits, axis=1))
                if not is_test:
                    labels += list(batch["targets"])
        return {"y_pre":preds, "y_true":labels}
            
    def eval_model(self, model, test_data, is_test=False):
        # 输出模型的acc、f1-score和classification report
        report = self.predict_batch(model, test_data, is_test)
        acc = accuracy_score(report["y_true"], report["y_pre"])
        f1 = f1_score(report["y_true"], report["y_pre"], average='macro')
        confusion_matrixs = confusion_matrix(report["y_true"], report["y_pre"])
        report = classification_report(report["y_true"], report["y_pre"], digits=4)
        return {"acc":acc, "f1":f1, "report": report, "confusion_matrix": confusion_matrixs}

if __name__ == "__main__":
    # for n_tokens in [15,12,10, 8]:
    # CFG.n_tokens = n_tokens
    setup_seed(CFG.seed)
    f = open("outcome.txt", "a")
    f.write(CFG.model_path+"\n")
    f.write(CFG.train_path+"\n")
    f.write(CFG.val_path+"\n")
    f.write(str(CFG.query)+"\n")
    pii_model = BuildPiiModel(CFG)
    report = pii_model.train_model()
    print(report["report"])
    print(report["confusion_matrix"])
    f.write(report["report"])
    f.write(str(report["confusion_matrix"]))
    f.close()
