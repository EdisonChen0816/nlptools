# -*- encoding: utf-8 -*-

import random
from typing import List

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig
from eval.mt_eval import eval

# 基本参数
EPOCHS = 100
SAMPLES = 10000
BATCH_SIZE = 32
LR = 1e-5
DROPOUT = 0.0
MAXLEN = 32
POOLING = 'cls'   # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 预训练模型目录
# model_path = '/Users/edison/Documents/工作/数据和模型/模型/chinese-roberta-wwm-ext-large'
model_path = '/home/edison.chen/models/chinese-roberta-wwm-ext-large'
# 微调后参数存放位置
SAVE_PATH = './saved_model/pytorch_model.bin'
BEST_PATH = './best_model/pytorch_model.bin'

# 数据位置
train_path = 'data/mt/train_v4.txt'
dev_path = './data/mt/dev.txt'


def load_data(path, mode):
    datas = []
    with open(path) as f:
        for line in f:
            if mode == 'train':
                text = line.replace('\n', '').split('\t')
                datas.append(text)
            else:
                sentence1, sentence2, label = line.replace('\n', '').split('\t')
                datas.append((sentence1, sentence2, label))
    return datas


class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法"""
    def __init__(self, data: List):
        self.data = data
      
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        # 添加自身两次, 经过bert编码之后, 互为正样本
        return tokenizer([text[0], text[1]], max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])


class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""
    def __init__(self, pretrained_model, pooling):
        super(SimcseModel, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = DROPOUT   # 修改config的dropout系数
        config.hidden_dropout_prob = DROPOUT           
        self.bert = AutoModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling
        
    def forward(self, input_ids, attention_mask, token_type_ids):

        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        
        if self.pooling == 'pooler':
            return out.pooler_output            # [batch, 768]
        
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]
        
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]                   
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
    
    
def simcse_unsup_loss(y_pred: 'tensor') -> 'tensor':
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]

    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    print(y_true.shape, sim.shape)
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss

            
def train(model, train_dl, optimizer) -> None:
    """模型训练函数"""
    model.train()
    global best
    best_loss = 999.0
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(DEVICE)
        
        out = model(input_ids, attention_mask, token_type_ids)        
        loss = simcse_unsup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            logger.info(f'loss: {loss.item():.4f}')
            model.train()
            if best_loss >= loss.item():
                best_loss = loss.item()
                torch.save(model.state_dict(), SAVE_PATH)
            f1 = eval()
            if best < f1:
                best = f1
                torch.save(model.state_dict(), BEST_PATH)
                logger.info(f"higher f1: {best:.4f} in batch: {batch_idx}, save model")
       
            
if __name__ == '__main__':
    logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # load data
    train_data = load_data(train_path, 'train')  # 两个数据集组合
    # train_data = random.sample(train_data, SAMPLES)                 # 随机采样
    dev_data = load_data(dev_path, 'dev')
    train_dataloader = DataLoader(TrainDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    # load model
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=model_path, pooling=POOLING).to(DEVICE)
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # train
    best = 0
    for epoch in range(EPOCHS):
        logger.info(f'epoch: {epoch}')
        train(model, train_dataloader, optimizer)
    logger.info(f'train is finished, best model is saved at {SAVE_PATH}')
    # eval
    # model.load_state_dict(torch.load(SAVE_PATH))
    # dev_corrcoef = eval(model)
    # logger.info(f'dev_f1: {dev_corrcoef:.4f}')
