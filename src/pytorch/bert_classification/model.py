# encoding=utf-8
from torch import nn
from transformers import BertModel, AutoModel


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.2):
        super(BertClassifier, self).__init__()
        # self.bert = AutoModel.from_pretrained('/home/edison.chen/models/chinese-roberta-wwm-ext-large')
        # self.bert = BertModel.from_pretrained('/home/edison.chen/models/bert-base-chinese')
        self.bert = AutoModel.from_pretrained('/home/edison.chen/models/MTBert_v0.0.1')
        # self.bert= AutoModel.from_pretrained('/Users/edison/Documents/工作/数据和模型/模型/bert-base-chinese')
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output1 = self.linear1(dropout_output)
        linear_output2 = self.linear2(linear_output1)
        linear_output3 = self.linear3(linear_output2)
        final_layer = self.relu(linear_output3)
        return final_layer
