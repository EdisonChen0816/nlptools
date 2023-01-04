# encoding=utf-8
from torch import nn
import torch
from transformers import AutoTokenizer, AutoModel
from torch.optim import Adam
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from typing import List


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
simbert_model_path = './model/simbert-pytorch'
simbert_model = AutoModel.from_pretrained(simbert_model_path)
tokenizer = AutoTokenizer.from_pretrained(simbert_model_path)
simbert_model.to(device)
save_path = './model/mt_emotion_model.pt'

# happy
# sad
# angry
# disgust
# fear
# surprise
# 无情绪


def load_data(path):
    datas = []
    with open(path) as f:
        for line in f:
            text, emotion = line.replace('\n', '').split('\t')
            if emotion == 'happy':
                label = 1
            elif emotion == 'sad':
                label = 2
            elif emotion == 'angry':
                label = 3
            elif emotion == 'disgust':
                label = 4
            elif emotion == 'fear':
                label = 5
            elif emotion == 'surprise':
                label = 6
            else:
                label = 0
            datas.append((text, label))
    return datas


class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法"""

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def get_feature(self, text: str):
        inputs = tokenizer(text[0], padding='max_length', truncation=True, return_tensors="pt", max_length=64).to(device)
        with torch.no_grad():
            embeddings = simbert_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu().numpy()
        return embeddings[0], text[1]

    def __getitem__(self, index: int):
        return self.get_feature(self.data[index])


class MTEmotionModel(nn.Module):

    def __init__(self):
        super(MTEmotionModel, self).__init__()
        self.liner_1 = nn.Linear(in_features=768, out_features=128)
        self.liner_2 = nn.Linear(in_features=128, out_features=7)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, input):
        output1 = self.relu(self.liner_1(input))
        output1 = self.dropout(output1)
        logits = self.liner_2(output1)
        return logits


def evaluate(model, dev_dataloader):
    model.eval()
    y_true = []
    y_pred = []
    for dev_input, dev_label in dev_dataloader:
        dev_input = dev_input.to(device)
        dev_label = dev_label.to(device)
        y_true += dev_label.cpu().numpy().tolist()
        output = torch.argmax(nn.Softmax(dim=1)(model(dev_input)), dim=1)
        y_pred += output.cpu().numpy().tolist()
    acc = accuracy_score(y_true, y_pred)
    return acc


def train(model, train_path, dev_path, lr, epochs, batch_size):
    train, dev = load_data(train_path), load_data(dev_path)
    dev_dataloader = DataLoader(TrainDataset(dev), batch_size=batch_size, shuffle=True)
    loss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(device)
    loss = loss.to(device)
    best = 0
    for epoch in range(epochs):
        train_dataloader = DataLoader(TrainDataset(train), batch_size=batch_size, shuffle=True)
        for idx, source in enumerate(tqdm(train_dataloader), start=1):
            train_input = source[0].to(device)
            train_label = source[1].to(device)
            ouput = model(train_input)
            batch_loss = loss(ouput, train_label)
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            if (idx + 1) % 200 == 0:
                logger.info(f'loss: {batch_loss}:.4f')
                acc = evaluate(model, dev_dataloader)
                model.train()
                if acc > best:
                    logger.info(f'best acc: {acc}:.4f, save model')
                    best = acc
                    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    epochs = 1000000
    model = MTEmotionModel()
    lr = 1e-5
    batch_size = 64
    train_path = './data/train1.txt'
    dev_path = './data/train1.txt'
    train(model, train_path, dev_path, lr, epochs, batch_size)