# encoding=utf-8
import torch
from data import Dataset
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from model import BertClassifier


def train(model, train_path, val_path, learning_rate, epochs, batch_size):
    # 通过Dataset类获取训练和验证集
    train, val = Dataset(train_path), Dataset(val_path)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:7" if use_cuda else "cpu")
    # 定义损失函数和优化器
    loss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.to(device)
        loss = loss.to(device)
    # 开始进入训练循环
    for epoch_num in range(epochs):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        # 进度条函数tqdm
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            # 通过模型得到输出
            output = model(input_id, mask)
            # 计算损失
            batch_loss = loss(output, train_label)
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失
        total_acc_val = 0
        total_loss_val = 0
        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for val_input, val_label in val_dataloader:
                # 如果有GPU，则使用GPU，接下来的操作同训练
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = loss(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train.labels): .3f} 
              | Train Accuracy: {total_acc_train / len(train.labels): .3f} 
              | Val Loss: {total_loss_val / len(val.labels): .3f} 
              | Val Accuracy: {total_acc_val / len(val.labels): .3f}''')


if __name__ == '__main__':
    epochs = 10000
    model = BertClassifier()
    lr = 1e-6
    batch_size = 32
    train(model, './train.txt', './dev.txt', lr, epochs, batch_size)
