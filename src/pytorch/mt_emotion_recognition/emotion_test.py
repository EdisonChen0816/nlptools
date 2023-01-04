# encoding=utf-8
import torch
from src.nlu_component.mt_emotion_recognition.mt_emotion_model import MTEmotionModel
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
simbert_model_path = './model/prd/simbert-pytorch'
simbert_model = AutoModel.from_pretrained(simbert_model_path)
tokenizer = AutoTokenizer.from_pretrained(simbert_model_path)
simbert_model.to(device)

model = MTEmotionModel()
model.load_state_dict(torch.load('./mt_emotion_model'))
model.eval()


y_pred = []
y_true = []
with open('/Users/edison/Documents/workspaces/python/chatbot/data/emotion/test.txt') as f:
    for line in f:
        text, label = line.replace('\n', '').split('\t')
        if label == 'happy':
            label = 1
        elif label == 'sad':
            label = 2
        elif label == 'angry':
            label = 3
        elif label == 'disgust':
            label = 4
        elif label == 'fear':
            label = 5
        elif label == 'surprise':
            label = 6
        else:  # 无情绪
            label = 0
        y_true.append(int(label))
        inputs = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=64).to(device)
        with torch.no_grad():
            embeddings = simbert_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu().numpy()
        output = model(torch.tensor(embeddings[0]))
        pred = torch.argmax(output).item()
        y_pred.append(int(pred))


print('acc:', accuracy_score(y_true, y_pred))

f1 = f1_score(y_true, y_pred, labels=[1], average='macro')
recall = recall_score(y_true, y_pred, labels=[1], average='macro')
precision = precision_score(y_true, y_pred, labels=[1], average='macro')
print('happy:', recall, precision, f1)

f1 = f1_score(y_true, y_pred, labels=[2], average='macro')
recall = recall_score(y_true, y_pred, labels=[2], average='macro')
precision = precision_score(y_true, y_pred, labels=[2], average='macro')
print('sad:', recall, precision, f1)

f1 = f1_score(y_true, y_pred, labels=[3], average='macro')
recall = recall_score(y_true, y_pred, labels=[3], average='macro')
precision = precision_score(y_true, y_pred, labels=[3], average='macro')
print('angry:', recall, precision, f1)

f1 = f1_score(y_true, y_pred, labels=[4], average='macro')
recall = recall_score(y_true, y_pred, labels=[4], average='macro')
precision = precision_score(y_true, y_pred, labels=[4], average='macro')
print('disgust:', recall, precision, f1)

f1 = f1_score(y_true, y_pred, labels=[5], average='macro')
recall = recall_score(y_true, y_pred, labels=[5], average='macro')
precision = precision_score(y_true, y_pred, labels=[5], average='macro')
print('fear:', recall, precision, f1)

f1 = f1_score(y_true, y_pred, labels=[6], average='macro')
recall = recall_score(y_true, y_pred, labels=[6], average='macro')
precision = precision_score(y_true, y_pred, labels=[6], average='macro')
print('surprise:', recall, precision, f1)

f1 = f1_score(y_true, y_pred, labels=[0], average='macro')
recall = recall_score(y_true, y_pred, labels=[0], average='macro')
precision = precision_score(y_true, y_pred, labels=[0], average='macro')
print('none:', recall, precision, f1)
