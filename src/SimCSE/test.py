# encoding=utf-8
from transformers import AutoModel, AutoTokenizer
import torch
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('./saved_model')
model = AutoModel.from_pretrained('./saved_model').to(DEVICE)

threshold = 0.7
y_true = []
y_pred = []
with open('./test1.txt') as f:
    for line in f:
        text1, text2, label = line.replace('\n', '').split('\t')
        texts = [text1, text2]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu().numpy()
        cosine_sim = 1 - cosine(embeddings[0], embeddings[1])
        if cosine_sim > threshold:
            pred = 1
        else:
            pred = 0
        y_true.append(int(label))
        y_pred.append(pred)
acc = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print(acc, recall, precision, f1)

