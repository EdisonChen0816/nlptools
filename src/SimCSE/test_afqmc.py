# encoindg=utf-8
from transformers import AutoModel, AutoTokenizer
import torch
from scipy.spatial.distance import cosine
import json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('./saved_model5')
model = AutoModel.from_pretrained('./saved_model5').to(DEVICE)

threshold = 0.9
y_true = []
y_pred = []
f1 = open('./data/afqmc_predict.json', 'a+', encoding='utf-8')
with open('./data/afqmc_public/test.json') as f:
    for line in f:
        data = json.loads(line.replace('\n', ''))
        id = data['id']
        text1 = data['sentence1']
        text2 = data['sentence2']
        texts = [text1, text2]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu().numpy()
        cosine_sim = 1 - cosine(embeddings[0], embeddings[1])
        if cosine_sim > threshold:
            pred = 1
        else:
            pred = 0
        f1.writelines("{\"id\":" + str(id) + ", \"label\": " + str(pred) + "}\n")

