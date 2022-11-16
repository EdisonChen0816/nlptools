# encoding=utf-8
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

text = '花呗不消费有没有年费[SEP]花呗不用就不会产生费用'
t = tokenizer(text, padding='max_length', max_length=64, truncation=True, return_tensors="pt")
print(t)