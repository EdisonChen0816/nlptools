# encoding=utf-8


from flask import Flask
from flask import request
from flask_json import FlaskJSON, as_json
import torch
from src.nlu_component.mt_emotion_recognition.mt_emotion_model import MTEmotionModel
from transformers import AutoTokenizer, AutoModel
from loguru import logger
import torch.nn.functional as F


app = Flask(__name__)
flask_json = FlaskJSON(app)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
simbert_model_path = './model/simbert-pytorch'
simbert_model = AutoModel.from_pretrained(simbert_model_path)
tokenizer = AutoTokenizer.from_pretrained(simbert_model_path)
simbert_model.to(device)

model = MTEmotionModel()
model.load_state_dict(torch.load('./model/mt_emotion_model1.pt'))
model.eval()


@app.route("/emotion", methods=["POST"])
@as_json
def http_server():
    try:
        data = request.get_json(force=False, silent=False)
        text = data['text']
        inputs = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=64).to(device)
        with torch.no_grad():
            embeddings = simbert_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu().numpy()
        pred = F.softmax(model(torch.tensor(embeddings[0]))).tolist()
        logger.info(text)
        return {
            'happy': pred[1],
            'sad': pred[2],
            'angry': pred[3],
            'disgust': pred[4],
            'fear': pred[5],
            'surprise': pred[6],
            'none': pred[0],
            'res_code': 'ok'
        }
    except Exception as e:
        logger.error(e)
        return {
            'res_code': 'error'
        }


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=56668)