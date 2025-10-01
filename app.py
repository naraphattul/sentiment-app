import os
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests

MODEL_NAME = "Naraphat/sentiment_model"
TOKENIZER_NAME = "distilbert-base-uncased"

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

id2label_mapping = {0: "positive", 1: "neutral", 2: "negative"}
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded directly from Hugging Face!")

# 4. หน้าหลัก
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

# 5. Endpoint สำหรับวิเคราะห์ Sentiment
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        return jsonify({'error': 'Internal Server Error: Model or Tokenizer not initialized.'}), 500

    try:
        data = request.get_json()
        text_input = data.get('text', '')
        if not text_input:
            return jsonify({'error': 'Invalid request: "text" field is missing.'}), 400
        
        # Tokenize input
        inputs = tokenizer(
            text_input, 
            return_tensors="pt",
            padding=True, 
            truncation=True
        )

        # ทำนายผล (Inference)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # ดึงผลลัพธ์ logits
        logits = outputs.logits

        # แปลง logits เป็นความน่าจะเป็นด้วย Softmax
        import torch.nn.functional as F
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]

        # ดึง label หลัก
        predictions = torch.argmax(logits, dim=-1).item()
        sentiment_label = id2label_mapping.get(predictions, "unknown")

        # เพิ่มตัวแปรใหม่สำหรับแต่ละระดับ
        positive = float(probs[0])
        neutral = float(probs[1])
        negative = float(probs[2])

        # ส่ง response กลับ
        return jsonify({
            'sentiment': sentiment_label,
            'positive': positive,
            'neutral': neutral,
            'negative': negative,
            'text': text_input
        })

    except Exception as e:
        print(f"Prediction Error Traceback: {e}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}. Please check backend terminal for details.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
