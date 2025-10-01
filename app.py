import pickle
import os
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer
import requests
import torch.nn.functional as F

# -------------------------------------------------
# 1. ตั้งค่าชื่อไฟล์และ Tokenizer
# -------------------------------------------------
MODEL_FILE = 'sentiment_model.pkl'
TOKENIZER_NAME = "distilbert-base-uncased"
id2label_mapping = {0: "positive", 1: "neutral", 2: "negative"}

# -------------------------------------------------
# 2. เริ่มต้น Flask
# -------------------------------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

# -------------------------------------------------
# 3. โหลด Model และ Tokenizer
# -------------------------------------------------
model = None
tokenizer = None

try:
    # โหลด Tokenizer จาก Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print("Tokenizer loaded successfully!")

    # URL ของไฟล์โมเดล .pkl ใน Hugging Face
    url = "https://huggingface.co/Naraphat/sentiment_model/resolve/main/sentiment_model.pkl"

    # ถ้ายังไม่มีไฟล์ในเครื่อง → ดาวน์โหลด
    if not os.path.exists(MODEL_FILE):
        print("Downloading model from Hugging Face ...")
        resp = requests.get(url)
        with open(MODEL_FILE, "wb") as f:
            f.write(resp.content)
        print("Model downloaded successfully!")

    # โหลดโมเดลจาก .pkl
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

    model.eval()  # ตั้งโหมดประเมินผล
    print(f"✅ Model loaded successfully from {MODEL_FILE}!")

except Exception as e:
    print(f"❌ ERROR: Failed to load model or tokenizer. Detail: {e}")
    model = None

# -------------------------------------------------
# 4. หน้าเว็บหลัก
# -------------------------------------------------
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

# -------------------------------------------------
# 5. API สำหรับทำนาย Sentiment
# -------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model or Tokenizer not initialized.'}), 500

    try:
        data = request.get_json()
        text_input = data.get('text', '')
        if not text_input.strip():
            return jsonify({'error': 'Invalid request: "text" field is missing.'}), 400
        
        # Tokenize input
        inputs = tokenizer(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]

        # แปลงผลลัพธ์
        pred_idx = int(torch.argmax(logits, dim=-1).item())
        sentiment_label = id2label_mapping.get(pred_idx, "unknown")

        response = {
            'sentiment': sentiment_label,
            'positive': float(probs[0]),
            'neutral': float(probs[1]),
            'negative': float(probs[2]),
            'text': text_input
        }

        return jsonify(response)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# -------------------------------------------------
# 6. Run Server
# -------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
