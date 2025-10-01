import pickle
import os
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests

# 1. กำหนดค่าคงที่
MODEL_FILE = 'sentiment_model.pkl'
TOKENIZER_NAME = "distilbert-base-uncased" 
id2label_mapping = {0: "positive", 1: "neutral", 2: "negative"}

# 2. เริ่มต้น Flask
app = Flask(__name__, template_folder='.')
CORS(app)

# 3. โหลด Model และ Tokenizer
model = None
tokenizer = None
try:
    # โหลด Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print("Tokenizer loaded successfully!")

    url = "https://huggingface.co/Naraphat/sentiment_model/resolve/main/sentiment_model.pkl"

    if not os.path.exists("sentiment_model.pkl"):
        print("Downloading model from Hugging Face ...")
        resp = requests.get(url)
        with open("sentiment_model.pkl", "wb") as f:
            f.write(resp.content)
        print("Model downloaded successfully!")

    # โหลดโมเดลทั้งก้อนจากไฟล์ .pkl
    with open(MODEL_FILE, 'rb') as file:
        model = pickle.load(file)
        
    model.eval() # ตั้งค่าโมเดลเป็นโหมดประเมินผล
    print(f"Model Object loaded successfully from {MODEL_FILE} and ready for inference!")

except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_FILE}. Please ensure it is in the same directory.")
    model = None
except Exception as e:
    # แสดง Error ที่เกิดขึ้น
    print(f"ERROR: Failed to load model or tokenizer. Detail: {e}")
    model = None
    
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
    app.run(host='0.0.0.0', port=5000, debug=True)