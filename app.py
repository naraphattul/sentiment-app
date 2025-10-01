import pickle
import os
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer
import requests
import torch.nn.functional as F

# ===== 1. Config =====
MODEL_FILE = "sentiment_model.pkl"
MODEL_URL = "https://huggingface.co/Naraphat/sentiment_model/resolve/main/sentiment_model.pkl"
TOKENIZER_NAME = "distilbert-base-uncased"
id2label_mapping = {0: "positive", 1: "neutral", 2: "negative"}

# ===== 2. Init Flask =====
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ===== 3. Load Tokenizer =====
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
print("Tokenizer loaded successfully!")

# ===== 4. Download & Load Model (.pkl) =====
model = None
try:
    if not os.path.exists(MODEL_FILE):
        print("Downloading model from Hugging Face ...")
        resp = requests.get(MODEL_URL)
        with open(MODEL_FILE, "wb") as f:
            f.write(resp.content)
        print("Model downloaded successfully!")

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
        print("Model loaded successfully from sentiment_model.pkl!")

    model.eval()

except Exception as e:
    print(f"ERROR: Failed to load model. Detail: {e}")

# ===== 5. Routes =====
@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text_input = data.get("text", "").strip()
        if not text_input:
            return jsonify({"error": "Missing 'text' field"}), 400

        # Tokenize input
        inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]

        label_id = torch.argmax(logits, dim=-1).item()
        sentiment_label = id2label_mapping.get(label_id, "unknown")

        return jsonify({
            "sentiment": sentiment_label,
            "positive": float(probs[0]),
            "neutral": float(probs[1]),
            "negative": float(probs[2]),
            "text": text_input
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
