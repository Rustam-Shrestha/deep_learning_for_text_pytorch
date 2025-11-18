# app.py
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
import pickle
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

app = FastAPI()
templates = Jinja2Templates(directory="skeleton")

# === Load your model once at startup ===
class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 150, batch_first=True)
        self.fc = nn.Linear(150, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h, _) = self.lstm(embedded)
        out = self.fc(h.squeeze(0))
        return out

# Load saved data
print("Loading model...")
with open('lyrics_lstm_model.pkl', 'rb') as f:
    data = pickle.load(f)

vocab = data['vocab']
max_len = data['max_len']
model = LSTMModel(len(vocab))
model.load_state_dict(data['model_state_dict'])
model.eval()
device = torch.device("cpu")
model.to(device)

# Reverse vocab for prediction
idx_to_word = {i: word for word, i in vocab.items()}

def predict_next_words(text, num_words=6):
    if not text.strip():
        return ""
    
    model.eval()
    words = word_tokenize(text.lower())
    generated = words.copy()
    
    for _ in range(num_words):
        indices = [vocab.get(w, vocab['<unk>']) for w in generated]
        if len(indices) > max_len:
            indices = indices[-max_len:]
        padded = [0] * (max_len - len(indices)) + indices
        input_tensor = torch.tensor([padded], dtype=torch.long).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            next_idx = torch.argmax(output, dim=1).item()
        
        next_word = idx_to_word.get(next_idx, "")
        if next_word in ".,!?:;\"'":
            continue
        generated.append(next_word)
    
    # Return only the newly generated words
    return " ".join(generated[len(words):])

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict")
async def predict(q: str = ""):
    words = predict_next_words(q, num_words=14)  # ← change 6 here for more/less
    return {"next_words": words}