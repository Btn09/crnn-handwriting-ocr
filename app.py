import torch
import torch.nn as nn
import cv2
import numpy as np
import gradio as gr
from spellchecker import SpellChecker

device = torch.device("cpu")
MODEL_DOSYASI = "lines_model_best.pth" 
IMG_WIDTH = 512
IMG_HEIGHT = 32
BEAM_SIZE = 3

class CRNN(nn.Module):
    def __init__(self, num_chars):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1))
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_chars + 1)

    def forward(self, x):
        x = self.cnn(x).permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1) 
        x, _ = self.rnn(x)
        return self.fc(x)

def decode_beam_search(preds, num_to_char, beam_size=3):
    preds = preds[:, 0, :] 
    paths = [(0.0, -1, [])]
    for t in range(preds.size(0)):
        step_probs = preds[t]
        topk_logprobs, topk_indices = torch.topk(step_probs, beam_size)
        new_paths = []
        for score, prev_idx, char_list in paths:
            for i in range(beam_size):
                idx = topk_indices[i].item()
                logp = topk_logprobs[i].item()
                new_score = score + logp
                new_char_list = list(char_list)
                if idx != 0 and idx != prev_idx:
                    new_char_list.append(num_to_char[idx])
                new_paths.append((new_score, idx, new_char_list))
        new_paths.sort(key=lambda x: x[0], reverse=True)
        paths = new_paths[:beam_size]
    return "".join(paths[0][2])

def resize_with_pad(img, target_w, target_h):
    h, w = img.shape
    ratio = w / h
    target_ratio = target_w / target_h
    if ratio > target_ratio:
        new_w = target_w
        new_h = int(target_w / ratio)
    else:
        new_h = target_h
        new_w = int(target_h * ratio)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, 0:new_w] = resized
    return canvas

with open('alphabet.txt', 'r', encoding='utf-8') as f: alphabet = f.read()
num_to_char = {i + 1: char for i, char in enumerate(alphabet)}
model = CRNN(len(alphabet)).to(device)
model.load_state_dict(torch.load(MODEL_DOSYASI, map_location=device))
model.eval()
spell = SpellChecker()

def yapay_zeka_okuyucu(image):
    if image is None: return "Hata: Resim yüklenmedi.", ""
    
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    
    img = resize_with_pad(img, IMG_WIDTH, IMG_HEIGHT)
    img_tensor = torch.FloatTensor(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor).log_softmax(2).permute(1, 0, 2)
        raw_text = decode_beam_search(preds, num_to_char, BEAM_SIZE)

    corrected_words = []
    for word in raw_text.split():
        if word.isalpha():
            corr = spell.correction(word)
            corrected_words.append(corr if corr else word)
        else:
            corrected_words.append(word)
    final_text = " ".join(corrected_words)

    return raw_text, final_text

aruyuz = gr.Interface(
    fn=yapay_zeka_okuyucu,
    inputs=gr.Image(label="El Yazısı Fotoğrafı Çek veya Yükle"),
    outputs=[
        gr.Textbox(label="🤖 Modelin Saf Görüşü (Ham Okuma)"),
        gr.Textbox(label="✨ Sözlükle Düzeltilmiş Sonuç")
    ],
    title="Batın'ın Yapay Zeka OCR Okuyucusu 🚀",
    description="Telefonunuzun kamerasıyla bir el yazısı çekin, yapay zeka anında okusun!",
    theme="default"
)

if __name__ == "__main__":
    aruyuz.launch(server_name="0.0.0.0", server_port=7860, share=True)