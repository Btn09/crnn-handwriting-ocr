import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np

device = torch.device("cpu")
IMG_WIDTH = 512
IMG_HEIGHT = 32
BATCH_SIZE = 16 
EPOCHS = 15

def parse_lines_txt(file_path):
    lines_list = []
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.split()
            if parts[1] == "ok":
                img_id = parts[0]
                label = " ".join(parts[8:]).replace("|", " ")
                lines_list.append((img_id, label))
    return lines_list

class IAMLinesDataset(Dataset):
    def __init__(self, lines_list, img_dir, char_to_num):
        self.lines_list = lines_list
        self.img_dir = img_dir
        self.char_to_num = char_to_num

    def __len__(self): return len(self.lines_list)

    def __getitem__(self, idx):
        img_id, label = self.lines_list[idx]
        p = img_id.split('-')
        path = os.path.join(self.img_dir, p[0], f"{p[0]}-{p[1]}", f"{img_id}.png")
        
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        else:
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.astype(np.float32) / 255.0
        
        img = np.expand_dims(img, axis=0)
        target = [self.char_to_num[c] for c in label if c in self.char_to_num]
        if not target: target = [self.char_to_num.get(' ', 1)]
            
        return torch.FloatTensor(img), torch.LongTensor(target), len(target)

def collate_fn(batch):
    imgs, targets, target_lens = zip(*batch)
    return torch.stack(imgs), torch.cat(targets), torch.tensor(target_lens)

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

if __name__ == "__main__":
    if os.path.exists('alphabet.txt'):
        with open('alphabet.txt', 'r', encoding='utf-8') as f: 
            alphabet = f.read()
    else:
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-() "

    char_to_num = {char: i + 1 for i, char in enumerate(alphabet)}
    
    lines_txt_path = r"C:\Users\dikil\Desktop\proje\ascii\lines.txt"
    lines_img_dir = r"C:\Users\dikil\Desktop\proje\lines"
    
    lines_data = parse_lines_txt(lines_txt_path)
    if not lines_data: exit()

    dataset = IAMLinesDataset(lines_data, lines_img_dir, char_to_num)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=4, collate_fn=collate_fn)

    model = CRNN(len(alphabet)).to(device)

    weights_path = "words_model_best.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Transfer Learning: {weights_path} loaded successfully")
    else:
        print("Warning: Words model not found, starting training from scratch")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0002) 
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5) 

    print("Stage 2: Pro Lines Training Started")
    print(f"Total Lines: {len(lines_data)}")

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for i, (imgs, targets_flat, target_lens) in enumerate(loader):
            preds = model(imgs).log_softmax(2).permute(1, 0, 2)
            input_lens = torch.full(size=(imgs.size(0),), fill_value=preds.size(0), dtype=torch.long)
            
            loss = criterion(preds, targets_flat, input_lens, target_lens)
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            if i % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch} | Batch {i}/{len(loader)} | Loss: {loss.item():.4f} | LR: {current_lr}")
        
        avg_loss = epoch_loss / len(loader)
        scheduler.step()
        
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "lines_model_best.pth")
            print(f"New Record: Best model saved to lines_model_best.pth (Loss: {best_loss:.4f})")