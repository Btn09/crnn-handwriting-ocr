# ✍️ CRNN Handwriting Recognition with Trigram Language Model

This project implements a high-performance **Convolutional Recurrent Neural Network (CRNN)** for recognizing handwritten text, specifically trained and evaluated on the world-renowned **IAM Handwriting Database**. It features a hybrid deep learning architecture combined with a **Trigram-based Language Model** for advanced post-processing.

## 🚀 Key Features
- **Hybrid Architecture:** Uses CNN for visual feature extraction and Bi-LSTM for sequence modeling.
- **Two-Stage Training:** Progressive learning starting from Word-level to Line-level for maximum stability.
- **Advanced Post-Processing:** CTC Loss combined with Beam Search and a Trigram Autocorrect system.
- **High Accuracy:** Optimized with learning rate scheduling and gradient clipping.

## 📂 Dataset Sources

This project is built using the **IAM Handwriting Database**. To replicate the training and evaluation, please download the specific versions used from the links below:

1. **Line Level Dataset:** [IAM Handwriting Dataset (Lines)](https://www.kaggle.com/datasets/sumitparsad/iam-handwritting-dataset?select=lines.tgz)
   - *Used for fine-tuning the model on full sentence structures.*
2. **Word Level Dataset:** [IAM Word Dataset](https://www.kaggle.com/datasets/ngkinwang/iam-dataset)
   - *Used for initial training to recognize individual characters and vocabulary.*

## 🏗️ Architecture Detail
1. **CNN (VGG-style):** Extracts spatial features from handwritten images.
2. **RNN (Bi-LSTM):** Captures temporal dependencies between characters.
3. **CTC Loss:** Maps sequences to text without needing per-character alignment.
4. **Trigram Correction:** Refines the output by calculating the probability of word sequences, significantly reducing Word Error Rate (WER).

## 📊 Evaluation Results
The following results were achieved on a random test set of **500 samples** from the IAM dataset:

| Metric | Accuracy / Rate |
| :--- | :--- |
| **Character Accuracy (1 - CER)** | **94.26%** |
| **Word Accuracy (1 - WER)** | **90.40%** |
| **Exact Match Accuracy** | **90.40%** |
| **Character Error Rate (CER)** | 5.74% |
| **Word Error Rate (WER)** | 9.60% |

## 🛠️ Installation & Usage
### Requirements
```bash
pip install torch opencv-python numpy jiwer pyspellchecker
