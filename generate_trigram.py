import collections
import re
import os

def create_trigram_model(file_path):
    words = []
    print("Scanning IAM Data...")
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.split()
            text = " ".join(parts[8:]).lower().replace("|", " ")
            clean_text = re.sub(r'[^a-z\s]', '', text)
            words.extend(clean_text.split())

    trigrams = collections.Counter(zip(words, words[1:], words[2:]))
    bigrams = collections.Counter(zip(words, words[1:]))
    unigrams = collections.Counter(words)
    
    print(f"Model generated: {len(trigrams)} unique trigram patterns found.")
    return trigrams, bigrams, unigrams

IAM_PATH = r"C:\Users\dikil\Desktop\proje\ascii\lines.txt"
TRIGRAMS, BIGRAMS, UNIGRAMS = create_trigram_model(IAM_PATH)

print("\nMost frequent trigrams:")
for gram, count in TRIGRAMS.most_common(5):
    print(f"{gram[0]} {gram[1]} {gram[2]}: {count} times")