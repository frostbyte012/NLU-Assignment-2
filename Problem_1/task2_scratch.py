import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter

# ==========================================
# 1. DATA PREPARATION FOR PYTORCH
# ==========================================
print("Loading data and building vocabulary...")
with open("Cleaned_Corpus.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

# Flatten corpus and build vocab
words = [word for sentence in sentences for word in sentence]
vocab = Counter(words)
vocab_size = len(vocab)
word2idx = {w: idx for idx, (w, _) in enumerate(vocab.items())}
idx2word = {idx: w for w, idx in word2idx.items()}

WINDOW_SIZE = 3
EMBEDDING_DIM = 30

def generate_training_data(sentences, word2idx, window_size):
    cbow_data = []
    skipgram_data = []
    for sentence in sentences:
        indices = [word2idx[w] for w in sentence if w in word2idx]
        for i, target_idx in enumerate(indices):
            context_indices = indices[max(0, i - window_size):i] + indices[i + 1:min(len(indices), i + window_size + 1)]
            if len(context_indices) == 2 * window_size: # Keep only full windows for simplicity
                cbow_data.append((context_indices, target_idx))
                for context_idx in context_indices:
                    skipgram_data.append((target_idx, context_idx))
    return cbow_data, skipgram_data

cbow_data, skipgram_data = generate_training_data(sentences, word2idx, WINDOW_SIZE)
print(f"CBOW pairs: {len(cbow_data)} | Skip-gram pairs: {len(skipgram_data)}")

# ==========================================
# 2. PYTORCH MODEL ARCHITECTURES
# ==========================================
class CBOW_Model(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, inputs):
        # Average the context embeddings
        embeds = self.embeddings(inputs).mean(dim=1) 
        out = self.linear(embeds)
        return out

class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        return out

# ==========================================
# 3. TRAINING LOOP & EVALUATION
# ==========================================
def get_nearest_neighbors(model, target_word, topn=5):
    if target_word not in word2idx:
        return ["Not in vocab"]
    
    model.eval()
    target_idx = torch.tensor(word2idx[target_word])
    with torch.no_grad():
        target_embed = model.embeddings(target_idx).unsqueeze(0)
        all_embeds = model.embeddings.weight
        # Compute Cosine Similarity manually
        cos_sim = torch.nn.functional.cosine_similarity(target_embed, all_embeds)
        top_indices = torch.topk(cos_sim, topn + 1).indices.tolist() # +1 to ignore the word itself
    
    neighbors = [idx2word[idx] for idx in top_indices if idx2word[idx] != target_word][:topn]
    return neighbors

def train_model(model, data, model_name, epochs=10):
    print(f"\n--- Training {model_name} (Scratch) ---")
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        total_loss = 0
        for context, target in data[:50000]: # Slicing to speed up last-minute training
            
            # Format inputs based on model type
            if model_name == "CBOW":
                x = torch.tensor(context, dtype=torch.long).unsqueeze(0)
            else: # Skipgram
                x = torch.tensor(target, dtype=torch.long)
                
            y_true = torch.tensor([target if model_name == "CBOW" else context], dtype=torch.long)
            
            model.zero_grad()
            log_probs = model(x)
            if log_probs.dim() == 1: log_probs = log_probs.unsqueeze(0)
            
            loss = loss_function(log_probs, y_true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/50000:.4f}")

# Initialize and Train
cbow_scratch = CBOW_Model(vocab_size, EMBEDDING_DIM)
sg_scratch = SkipGram_Model(vocab_size, EMBEDDING_DIM)

train_model(cbow_scratch, cbow_data, "CBOW", epochs=5)
train_model(sg_scratch, skipgram_data, "Skip-gram", epochs=5)

# Evaluate
print("\n****************************************")
print(" SCRATCH MODELS: SEMANTIC ANALYSIS")
print("****************************************")
targets = ['research', 'student', 'phd', 'exam']
print("\nCBOW (Scratch) Neighbors:")
for w in targets: print(f"{w.upper()}: {get_nearest_neighbors(cbow_scratch, w)}")

print("\nSkip-gram (Scratch) Neighbors:")
for w in targets: print(f"{w.upper()}: {get_nearest_neighbors(sg_scratch, w)}")
    
    
    
# ==========================================
# 4. SCRATCH ANALOGY EXPERIMENTS
# ==========================================
def perform_analogy(model, pos1, neg1, pos2):
    """ Calculates: pos1 - neg1 + pos2 (e.g., BTech - UG + PG) """
    for w in [pos1, neg1, pos2]:
        if w not in word2idx:
            return f"Skipped (Missing word: '{w}')"

    model.eval()
    with torch.no_grad():
        v_pos1 = model.embeddings(torch.tensor(word2idx[pos1]))
        v_neg1 = model.embeddings(torch.tensor(word2idx[neg1]))
        v_pos2 = model.embeddings(torch.tensor(word2idx[pos2]))
        
        # Vector Arithmetic
        target_vec = v_pos1 - v_neg1 + v_pos2
        target_vec = target_vec.unsqueeze(0)
        
        all_embeds = model.embeddings.weight
        cos_sim = torch.nn.functional.cosine_similarity(target_vec, all_embeds)
        
        # Get top 5 to filter out the input words
        top_indices = torch.topk(cos_sim, 5).indices.tolist()

    # Return the highest match that isn't one of the input words
    for idx in top_indices:
        word = idx2word[idx]
        if word not in [pos1, neg1, pos2]:
            return word
    return "None"

print("\n--- Analogy Experiments (Scratch) ---")
analogies = [
    ('btech', 'ug', 'pg', "UG : BTech :: PG : ?"),
    ('phd', 'student', 'faculty', "Student : PhD :: Faculty : ?"),
    ('project', 'course', 'research', "Course : Project :: Research : ?")
]

print("CBOW (Scratch):")
for pos1, neg1, pos2, desc in analogies:
    result = perform_analogy(cbow_scratch, pos1, neg1, pos2)
    print(f"{desc} => {result}")

print("\nSkip-gram (Scratch):")
for pos1, neg1, pos2, desc in analogies:
    result = perform_analogy(sg_scratch, pos1, neg1, pos2)
    print(f"{desc} => {result}")
