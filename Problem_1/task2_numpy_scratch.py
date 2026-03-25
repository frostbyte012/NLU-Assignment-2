import json
import numpy as np
from collections import defaultdict

print("Loading data and building vocabulary...")
with open("Cleaned_Corpus.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

# To make sure this finishes quickly in pure NumPy, we use a subset of the corpus
sentences = sentences[:500] 

# Build Vocabulary
words = [word for sentence in sentences for word in sentence]
vocab = list(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
V = len(vocab)
N = 30  # Embedding Dimension
WINDOW_SIZE = 2

print(f"Vocabulary Size (Subset): {V}")

# Generate Skip-gram Training Data
training_data = []
for sentence in sentences:
    indices = [word2idx[w] for w in sentence]
    for i, target_idx in enumerate(indices):
        # Get context window
        start = max(0, i - WINDOW_SIZE)
        end = min(len(indices), i + WINDOW_SIZE + 1)
        for j in range(start, end):
            if i != j:
                training_data.append((target_idx, indices[j]))

print(f"Generated {len(training_data)} training pairs.")

# ==========================================
# NUMPY NEURAL NETWORK FROM SCRATCH
# ==========================================
# Initialize Weights (W1: hidden layer, W2: output layer)
np.random.seed(42)
W1 = np.random.uniform(-0.1, 0.1, (V, N))
W2 = np.random.uniform(-0.1, 0.1, (N, V))

def softmax(x):
    e_x = np.exp(x - np.max(x)) # Subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

epochs = 50
learning_rate = 0.05

print("\nStarting NumPy Training Loop...")
for epoch in range(epochs):
    loss = 0
    for target, context in training_data:
        # FORWARD PASS
        h = W1[target]          # Look up the embedding (1xN)
        u = np.dot(W2.T, h)     # Matrix multiplication (Vx1)
        y_pred = softmax(u)     # Probabilities

        # CALCULATE LOSS (Cross-Entropy)
        loss += -np.log(y_pred[context] + 1e-9)

        # BACKWARD PASS (Calculus / Gradients)
        e = y_pred.copy()
        e[context] -= 1         # Gradient of cross-entropy + softmax

        dW2 = np.outer(h, e)    # Gradient for W2
        dW1 = np.dot(W2, e)     # Gradient for W1

        # WEIGHT UPDATE
        W2 -= learning_rate * dW2
        W1[target] -= learning_rate * dW1
        
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

# ==========================================
# EVALUATION (Cosine Similarity)
# ==========================================
def get_numpy_neighbors(target_word, topn=5):
    if target_word not in word2idx:
        return ["Not in vocab subset"]
    
    target_idx = word2idx[target_word]
    target_vec = W1[target_idx]
    
    # Calculate cosine similarity manually across all vectors
    similarities = []
    for i in range(V):
        if i == target_idx: continue
        vec = W1[i]
        cos_sim = np.dot(target_vec, vec) / (np.linalg.norm(target_vec) * np.linalg.norm(vec))
        similarities.append((cos_sim, idx2word[i]))
        
    # Sort by highest similarity
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [word for _, word in similarities[:topn]]

print("\n****************************************")
print(" NUMPY SCRATCH MODEL: SEMANTIC ANALYSIS")
print("****************************************")
targets = ['research', 'student', 'phd', 'exam']
for w in targets: 
    print(f"{w.upper()}: {get_numpy_neighbors(w)}")
