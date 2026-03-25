import json
from collections import Counter
from gensim.models import Word2Vec

# Load your corpus
print("Loading corpus...")
with open("Cleaned_Corpus.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

# ---------------------------------------------------------
# ANSWER 1: TOP 10 WORDS
# ---------------------------------------------------------
words = [word for sentence in sentences for word in sentence]
top_10 = Counter(words).most_common(10)

print("\n--- P1: TOP 10 WORDS ---")
# Formatting exactly as the professor requested: word1, frequency, word2, frequency...
top_10_formatted = ", ".join([f"{word}, {count}" for word, count in top_10])
print(top_10_formatted)

# ---------------------------------------------------------
# ANSWER 2: 300-DIMENSIONAL VECTOR
# ---------------------------------------------------------
print("\n--- P1: 300-DIMENSIONAL VECTOR ---")
print("Training a quick 300-dim model just for this question...")

# We train a quick CBOW model with 300 dimensions as requested
model_300 = Word2Vec(sentences=sentences, vector_size=300, window=5, min_count=1, workers=4)

# Pick a word that is guaranteed to be in your academic corpus
target_word = "research" 

if target_word in model_300.wv:
    vector = model_300.wv[target_word]
    # Format the vector as a comma-separated string rounded to 4 decimals
    vector_str = ", ".join([f"{val:.4f}" for val in vector])
    print(f"{target_word} - {vector_str}")
else:
    print(f"Word '{target_word}' not found in vocab.")
