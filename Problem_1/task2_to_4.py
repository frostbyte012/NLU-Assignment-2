import numpy as np
import matplotlib.pyplot as plt
import json
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

def train_optimal_models(sentences):
    """Trains the best models for the main assignment requirements."""
    print("\n" + "="*50)
    print(" PHASE 1: TRAINING OPTIMAL MODELS")
    print("="*50)
    
    VECTOR_SIZE, WINDOW, NEGATIVE, EPOCHS = 30, 3, 10, 100
    
    print(f"Training CBOW Model (Dims: {VECTOR_SIZE}, Epochs: {EPOCHS})...")
    cbow_model = Word2Vec(sentences=sentences, vector_size=VECTOR_SIZE, 
                          window=WINDOW, negative=NEGATIVE, sg=0, min_count=1, 
                          epochs=EPOCHS, workers=4)

    print(f"Training Skip-gram Model (Dims: {VECTOR_SIZE}, Epochs: {EPOCHS})...")
    sg_model = Word2Vec(sentences=sentences, vector_size=VECTOR_SIZE, 
                        window=WINDOW, negative=NEGATIVE, sg=1, min_count=1, 
                        epochs=EPOCHS, workers=4)
    return cbow_model, sg_model

def semantic_analysis(model, model_name):
    """Runs Task 3: Nearest Neighbors and Analogies."""
    print(f"\n{'*'*40}\n SEMANTIC ANALYSIS: {model_name}\n{'*'*40}")
    
    target_words = ['research', 'student', 'phd', 'exam']
    print("--- Top 5 Nearest Neighbors ---")
    for word in target_words:
        try:
            neighbors = model.wv.most_similar(word, topn=5)
            print(f"{word.upper()}: {[n[0] for n in neighbors]}")
        except KeyError:
            print(f"{word.upper()}: Not in vocabulary")

    print("\n--- Analogy Experiments ---")
    analogies = [
        (['pg', 'btech'], ['ug'], "UG : BTech :: PG : ?"),
        (['faculty', 'phd'], ['student'], "Student : PhD :: Faculty : ?"),
        (['research', 'project'], ['course'], "Course : Project :: Research : ?")
    ]
    for pos, neg, desc in analogies:
        try:
            result = model.wv.most_similar(positive=pos, negative=neg, topn=1)
            print(f"{desc} => {result[0][0]}")
        except KeyError as e:
            print(f"{desc} => Skipped (Missing word: {e})")

def visualize_embeddings(model, title):
    """Runs Task 4: t-SNE visualization."""
    words = model.wv.index_to_key[:150]
    word_vectors = np.array([model.wv[w] for w in words])

    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    twodim_vectors = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(12, 10))
    plt.scatter(twodim_vectors[:, 0], twodim_vectors[:, 1], edgecolors='k', c='skyblue')
    for word, (x, y) in zip(words, twodim_vectors):
        plt.text(x+0.2, y+0.2, word, fontsize=9)
        
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def run_ablation_study(sentences):
    """Runs the extra hyperparameter experiments."""
    print("\n" + "="*50)
    print(" PHASE 2: HYPERPARAMETER ABLATION STUDY")
    print("="*50)

    experiments = [
        {"name": "Exp 1 (Broad Topic)", "dim": 30, "win": 10, "neg": 10},
        {"name": "Exp 2 (High Penalty)", "dim": 30, "win": 3,  "neg": 20},
        {"name": "Exp 3 (Extreme Squish)", "dim": 10, "win": 3,  "neg": 10}
    ]

    for exp in experiments:
        print(f"\n--- {exp['name']} ---")
        print(f"Dims: {exp['dim']} | Window: {exp['win']} | Negative: {exp['neg']}")
        
        cbow = Word2Vec(sentences=sentences, vector_size=exp['dim'], 
                        window=exp['win'], negative=exp['neg'], sg=0, min_count=1, epochs=100)
                        
        sg = Word2Vec(sentences=sentences, vector_size=exp['dim'], 
                      window=exp['win'], negative=exp['neg'], sg=1, min_count=1, epochs=100)

        # Print how the neighbors for 'RESEARCH' change based on the parameters
        print(f"CBOW 'RESEARCH' Neighbors: {[n[0] for n in cbow.wv.most_similar('research', topn=5)]}")
        print(f"SKIP-GRAM 'RESEARCH' Neighbors: {[n[0] for n in sg.wv.most_similar('research', topn=5)]}")

def main():
    with open("Cleaned_Corpus.json", "r", encoding="utf-8") as f:
        sentences = json.load(f)

    # 1. Train optimal models and print main assignment analysis
    cbow_model, sg_model = train_optimal_models(sentences)
    semantic_analysis(cbow_model, "CBOW MODEL")
    semantic_analysis(sg_model, "SKIP-GRAM MODEL")

    # 2. Generate and save visualizations
    print("\nGenerating t-SNE visualizations (This takes a few seconds)...")
    visualize_embeddings(cbow_model, "CBOW Word Embeddings Visualization")
    visualize_embeddings(sg_model, "Skip-gram Word Embeddings Visualization")
    print("Visualizations saved successfully!")

    # 3. Run the extra experiments for the report discussion
    run_ablation_study(sentences)
    
    print("\nPipeline Complete! All data generated.")

if __name__ == "__main__":
    main()