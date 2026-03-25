import re
import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

def clean_sentence(sentence):
    """Cleans a single sentence."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', sentence)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text.lower())
    
    stop_words = set(stopwords.words('english')).union({'iit', 'jodhpur', 'indian', 'institute', 'technology'})
    
    # Keep words that are not stopwords, longer than 1 char, and shorter than 20 chars (removes PDF gibberish)
    clean_tokens = [w for w in tokens if w not in stop_words and 1 < len(w) < 20]
    return clean_tokens

def main():
    with open("iitj_raw_corpus.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Cleaning text into sentences...")
    # Split by periods to get actual sentences
    raw_sentences = raw_text.split('.')
    
    clean_sentences = [clean_sentence(s) for s in raw_sentences if len(clean_sentence(s)) > 2]
    
    # Flatten list for statistics
    all_tokens = [word for sentence in clean_sentences for word in sentence]
    vocab = set(all_tokens)
    
    print("\n" + "="*40 + "\nDATASET STATISTICS\n" + "="*40)
    print(f"Total number of documents (sentences): {len(clean_sentences):,}")
    print(f"Total number of tokens: {len(all_tokens):,}")
    print(f"Vocabulary size: {len(vocab):,}")
    
    # Save as JSON so we keep the sentence structure for Task 2
    with open("Cleaned_Corpus.json", "w", encoding="utf-8") as f:
        json.dump(clean_sentences, f)
        
    print("Generating Word Cloud...")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(all_tokens))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig("Task1_WordCloud.png")
    plt.close()

if __name__ == "__main__":
    main()