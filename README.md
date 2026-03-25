Natural Language Understanding: Word Embeddings & RNN Name Generation

Course: CSL7640: Natural Language Understanding

Institution: IIT Jodhpur

Author: Deepraj Majumdar (P25CS0003)

Project Overview

This repository contains the source code, datasets, and final report for Assignment 2 of the CSL7640 NLU course. The project is divided into two primary natural language processing tasks:

Problem 1: Word Embeddings (CBOW & Skip-gram)

Curating and preprocessing a domain-specific text corpus from the IIT Jodhpur official website.

Training Continuous Bag of Words (CBOW) and Skip-gram models using gensim.

Implementing a Skip-gram model entirely from scratch using pure NumPy to understand the underlying mathematics and backpropagation.

Performing semantic analysis (nearest neighbors, vector analogies) and visualizing high-dimensional embeddings using t-SNE.

Problem 2: Character-Level Name Generation using RNNs

Training generative models on a dataset of 1,000 Indian names.

Implementing three architectures from scratch in PyTorch: Vanilla RNN, Bidirectional LSTM (BiLSTM), and RNN with Attention.

Evaluating models based on Novelty Rate, Diversity Rate, and the realism of generated samples.

📂 Repository Structure

├── Problem_1/
│   ├── Cleaned_Corpus.json         # Preprocessed IITJ academic corpus (~14.9k tokens)
│   ├── IITJ_WordCloud.png          # Word cloud of the IITJ corpus
│   ├── CBOW_Visualization.png      # t-SNE 2D projection for CBOW
│   ├── Skipgram_Visualization.png  # t-SNE 2D projection for Skip-gram
│   ├── task_2_4.py                 # Main pipeline for Word2Vec and t-SNE
│   ├── task2_numpy_scratch.py      # Pure NumPy implementation of Skip-gram
│   └── get_p1_answers.py           # Helper script for corpus stats and 300-dim vectors
├── Problem_2/
│   ├── TrainingNames.txt           # Dataset of 1000 Indian names for RNN training
│   ├── train.py                    # PyTorch training script for RNN, BiLSTM, and Attention
│   └── evaluate.py                 # Evaluation script for Novelty and Diversity metrics
├── Report/
│   └── Deepraj_Majumdar_NLU_A2.pdf # Comprehensive academic report and analysis
└── README.md


(Note: Ensure your actual folder structure matches this, or adjust the tree above accordingly).

Dependencies & Setup

This project requires Python 3.8+ and the following libraries. You can install them using pip:

pip install numpy matplotlib scikit-learn wordcloud gensim torch


How to Run the Code

Part 1: Word Embeddings

First, navigate to the Problem 1 directory:

cd Problem_1


1. Run the Main Pipeline (Gensim Models & Visualizations):
This script trains the optimal CBOW and Skip-gram models, prints the nearest neighbors and analogies, runs a hyperparameter ablation study, and saves the t-SNE plots.

python3 task_2_4.py


2. Run the "From Scratch" NumPy Model:
This script trains a Skip-gram model using pure NumPy matrix multiplication and manually derived gradients (trained on a subset of the corpus for CPU efficiency).

python3 task2_numpy_scratch.py


3. Fetch Corpus Statistics (P1 Specifics):
This helper script quickly calculates the top-10 most frequent words and extracts a 300-dimensional vector for a target word.

python3 get_p1_answers.py


Part 2: RNN Name Generation

First, navigate back to the root and into the Problem 2 directory:

cd ../Problem_2


1. Train the Models:
Train the Vanilla RNN, BiLSTM, and Attention models. This will output training/validation loss per epoch and save the model weights.

python3 train.py --epochs 250


2. Evaluate the Models:
Generates samples from the trained models and calculates the Novelty Rate (%) and Diversity Rate (%).

python3 evaluate.py


Key Findings & Results

Embeddings: Gensim's C-level optimizations (Hierarchical Softmax/Negative Sampling) are crucial for efficient semantic clustering. While the pure NumPy implementation mathematically proves embedding updates, it is computationally bound on standard CPUs. Both models successfully captured institutional jargon (e.g., grouping cgpa, cpi, and marks).

RNN Generation: The Bidirectional LSTM significantly outperformed the baseline RNN and Attention models, achieving a 95.00% Novelty Rate and 98.50% Diversity Rate. The bidirectional context allowed the network to generate structurally complex and phonetically realistic Indian names (e.g., 'Karitesh', 'Daprashra').

Developed for CSL7640 at IIT Jodhpur.