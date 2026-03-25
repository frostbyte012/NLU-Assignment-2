Natural Language Understanding: Word Embeddings & RNN-based Name Generation

Course: CSL7640 – Natural Language Understanding
Institution: Indian Institute of Technology Jodhpur
Author: Deepraj Majumdar (P25CS0003)

⸻

📌 Project Overview

This repository presents the implementation and analysis of two core Natural Language Processing (NLP) tasks as part of Assignment 2 for the CSL7640 course. The project emphasizes both theoretical understanding and practical implementation, including from-scratch model development.

The work is organized into two main components:

⸻

🧠 Problem 1: Word Embeddings (CBOW & Skip-gram)
	•	Curated and preprocessed a domain-specific corpus from the IIT Jodhpur official website (~14.9K tokens).
	•	Trained Continuous Bag of Words (CBOW) and Skip-gram models using the gensim library.
	•	Implemented the Skip-gram model from scratch using NumPy, including forward propagation and manual gradient-based backpropagation.
	•	Performed semantic analysis using:
	•	Nearest neighbor queries
	•	Word analogy tasks
	•	Visualized high-dimensional embeddings using t-SNE for qualitative evaluation.

⸻

🔤 Problem 2: Character-Level Name Generation using RNNs
	•	Trained generative models on a dataset of 1,000 Indian names.
	•	Implemented three neural architectures in PyTorch:
	•	Vanilla RNN
	•	Bidirectional LSTM (BiLSTM)
	•	RNN with Attention mechanism
	•	Evaluated model performance using:
	•	Novelty Rate
	•	Diversity Rate
	•	Qualitative assessment of generated names

⸻

📂 Repository Structure

├── Problem_1/
│   ├── Cleaned_Corpus.json         # Preprocessed IITJ academic corpus (~14.9k tokens)
│   ├── IITJ_WordCloud.png          # Word cloud visualization
│   ├── CBOW_Visualization.png      # t-SNE projection (CBOW)
│   ├── Skipgram_Visualization.png  # t-SNE projection (Skip-gram)
│   ├── task_2_4.py                 # Training, evaluation, and visualization pipeline
│   ├── task2_numpy_scratch.py      # NumPy-based Skip-gram implementation
│   └── get_p1_answers.py           # Corpus statistics and vector extraction
│
├── Problem_2/
│   ├── TrainingNames.txt           # Dataset of Indian names
│   ├── train.py                    # Model training script
│   └── evaluate.py                 # Evaluation and sample generation
│
├── Report/
│   └── Deepraj_Majumdar_NLU_A2.pdf # Detailed report and analysis
│
└── README.md


⸻

⚙️ Dependencies & Setup

Ensure Python 3.8+ is installed. Install required libraries using:

pip install numpy matplotlib scikit-learn wordcloud gensim torch


⸻

🚀 How to Run the Code

Part 1: Word Embeddings

cd Problem_1

1. Run Main Pipeline (Gensim Models & Visualization):

python3 task_2_4.py

2. Run NumPy-based Skip-gram Implementation:

python3 task2_numpy_scratch.py

3. Fetch Corpus Statistics:

python3 get_p1_answers.py


⸻

Part 2: RNN Name Generation

cd ../Problem_2

1. Train Models:

python3 train.py --epochs 250

2. Evaluate Models:

python3 evaluate.py


⸻

📊 Key Findings & Results

Word Embeddings
	•	Gensim’s optimized implementations (e.g., negative sampling and hierarchical softmax) significantly improve computational efficiency and embedding quality.
	•	The NumPy-based implementation validates the mathematical foundations of embedding learning but is computationally expensive on CPU.
	•	Models successfully captured domain-specific semantics, clustering related academic terms such as CGPA, CPI, and marks.

RNN-based Name Generation
	•	The Bidirectional LSTM (BiLSTM) outperformed Vanilla RNN and Attention-based models.
	•	Achieved:
	•	Novelty Rate: 95.00%
	•	Diversity Rate: 98.50%
	•	Generated names exhibited strong phonetic plausibility and structural coherence (e.g., Karitesh, Daprashra).

⸻

📎 Notes
	•	Ensure directory structure matches the layout above before execution.
	•	Training deep models may require significant compute time depending on hardware.

⸻

Developed as part of CSL7640: Natural Language Understanding at IIT Jodhpur.