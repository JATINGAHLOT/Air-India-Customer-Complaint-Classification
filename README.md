# ✈️ Air India Customer Complaint Classification
### Natural Language Processing + Neural Networks

> Automatically classify Air India customer complaints into 6 categories using NLP preprocessing, Word2Vec-style embeddings, and three neural network architectures — Feedforward MLP, LSTM, and 1D CNN.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline Overview](#-pipeline-overview)
- [Installation & Requirements](#-installation--requirements)
- [How to Run](#-how-to-run)
- [Models & Architectures](#-models--architectures)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Limitations](#-limitations)
- [File Descriptions](#-file-descriptions)

---

## 🎯 Project Overview

Air India receives thousands of customer complaints every day across a wide range of issues — delayed flights, missing baggage, booking errors, refund disputes, and more. Manually reading and routing each complaint is slow and error-prone at scale.

This project builds an **automated complaint classification system** that reads a complaint in plain English and assigns it to one of six predefined categories, enabling instant routing to the right department.

**Objective:** Build, train, and compare neural network classifiers using NLP techniques — strictly no traditional ML models (SVM, Random Forest, etc.).

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **File** | `Air_India_Complaints_Dataset.csv` |
| **Total records** | 5,000 |
| **Input feature** | `Complaint Text` (free-text English) |
| **Target** | `Category` (6 classes) |
| **Missing values** | None |
| **Avg complaint length** | ~65 characters / ~11 words |

### Class Distribution

| Category | Count | Share |
|----------|-------|-------|
| Flight Delay | 907 | 18.1% |
| Baggage Issues | 866 | 17.3% |
| Customer Service | 842 | 16.8% |
| In-flight Experience | 823 | 16.5% |
| Ticket Booking Problems | 798 | 16.0% |
| Refund Issues | 764 | 15.3% |

The dataset is **well-balanced** — the largest class is only 1.19× the smallest, so no oversampling or class-weighting was needed.

---

## 📁 Project Structure

```
air-india-complaint-classification/
│
├── Air_India_Complaints_Dataset.csv          # Raw dataset
├── Air_India_Complaint_Classification.ipynb  # Main Jupyter notebook
├── README.md                                 # This file
│
├── outputs/
│   ├── Air_India_NLP_Project_Report.docx     # Full project report
│   └── Air_India_Model_Metrics_Report.docx   # Metrics summary report
```

---

## 🔄 Pipeline Overview

```
Raw Text
   │
   ▼
┌─────────────────────────────────┐
│     1. TEXT PREPROCESSING       │
│  Lowercase → Remove URLs/digits │
│  → Strip punctuation → Remove   │
│  stopwords → Suffix stemming    │
└─────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────┐
│  2. WORD2VEC-STYLE EMBEDDINGS   │
│  TF-IDF (8000 features, 1-2     │
│  ngrams) → Truncated SVD/LSA    │
│  → 100-dim L2-normalised vectors│
└─────────────────────────────────┘
   │
   ├──────────────────────────────────────────────────────┐
   ▼                                                      ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│  MLP Model   │    │  LSTM Model  │    │   CNN Model      │
│ (on 100-dim  │    │ (on token    │    │  (on token       │
│  doc vector) │    │  sequences)  │    │   sequences)     │
└──────────────┘    └──────────────┘    └──────────────────┘
   │                      │                      │
   └──────────────────────┴──────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │   EVALUATION        │
              │ Accuracy · F1       │
              │ Precision · Recall  │
              │ Confusion Matrix    │
              └─────────────────────┘
```

---

## ⚙️ Installation & Requirements

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Optional (for word clouds)

```bash
pip install wordcloud
```

> **Note:** TensorFlow, Keras, PyTorch, NLTK, and Gensim are **not required**. The LSTM and CNN models are implemented from scratch in NumPy. The MLP uses `sklearn.neural_network.MLPClassifier`.

### Library Versions Used

| Library | Version |
|---------|---------|
| pandas | 2.3.3 |
| numpy | 2.3.4 |
| scikit-learn | 1.7.2 |
| matplotlib | latest |
| seaborn | latest |

---

## ▶️ How to Run

1. **Clone or download** this repository and ensure `Air_India_Complaints_Dataset.csv` is in the same directory as the notebook.

2. **Launch Jupyter:**
   ```bash
   jupyter notebook Air_India_Complaint_Classification.ipynb
   ```

3. **Run all cells** in order:
   ```
   Kernel → Restart & Run All
   ```

4. The notebook will automatically run through:
   - EDA and visualisations
   - Text preprocessing
   - Embedding generation
   - Training all 6 model configurations
   - Evaluation and comparison plots

> ⏱️ **Expected runtime:** ~10–15 minutes total (MLP trains in under 3 seconds; LSTM and CNN take a few minutes each for 12 epochs).

---

## 🧠 Models & Architectures

### Model 1 — Feedforward MLP

Input: 100-dimensional LSA document embedding

```
Input(100) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(6, Softmax)
```

| Config | Optimiser | Learning Rate | Batch | L2 Reg | Layers |
|--------|-----------|--------------|-------|--------|--------|
| A | Adam | 0.001 | 64 | 1e-4 | (256, 128) |
| B | SGD (adaptive) | 0.010 | 128 | 1e-3 | (512, 256, 128) |

---

### Model 2 — LSTM (NumPy)

Input: Token sequences (length = 25, vocab = 5,000)

```
Embedding(32) → LSTM(H) → Dense(32, ReLU) → Dense(6, Softmax)
```

Full gate equations implemented in NumPy:
- **Forget gate:** $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
- **Input gate:** $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$
- **Cell update:** $\tilde{c}_t = \tanh(W_g [h_{t-1}, x_t] + b_g)$
- **Output gate:** $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$
- **Cell state:** $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$
- **Hidden state:** $h_t = o_t \odot \tanh(c_t)$

| Config | Learning Rate | Hidden Size | Batch |
|--------|--------------|-------------|-------|
| A | 0.005 | 64 | 128 |
| B | 0.010 | 128 | 64 |

---

### Model 3 — 1D CNN (NumPy)

Input: Token sequences (length = 25, vocab = 5,000)

```
Embedding(32) → Conv1D(F, kernel_k) → ReLU → GlobalMaxPool → Dense(64, ReLU) → Dense(6, Softmax)
```

| Config | Learning Rate | Filters | Kernel Size | Batch |
|--------|--------------|---------|-------------|-------|
| A | 0.005 | 64 | 3 | 128 |
| B | 0.010 | 128 | 5 | 64 |

---

### Word2Vec-style Embeddings

All models use embeddings derived from:

```
TF-IDF (8,000 features, 1-2 ngrams, sublinear TF)
    ↓
Truncated SVD / LSA  →  100-dim dense vectors
    ↓
L2-normalisation  →  unit vectors (like Word2Vec output)
```

For LSTM and CNN, individual word vectors from the SVD component matrix are used to **pre-initialise the embedding layers** — directly analogous to loading pre-trained Word2Vec weights.

---

## 📈 Results

All metrics are weighted averages evaluated on the **750-sample held-out test set**.

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| MLP — Config A (Adam) | **100%** | **1.000** | **1.000** | **1.000** |
| MLP — Config B (SGD) | **100%** | **1.000** | **1.000** | **1.000** |
| LSTM — Config A | 34.0% | 0.211 | 0.340 | 0.236 |
| LSTM — Config B | 63.5% | 0.728 | 0.635 | 0.601 |
| CNN — Config A | 66.1% | 0.733 | 0.661 | 0.664 |
| **CNN — Config B** | **97.2%** | **0.974** | **0.972** | **0.971** |

### Best Sequence Model — CNN Config B (Per-Class)

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Baggage Issues | 1.00 | 1.00 | **1.00** | 130 |
| Customer Service | 0.93 | 1.00 | **0.97** | 126 |
| Flight Delay | 0.92 | 1.00 | **0.96** | 136 |
| In-flight Experience | 1.00 | 1.00 | **1.00** | 124 |
| Refund Issues | 1.00 | 0.82 | **0.90** | 114 |
| Ticket Booking Problems | 1.00 | 1.00 | **1.00** | 120 |
| **Weighted Average** | **0.974** | **0.972** | **0.971** | **750** |

---

## 💡 Key Findings

### Why MLP scores perfectly
The MLP receives **100-dimensional LSA document embeddings** that already encode the full document-level semantics. The six complaint categories are syntactically and lexically distinct enough that the embedding space becomes nearly linearly separable — the MLP just needs to find the boundary. This is a feature engineering win, not a model complexity win.

### Why CNN-B is the most trustworthy result
CNN Config B earns its **97.2% accuracy** from raw token sequences — no pre-solved feature space. It converges cleanly over 12 epochs with a consistently decreasing training loss and rising validation accuracy, which is textbook healthy learning. This is the most production-relevant result in the project.

### Optimisation insights

| Change | Effect |
|--------|--------|
| Adam vs SGD (MLP) | Same accuracy; Adam is 14× faster (0.2s vs 2.9s) |
| LSTM hidden 64 → 128 | F1: 0.236 → 0.601 (+155%) — capacity is critical |
| CNN filters 64 → 128 | F1: 0.664 → 0.971 — biggest single improvement overall |
| CNN kernel 3 → 5 | Wider phrase context; all classes benefit |
| LR 0.005 → 0.010 (CNN) | Val acc at epoch 6 in Config B already beats final Config A |

---

## ⚠️ Limitations

- **LSTM & CNN use a small vocabulary** (180 unique stems after preprocessing) — a richer vocabulary or pre-trained GloVe/FastText weights would significantly improve sequence model performance.
- **Only 12 epochs** for LSTM/CNN — more epochs with a learning rate schedule would close the gap with the MLP further.
- **Dataset is template-like** — real-world complaint text is messier and more ambiguous. The perfect MLP score would not hold in production.
- **TensorFlow/Keras not used** — LSTM and CNN are pure NumPy implementations, which are slower and lack GPU acceleration. Reimplementing in Keras would enable faster training and more configurations.
- **No BERT** — transformer-based models were not necessary given the dataset characteristics, but would be recommended for production deployment or more ambiguous complaint categories.

---

## 📄 File Descriptions

| File | Description |
|------|-------------|
| `Air_India_Complaints_Dataset.csv` | Raw complaint dataset (5,000 rows) |
| `Air_India_Complaint_Classification.ipynb` | Complete Jupyter notebook with all code, outputs, and visualisations |
| `Air_India_NLP_Project_Report.docx` | Full written project report with methodology, analysis, and figures |
| `Air_India_Model_Metrics_Report.docx` | Focused metrics summary — scorecards, per-class tables, confusion matrices |
| `README.md` | This file |

---

## 🏗️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

*Air India Complaint Classification · NLP & Neural Networks Project · March 2026*
