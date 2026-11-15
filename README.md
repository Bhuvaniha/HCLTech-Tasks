# Multiclass Sentiment Analysis

A deep learning experiment that classifies text into **negative**, **neutral**, and **positive** sentiment. The notebook trains and compares four neural architectures — a feed-forward network, a vanilla RNN, an LSTM, and a GRU — under a shared hyperparameter search, then evaluates the best model on a held-out test set.

## Overview

| Item | Detail |
|------|--------|
| Task | 3-class text classification (negative / neutral / positive) |
| Dataset | [`Sp1786/multiclass-sentiment-analysis-dataset`](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset) (Hugging Face) |
| Framework | TensorFlow / Keras |
| Vectorization | `TextVectorization` (20,000 tokens, 100-token sequences) |
| Architectures | NN, SimpleRNN, Bidirectional LSTM, Bidirectional GRU |
| Output | Softmax over 3 classes |

## Dataset

Loaded directly through the `datasets` library, which provides three splits:

- **Train**
- **Validation**
- **Test**

Each sample contains a `text` field and an integer `label` (0 = negative, 1 = neutral, 2 = positive). Rows with missing text or labels are dropped during cleaning.

## Pipeline

1. **Import statements** — libraries for data handling, modeling, and visualization.
2. **Dataset exploration** — inspect column names, features, split sizes, and sample records.
3. **Cleaning** — convert splits to DataFrames, drop nulls, extract label arrays.
4. **Vectorization** — adapt a `TextVectorization` layer on the training text and convert all splits to integer sequences.
5. **Model building** — a `build_model` factory that assembles each architecture from shared units, dropout, and learning-rate settings.
6. **Hyperparameter tuning & training** — grid search over:
   - Units: `64`, `128`
   - Dropout: `0.2`, `0.3`
   - Learning rate: `0.001`, `0.0005`
   
   Each model is trained for 5 epochs with a batch size of 64.
7. **Evaluation** — pick the best configuration per architecture, score each on the test set, and rank them.
8. **Best-model analysis** — print the winning configuration, a classification report, a model comparison bar chart, and a confusion matrix heatmap.

## Model Architectures

| Model | Structure |
|-------|-----------|
| **NN** | Embedding → Flatten → Dense → Dropout → Dense(softmax) |
| **RNN** | Embedding (masked) → SimpleRNN → Dropout → Dense(softmax) |
| **LSTM** | Embedding (masked) → Bidirectional LSTM → Dropout → Dense(softmax) |
| **GRU** | Embedding (masked) → Bidirectional GRU → Dropout → Dense(softmax) |

All models compile with the Adam optimizer (`clipnorm=1.0`) and sparse categorical cross-entropy loss.

## Results

The notebook reports, for each architecture:

- Best hyperparameters (units, dropout, learning rate)
- Validation accuracy
- Test accuracy
- A ranked comparison table and bar chart
- A confusion matrix and per-class precision/recall/F1 for the best model

## Getting Started

### Requirements

```
tensorflow
datasets
scikit-learn
pandas
numpy
matplotlib
seaborn
jupyter
```

Install with:

```bash
pip install tensorflow datasets scikit-learn pandas numpy matplotlib seaborn jupyter
```

### Run

```bash
jupyter notebook "Assignment - 2 (Sentiment Analysis).ipynb"
```

Run the cells top to bottom. The dataset downloads automatically on first run. A GPU is recommended for the recurrent models.

## Project Structure

```
.
├── Multiclass Sentiment Analysis   # Main notebook
└── README.md
```

## Notes

- Sequences that vectorize to all-zero rows are filtered out before training the recurrent models.
- `clipnorm=1.0` stabilizes gradients in the RNN/LSTM/GRU training.
- Results may vary slightly between runs due to non-deterministic initialization.
