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
| Winning model | **Bidirectional LSTM** (test accuracy ≈ 0.652) |

## Dataset

Loaded through the `datasets` library, which provides three splits:

| Split | Samples |
|-------|---------|
| Train | 31,232 |
| Validation | 5,205 |
| Test | 5,206 |

Each sample contains `id`, `text`, `label` (0 = negative, 1 = neutral, 2 = positive), and `sentiment`. Rows missing text or labels are dropped, and after vectorization any all-zero (empty) sequences are removed before training the recurrent models (1 train row, 1 validation row dropped).

## Pipeline

1. **Import statements** — libraries for data handling, modeling, and visualization.
2. **Dataset exploration** — inspect columns, features, split sizes, and sample records.
3. **Cleaning** — convert splits to DataFrames, drop nulls, extract label arrays.
4. **Vectorization** — adapt a `TextVectorization` layer on training text and encode all splits as integer sequences.
5. **Model building** — a `build_model` factory that assembles each architecture from shared units, dropout, and learning-rate settings.
6. **Hyperparameter tuning & training** — grid search over all combinations.
7. **Evaluation** — select the best configuration per architecture, score each on the test set, and rank them.
8. **Best-model analysis** — print the winning configuration, a classification report, a model comparison bar chart, and a confusion matrix heatmap.

## How Tuning Works

For **each** model type, the notebook loops over the full grid using `itertools.product`:

| Hyperparameter | Values |
|----------------|--------|
| Units | `64`, `128` |
| Dropout | `0.2`, `0.3` |
| Learning rate | `0.001`, `0.0005` |

That is **8 configurations per architecture** (32 runs total). Each configuration is trained for **5 epochs** with a **batch size of 64** using Adam (`clipnorm=1.0`) and sparse categorical cross-entropy.

The selection happens in two stages:

1. **Within each architecture**, the configuration with the highest **validation accuracy** is kept as that model's best.
2. **Across architectures**, those four best models are compared by **test accuracy** and ranked. The top one becomes the final winning model.

This means a model can win the validation stage but lose overall — for example, the NN had the highest validation accuracy (0.677) but a lower test accuracy than the LSTM.

## Final Results

| Rank | Model | Best Units | Best Dropout | Best LR | Val Accuracy | Test Accuracy |
|------|-------|-----------|--------------|---------|--------------|---------------|
| 1 | **LSTM** | 64 | 0.3 | 0.0005 | 0.6549 | **0.6524** |
| 2 | GRU | 128 | 0.3 | 0.0010 | 0.6549 | 0.6400 |
| 3 | NN | 128 | 0.3 | 0.0010 | 0.6770 | 0.6311 |
| 4 | RNN | 128 | 0.2 | 0.0005 | 0.6217 | 0.6281 |

### Winning Model: Bidirectional LSTM

`embedding → Bidirectional LSTM(64) → Dropout(0.3) → Dense(3, softmax)`

Per-class performance on the test set:

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| negative | 0.66 | 0.64 | 0.65 | 1546 |
| neutral | 0.59 | 0.58 | 0.58 | 1929 |
| positive | 0.71 | 0.75 | 0.73 | 1730 |
| **Accuracy** | | | **0.65** | **5205** |

The **neutral** class is the hardest to classify, while **positive** is detected most reliably. The confusion matrix and model comparison bar chart are generated at the end of the notebook.

## Model Architectures

| Model | Structure |
|-------|-----------|
| **NN** | Embedding → Flatten → Dense → Dropout → Dense(softmax) |
| **RNN** | Embedding (masked) → SimpleRNN → Dropout → Dense(softmax) |
| **LSTM** | Embedding (masked) → Bidirectional LSTM → Dropout → Dense(softmax) |
| **GRU** | Embedding (masked) → Bidirectional GRU → Dropout → Dense(softmax) |

## Notes

- Sequences that vectorize to all-zero rows are filtered out before training the recurrent models.
- `clipnorm=1.0` stabilizes gradients in the RNN/LSTM/GRU training.
- Results may vary slightly between runs due to non-deterministic initialization.
