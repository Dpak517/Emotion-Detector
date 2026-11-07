# Emotion Detection with BERT

Classify English text into one of six emotions: **sadness, joy, love, anger, fear, surprise**.

This project compares classical machine learning, a BiLSTM, and a fine-tuned BERT model on the same data. The best model, a fine-tuned `bert-base-uncased`, reaches **95.2% accuracy** and is available on the Hugging Face Hub.

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/Deepakathf/bert-emotion-finetuned)

## Results

All models were trained and evaluated on the same stratified 80/20 split of a balanced subset of the dataset (8,000 samples per class, 48,000 in total).

| Model | Features / Architecture | Test Accuracy |
|-------|-------------------------|:-------------:|
| Logistic Regression | TF-IDF (unigrams + bigrams) | 90.7% |
| XGBoost | TF-IDF (unigrams + bigrams) | 91.2% |
| Bidirectional LSTM | Trainable embeddings + stacked BiLSTM | 94.2% |
| **BERT (`bert-base-uncased`)** | **Fine-tuned transformer** | **95.2%** |

## Dataset

- **Source:** [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) (`unsplit` version).
- **Classes:** sadness, joy, love, anger, fear, surprise.
- **Sampling:** 10,000 examples per class to balance the classes (60,000 rows).
- **Split:** 80% train / 20% test, stratified by label, `random_state=42`.

## Approach

1. Exploratory analysis: class balance, most frequent words per emotion, word clouds.
2. Classical baselines: TF-IDF with Logistic Regression and XGBoost.
3. Deep learning baseline: Keras Bidirectional LSTM.
4. Transformer: fine-tuning `bert-base-uncased` with the Hugging Face `Trainer`.

## Usage

Load the fine-tuned model from the Hugging Face Hub:

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="Deepakathf/bert-emotion-finetuned")
print(classifier("i feel so happy today"))
```

To reproduce the training, install the dependencies and run the notebook in order:

```bash
pip install -r requirements.txt
```

Notebook: `emotion-detection-bert.ipynb` (a GPU is recommended for the BiLSTM and BERT sections; it was developed on Kaggle).


## Tech stack

Python, Hugging Face Transformers and Datasets, PyTorch, TensorFlow / Keras, scikit-learn, XGBoost, NLTK, Matplotlib, Seaborn.

## Acknowledgements





















# 🎭 Emotion Detection with Fine-Tuned BERT

> A text emotion classification project comparing classical ML, a Bidirectional LSTM, and a fine-tuned BERT transformer on the `dair-ai/emotion` dataset, with Optuna hyperparameter tuning and a deployed model on the Hugging Face Hub.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Trainer-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-BERT-FFD21E)
![TensorFlow](https://img.shields.io/badge/Keras-BiLSTM-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-2E8B57)
![Optuna](https://img.shields.io/badge/Optuna-Tuning-4A90D9)
![Model](https://img.shields.io/badge/HF_Hub-bert--emotion--finetuned-yellow)
![License](https://img.shields.io/badge/License-MIT-yellowgreen)

---

## 📖 Overview

This project classifies short English text into one of six emotions: **sadness, joy, love, anger, fear, surprise**. It trains and compares four model families on the same balanced data split, from fast TF-IDF baselines up to a fine-tuned `bert-base-uncased`, so the accuracy gain of each step is easy to see.

The best model, a fine-tuned BERT, reaches **95.1% test accuracy** and is published on the Hugging Face Hub for direct use.

### Core Capabilities

- **Exploratory analysis:** most frequent words and word clouds for each emotion
- **Classical baselines:** TF-IDF + Logistic Regression and XGBoost
- **Hyperparameter tuning:** Optuna (TPE sampler) search over TF-IDF and Logistic Regression parameters
- **Deep learning:** stacked Bidirectional LSTM built with Keras
- **Transformer fine-tuning:** `bert-base-uncased` trained with the Hugging Face `Trainer`
- **Evaluation:** accuracy, macro F1, classification reports, confusion matrices, train vs validation loss curves
- **Deployment:** model and tokenizer pushed to the Hugging Face Hub, usable through a `pipeline` call

---

## 📊 Results

Test set: 12,000 samples (2,000 per class).

| Rank | Model | Test Accuracy |
|:----:|-------|:-------------:|
| **1** | **BERT (`bert-base-uncased`)** | **95.1%** |
| 2 | Bidirectional LSTM | 94.2% |
| 3 | Logistic Regression (Optuna-tuned, TF-IDF) | 91.2% |
| 4 | XGBoost (TF-IDF) | 91.1% |
| - | Logistic Regression (baseline, TF-IDF) | 90.2% |

---

## 🗂️ Dataset

- **Source:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion), `unsplit` configuration (416,809 rows)
- **Sampling:** 10,000 rows per label, giving a perfectly balanced 60,000-row dataset
- **Split:** stratified 80/20 (48,000 train / 12,000 test), `random_state=42`
- **Labels:** `0 sadness`, `1 joy`, `2 love`, `3 anger`, `4 fear`, `5 surprise`

---

## 🧠 Models

| Model | Details |
|-------|---------|
| **Logistic Regression** | TF-IDF (1-2 grams); Optuna tunes `max_ngram`, `min_df`, `max_df`, `max_features`, `sublinear_tf` and `C` (3-fold CV) |
| **XGBoost** | TF-IDF (1-2 grams, 10k features), 700 trees, depth 3, learning rate 0.07 |
| **BiLSTM** | Embedding (25k vocab, 256 dims), 3 stacked BiLSTM layers (64/64/32), dropout, `max_len=64`, EarlyStopping + ReduceLROnPlateau |
| **BERT** | `bert-base-uncased`, `max_length=64`, batch size 32, learning rate 2e-5, weight decay 0.01, fp16, up to 6 epochs (best epoch 3 kept) |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Dpak517/<repo-name>.git
cd <repo-name>
pip install transformers datasets accelerate torch scikit-learn xgboost optuna tensorflow nltk wordcloud seaborn matplotlib pandas
```

### Run the notebook

Open `emotion-detection-ft-bert-model.ipynb` in Jupyter, Kaggle or Colab (a GPU is recommended for the BERT section) and run the cells in order.

### Use the fine-tuned model

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="Deepakathf/bert-emotion-finetuned")

print(classifier("i feel so happy today"))
# [{'label': 'joy', 'score': 0.9996}]
```

Example predictions:

| Input | Predicted | Confidence |
|-------|-----------|:----------:|
| i feel so happy today | joy | 0.9996 |
| i can't believe this is happening, i'm terrified | fear | 0.9994 |
| wow, i did not expect that at all | surprise | 0.504 |

---

## ⚠️ Limitations

- The test set was also used to pick the best BERT epoch and for BiLSTM early stopping, so reported scores may be slightly optimistic. A separate validation split would give a cleaner estimate.
- BERT validation loss rises after epoch 3 (mild overfitting), so the best checkpoint is restored.
- Short, ambiguous sentences can get low-confidence predictions, as in the `surprise` example above.

---

## 📁 Repository Structure

```
├── emotion-detection-ft-bert-model.ipynb   # EDA, all models, evaluation, Hub upload
└── README.md
```

---

## 📄 License

Released under the MIT License.

## 👤 Author

**Deepak**, IIT BHU (Varanasi)
Hugging Face: [`Deepakathf`](https://huggingface.co/Deepakathf)

- Dataset: Saravia et al., *CARER: Contextualized Affect Representations for Emotion Recognition*, EMNLP 2018.
- Base model: [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) by Google.
## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details. Check the dataset and base model pages for their own licensing terms.
