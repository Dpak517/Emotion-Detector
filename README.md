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

- Dataset: Saravia et al., *CARER: Contextualized Affect Representations for Emotion Recognition*, EMNLP 2018.
- Base model: [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) by Google.
## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details. Check the dataset and base model pages for their own licensing terms.
