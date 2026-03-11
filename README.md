# Spam Email Classifier

A machine learning project that classifies emails as spam or ham using NLP preprocessing and three supervised learning models — with a tuned SVM achieving **98% accuracy** and **97.99% F1 score**.

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 98.51% | 98.86% | 94.89% | 96.83% |
| Decision Tree | 95.79% | 89.79% | 93.07% | 91.40% |
| **SVM (best)** | **99.03%** | **98.17%** | **97.81%** | **97.99%** |

---

## Project Structure

```
spam-email-classifier/
│
├── data/
│   └── emails.csv          # Dataset: 5,728 labeled emails
│
├── src/
│   └── classifier.py       # Full pipeline: preprocessing → models → evaluation
│
├── .gitignore
└── README.md
```

---

## How It Works

### 1. Preprocessing
Raw email text is cleaned through a pipeline:
- Strip `Subject:` prefix
- Lowercase, remove URLs, email addresses, punctuation
- Remove stopwords (NLTK)
- Stem words to root form using PorterStemmer

### 2. Vectorization
Cleaned text is transformed into numerical features using **TF-IDF** with the top 3,000 most important words. TF-IDF captures both how frequently a word appears in an email and how uniquely informative it is across the dataset.

### 3. Models
Three classifiers are trained and compared on the same vectorized data:
- **Logistic Regression** — strong linear baseline for text
- **Decision Tree** — interpretable but prone to overfitting on high-dimensional data
- **SVM (LinearSVC)** — optimal margin classifier, well suited for high-dimensional text

### 4. Tuning
GridSearchCV with 5-fold cross-validation was used to find the optimal `C` parameter for SVM across `[0.001, 0.01, 0.1, 1, 10, 100]`, confirming `C=1` as the best value with a cross-validated F1 of **97.56%**.

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/oluwaseyifatai556/spam-email-classifier.git
cd spam-email-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the classifier
```bash
python src/classifier.py
```

---

## Test on Custom Emails

You can test the trained SVM on your own email text:

```python
predict_email("Congratulations! You've won a free iPhone. Click here to claim your prize!")
# SPAM

predict_email("Hi, are we still meeting tomorrow at 10am?")
# HAM
```

---

##  Dataset

- **Source:** Kaggle - Spam Email Dataset
- **Size:** 5,728 emails
- **Balance:** 76% ham, 24% spam
- **Features:** Raw email text + binary spam label

---

## Key Learnings

- **TF-IDF + SVM is a classic high-performance combo** for text classification — the high dimensionality of text data is actually an advantage for SVMs finding a separating hyperplane
- **Decision trees overfit on text** — with 3,000 features, they memorize training paths instead of learning general patterns
- **Precision vs Recall tradeoff** matters in spam detection — missing a legitimate email (false positive) is often costlier than letting occasional spam through
- **Class imbalance** (76/24) makes accuracy a misleading standalone metric — F1 is the more reliable headline number

---

## Tech Stack

- Python
- scikit-learn
- NLTK
- pandas
