import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from  sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

nltk.download('stopwords', quiet=True)

# 1. LOADING THE DATA
df = pd.read_csv("data/emails.csv")

# 2. PREPROCESSING
df.drop_duplicates(inplace=True)


# 3.  Strip the "Subject:" prefix
df['text'] = df['text'].str.replace(r'^Subject:\s*', '', regex=True)

# 4. Text cleaning
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()                             # Lowercase
    text = re.sub(r'http\S+|www\S+', '', text)     # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)            # Remove email addresses
    text = re.sub(r'[^a-z\s]', '', text)           # Remove lowercase or a space & numbers
    text = re.sub(r'\s+', ' ', text).strip()       # Normalize whitespace
    tokens = text.split()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words]  # Remove stopwords + stem
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# 5. Train/test split (stratified to preserve 76/24 class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['spam'],
    test_size=0.2,
    random_state=42,
    stratify=df['spam']   # important given the class imbalance
)

# 6. VECTORIZATION
vectorizer = TfidfVectorizer(max_features=3000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 7. MODELS 
logistic_reg = LogisticRegression(max_iter=1000)
logistic_reg.fit(X_train_tfidf, y_train)

y_pred = logistic_reg.predict(X_test_tfidf)

print("Logistic Regression Results")
print("----------------------------")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_tfidf, y_train)

y_pred_dt = dt_model.predict(X_test_tfidf)

print("Decision Tree Results")
print("----------------------------")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_dt):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_dt)}")

svm_model = LinearSVC(max_iter=1000, random_state=42)
svm_model.fit(X_train_tfidf, y_train)

y_pred_svm = svm_model.predict(X_test_tfidf)

print("SVM Results")
print("----------------------------")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_svm):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_svm):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_svm):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_svm)}")

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# 8. TUNING 
grid_search = GridSearchCV(
    LinearSVC(random_state=42, max_iter=1000),
    param_grid,
    cv=5,                  # 5-fold cross validation
    scoring='f1',          # optimize for F1
    verbose=1              # shows progress
)

grid_search.fit(X_train_tfidf, y_train)

print(f"Best C value: {grid_search.best_params_}")
print(f"Best cross-val F1: {grid_search.best_score_:.4f}")

#  Retrieve the best model from grid search
best_svm = grid_search.best_estimator_

# Predict with the best model
y_pred_best_svm = best_svm.predict(X_test_tfidf)

# Evaluate
print("Tuned SVM Results")
print("----------------------------")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_best_svm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_best_svm):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_best_svm):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_best_svm):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_best_svm)}")

def predict_email(email_text):
    cleaned = clean_text(email_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = best_svm.predict(vectorized)[0]
    label = "🚨 SPAM" if prediction == 1 else "✅ HAM"
    print(f"Email: {email_text[:60]}...")
    print(f"Prediction: {label}\n")

# --- PREDICT CUSTOM EMAILS ---
predict_email("Hi John, just following up on our meeting tomorrow. Let me know if the time still works.")

predict_email("Limited time offer! Buy now and get 50% off. Free shipping on all orders today only!")

predict_email("Hey, are we still on for lunch on Friday? Let me know!")
