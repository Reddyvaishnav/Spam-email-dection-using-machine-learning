# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer

# Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)

# -------------------------------
# Load Dataset
# -------------------------------

data = pd.read_csv("spam.csv")

# Convert labels into numbers
data['label_num'] = data.label.map({
    'ham': 0,
    'spam': 1
})

# Input and Output
X = data['message']
y = data['label_num']

# -------------------------------
# Text Vectorization
# -------------------------------

vectorizer = TfidfVectorizer(stop_words='english')

X_vectorized = vectorizer.fit_transform(X)

# -------------------------------
# Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Logistic Regression
# -------------------------------

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]

# -------------------------------
# Random Forest
# -------------------------------

rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# -------------------------------
# Evaluation Function
# -------------------------------

def evaluate_model(y_test, pred, prob, model_name):

    print(f"\n========== {model_name} ==========")

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, prob)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC-AUC  :", roc_auc)

    print("\nClassification Report:\n")
    print(classification_report(y_test, pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d')

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, prob)

    plt.title(f"{model_name} ROC Curve")

    plt.show()

# -------------------------------
# Evaluate Both Models
# -------------------------------

evaluate_model(
    y_test,
    lr_pred,
    lr_prob,
    "Logistic Regression"
)

evaluate_model(
    y_test,
    rf_pred,
    rf_prob,
    "Random Forest"
)

# -------------------------------
# Cross Validation
# -------------------------------

print("\n========== Cross Validation ==========")

lr_scores = cross_val_score(
    lr_model,
    X_vectorized,
    y,
    cv=5,
    scoring='accuracy'
)

rf_scores = cross_val_score(
    rf_model,
    X_vectorized,
    y,
    cv=5,
    scoring='accuracy'
)

print("\nLogistic Regression CV Accuracy:",
      lr_scores.mean())

print("Random Forest CV Accuracy:",
      rf_scores.mean())