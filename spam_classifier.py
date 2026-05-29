# Import libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv")

# Convert labels into numbers
# ham = 0, spam = 1
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

# Input and output
X = data['message']
y = data['label_num']

# Convert text into numerical vectors
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# -------------------------------
# Predict New Messages
# -------------------------------

while True:

    msg = input("\nEnter a message: ")

    if msg.lower() == 'exit':
        print("Program Ended")
        break

    # Convert message into vector
    msg_vector = vectorizer.transform([msg])

    # Prediction
    prediction = model.predict(msg_vector)

    if prediction[0] == 1:
        print("This message is SPAM")
    else:
        print("This message is HAM")