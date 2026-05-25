import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

print("Fake News Rows:", len(fake))
print("Real News Rows:", len(true))

fake["label"] = 0
true["label"] = 1


data = pd.concat([fake, true])

data = data[["text", "label"]]

data = data.sample(frac=1, random_state=42)


def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text

# Apply cleaning
data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["label"]


vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# MODEL


model = MultinomialNB()

# TRAIN MODEL

model.fit(X_train, y_train)

# TEST MODEL

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel Saved Successfully!")