import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# LOAD DATA
fake = pd.read_csv("dataset/Fake.csv", low_memory=False)
true = pd.read_csv("dataset/True.csv", low_memory=False)

# REMOVE "Unnamed" 
fake = fake.loc[:, ~fake.columns.str.contains("Unnamed")]
true = true.loc[:, ~true.columns.str.contains("Unnamed")]

# REMOVE EMPTY COLUMNS 
fake = fake.dropna(axis=1, how="all")
true = true.dropna(axis=1, how="all")

# PRINT DATA SIZE
print("Fake News Rows:", len(fake))
print("Real News Rows:", len(true))

# LABEL DATA
fake["label"] = 0
true["label"] = 1

# MERGE DATA
data = pd.concat([fake, true], ignore_index=True)

# KEEP ONLY REQUIRED COLUMNS
data = data[["text", "label"]]

# SHUFFLE DATA
data = data.sample(frac=1, random_state=42)

# CLEAN TEXT FUNCTION
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

data["text"] = data["text"].apply(clean_text)

# SPLIT FEATURES & LABELS
X = data["text"]
y = data["label"]

# TF-IDF VECTORIZATION
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X = vectorizer.fit_transform(X)

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# MODEL
model = MultinomialNB()
model.fit(X_train, y_train)

# PREDICTION
y_pred = model.predict(X_test)

# RESULTS
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# SAVE MODEL
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel Saved Successfully!")
