import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -------------------------
# LOAD DATA
# -------------------------
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 1   # Fake
true["label"] = 0   # Real

# Balance dataset
min_len = min(len(fake), len(true))
fake = fake.sample(min_len, random_state=42)
true = true.sample(min_len, random_state=42)

df = pd.concat([fake, true])
df = df[["text", "label"]].dropna()

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------------
# CLEAN TEXT
# -------------------------
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["text"] = df["text"].apply(clean_text)

# -------------------------
# SPLIT DATA
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# -------------------------
# TF-IDF
# -------------------------
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    max_df=0.85
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# -------------------------
# MODEL
# -------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------
# EVALUATION
# -------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# -------------------------
# SAVE MODEL (IMPORTANT)
# -------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully!")

# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict_news(text):
    text = clean_text(text)
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)

    if prediction[0] == 1:
        print("❌ Fake News")
    else:
        print("✅ Real News")

# -------------------------
# INTERACTIVE TEST
# -------------------------
while True:
    sample_news = input("\nEnter news text (or type 'exit'): ")

    if sample_news.lower() == "exit":
        break

    predict_news(sample_news)