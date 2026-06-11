from flask import Flask, request, render_template
import pickle
import re

app = Flask(__name__)

# Load saved model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_news(text):
    text = clean_text(text)
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)

    return "Fake News ❌" if prediction[0] == 1 else "Real News ✅"

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    result = predict_news(news)
    return render_template("index.html", prediction=result, news=news)

if __name__ == "__main__":
    app.run(debug=True)