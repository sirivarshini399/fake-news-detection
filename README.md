📰 Fake News Detection using Machine Learning
📌 Project Overview

This project is a Machine Learning-based Fake News Detection system that classifies news articles as Real or Fake using Natural Language Processing (NLP). It uses TF-IDF Vectorization and Multinomial Naive Bayes classifier and is deployed using a Flask web application for real-time predictions.

🚀 Features
Detects whether a news article is Fake or Real
Text preprocessing and cleaning
NLP-based feature extraction using TF-IDF
Machine Learning model (Naive Bayes)
Web interface using Flask
Real-time prediction from user input
🛠️ Tech Stack
Python 🐍
Pandas
Scikit-learn
NLP (Natural Language Processing)
TF-IDF Vectorizer
Flask (Web Framework)
HTML/CSS (Frontend)
📂 Project Structure
Fake-News-Detection/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── templates/
│   └── index.html
│
├── train.py
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
⚙️ How It Works
Load dataset (Fake & True news)
Clean and preprocess text
Convert text into numerical features using TF-IDF
Train Naive Bayes classifier
Save trained model
Flask app takes user input and predicts:
❌ Fake News
✅ Real News
▶️ How to Run the Project
1. Clone the repository
git clone https://github.com/sirivarshini399/fake-news-detection.git
cd fake-news-detection
2. Create virtual environment
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Train the model
python train.py
5. Run Flask app
python app.py
6. Open in browser
http://127.0.0.1:5000/
🧪 Example Inputs
Real News:
Government announces new education policy for schools.
ISRO successfully launches communication satellite into orbit.
Fake News:
Scientists confirm humans can live for 200 years using special water.
Drinking lemon juice gives superhuman strength instantly.
📊 Model Performance
Accuracy: ~90–96% (varies based on dataset split)
Model: Multinomial Naive Bayes
🎯 Future Improvements
Upgrade to Deep Learning (LSTM / BERT)
Improve dataset quality
Deploy on cloud (Render / AWS)
Add API endpoint for external use
👩‍💻 Author

Siri Nadimpalli
B.Tech IT (AI/ML Enthusiast)
Project built for learning and portfolio development.

⭐ Acknowledgement

This project is created for learning Machine Learning, NLP, and Flask development for real-world applications.
