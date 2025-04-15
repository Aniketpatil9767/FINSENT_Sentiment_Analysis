# sentiment_analysis.py
import pandas as pd
import string
import sqlite3
import nltk
import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
with open('Models/logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocess input text
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Get user input
headline = input("Enter a financial news headline: ")
cleaned = preprocess(headline)
vectorized = vectorizer.transform([cleaned])
predicted_label = model.predict(vectorized)

# Map label back to string
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
predicted_sentiment = label_map[predicted_label[0]]
print("Predicted Sentiment:", predicted_sentiment)

# Save result to SQLite database
conn = sqlite3.connect('finsent.db')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS sentiment_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    headline TEXT,
    cleaned_headline TEXT,
    sentiment TEXT,
    timestamp TEXT
)
""")

# Insert the prediction result
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
cursor.execute("""
INSERT INTO sentiment_predictions (headline, cleaned_headline, sentiment, timestamp)
VALUES (?, ?, ?, ?)
""", (headline, cleaned, predicted_sentiment, timestamp))

conn.commit()
conn.close()
