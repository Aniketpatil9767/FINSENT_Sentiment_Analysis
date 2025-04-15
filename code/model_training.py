# model_training.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load cleaned data
df = pd.read_csv('Data/cleaned_financial_news.csv')

# 2. Drop missing values in case there are any
df = df.dropna(subset=['cleaned_headline'])

# 3. TF-IDF transformation
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['cleaned_headline'])

# 4. Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentiment'])

# 5. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Save model and vectorizer
joblib.dump(model, 'Models/financial_news_sentiment_model.pkl')
joblib.dump(tfidf_vectorizer, 'Models/tfidf_vectorizer.pkl')
import pickle

# Save the model
with open('Models/logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the vectorizer
with open('Models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("Model and vectorizer saved successfully.")
