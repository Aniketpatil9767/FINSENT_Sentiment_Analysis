# data_preprocessing.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os
print("Current working directory:", os.getcwd())


# Download necessary NLTK data (only the first time)
nltk.download('punkt')
nltk.download('stopwords')

# Load the data
def load_data():
    news_data = pd.read_csv('Data/financial_news.csv', encoding='ISO-8859-1', header=None)
    news_data.columns = ['sentiment', 'headline']  # Assign appropriate column names
    print(news_data.head())  # Print the first few rows to confirm the structure
    return news_data


# Clean the data
def clean_data(df):
    # Drop any rows with missing values
    df = df.dropna(subset=['headline'])

    # Remove duplicates based on headlines
    df = df.drop_duplicates(subset=['headline'])

    return df

# Text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Apply preprocessing to the dataset
def preprocess_data(df):
    df['cleaned_headline'] = df['headline'].apply(preprocess_text)
    return df

# Main function
def main():
    # Load and clean data
    df = load_data()
    df = clean_data(df)
    
    # Preprocess text
    df = preprocess_data(df)
    
    # Optionally, save the processed data back to CSV
    df.to_csv('Data/cleaned_financial_news.csv', index=False)

    print("Data preprocessing completed and saved.")

if __name__ == '__main__':
    main()

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# 1. Load the cleaned data from the CSV
df = pd.read_csv('Data/cleaned_financial_news.csv')
# Remove rows with NaN values in 'cleaned_headline' column
df = df.dropna(subset=['cleaned_headline'])

# 2. Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # This will take the top 5000 words

# 3. Apply the TF-IDF transformation on the 'cleaned_headline' column
X = tfidf_vectorizer.fit_transform(df['cleaned_headline'])

# After this, you can proceed with other steps like encoding the sentiment, training the model, etc.
# 4. Encode the sentiment labels using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentiment'])

# Now 'y' contains the numerical labels corresponding to sentiments
