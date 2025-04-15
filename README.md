# FINSENT | Financial Sentiment Analysis

FINSENT is a sentiment analysis project designed to classify financial news headlines into **positive**, **neutral**, or **negative** sentiments. It leverages NLP, machine learning, and SQL integration to analyze how news may impact market trends.

---

## 🔍 Features

- Preprocessing of financial news using NLTK
- TF-IDF vectorization for feature extraction
- Logistic Regression model for multi-class sentiment classification
- Label encoding of sentiments
- SQLite database integration for storing and querying user predictions

---

## 📁 Project Structure

 FINSENT_Sentiment_Analysis/
│
├── Code/
│   ├── data_preprocessing.py
│   ├── model_training.py               # Model training and saving
│   ├── sentiment_analysis.py           # Predict and store headline sentiment  
│   ├── query_predictions.py            # Query stored predictions from SQLite DB
│
├── Data/
│   ├── financial_news.csv
│   ├── cleaned_financial_news.csv
│
├── Models/
│   ├── logistic_model.pkl
│   ├── tfidf_vectorizer.pkl
│
├── finsent.db                # SQLite database
├── requirements.txt          # List of dependencies
├── README.md                 # Project overview
└── .gitignore                # (Optional) ignore DB, .pkl files etc.

yaml
Copy
Edit

---

## ⚙️ Technologies Used

- Python (NLTK, Scikit-learn, Pandas)
- SQLite (via `sqlite3`)
- VS Code
- Git / GitHub

---

## 🚀 How to Use

### 1. Clone the repository

```bash
git clone https://github.com/Aniketpatil9767/FINSENT_Sentiment_Analysis.git
cd FINSENT_Sentiment_Analysis
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Train the model
bash
Copy
Edit
python Code/model_training.py
4. Predict sentiment for new headlines
bash
Copy
Edit
python Code/sentiment_analysis.py
5. Query past predictions from database
bash
Copy
Edit
python Code/query_predictions.py
🧠 Sample Output
yaml
Copy
Edit
Enter a financial news headline: Tesla reports record profits for the first quarter
Predicted Sentiment: positive
💾 Database Integration
All predictions made through sentiment_analysis.py are automatically stored in a local SQLite database (finsent.db) and can be retrieved using query_predictions.py.

