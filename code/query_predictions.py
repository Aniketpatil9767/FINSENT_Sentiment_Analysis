# query_predictions.py
import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('finsent.db')
cursor = conn.cursor()

# Query all predictions
cursor.execute("SELECT * FROM sentiment_predictions")
rows = cursor.fetchall()

# Load into a DataFrame for nice formatting
df = pd.DataFrame(rows, columns=["ID", "Headline", "Cleaned Headline", "Sentiment", "Timestamp"])
print(df)

conn.close()
