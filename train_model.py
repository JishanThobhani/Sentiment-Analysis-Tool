import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load your dataset (replace with your dataset path)
data = pd.read_csv("Twitter_Data.csv")  # Adjust file path as needed


# Basic Text Preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-letter characters
        text = text.lower()  # Convert to lowercase
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
        return text
    else:
        return ""



# Apply preprocessing to the text data
data['clean_text'] = data['clean_text'].apply(preprocess_text)
data = data[(data['clean_text'] != "") & (data['category'] != "")]

# Split dataset into features (X) and labels (y)
X = data['clean_text']
y = data['category']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model and vectorizer as .pkl files
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved as 'model.pkl' and 'vectorizer.pkl'")
