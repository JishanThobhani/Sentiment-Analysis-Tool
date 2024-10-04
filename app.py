from flask import Flask, request, render_template, jsonify
import joblib
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the saved model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

# Preprocess the text the same way it was done during training
def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        return text
    else:
        return ""



# Home route to serve the frontend page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to process the input and return sentiment
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['comment']
        clean_input = preprocess_text(user_input)
        transformed_input = vectorizer.transform([clean_input])
        prediction = model.predict(transformed_input)[0]
        sentiment_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
        sentiment = sentiment_map.get(prediction)
        return jsonify({'result': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
