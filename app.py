from flask import Flask, render_template, request
import pandas as pd
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained SVC model
svc_model = joblib.load("trained_model/svc_model.pkl")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Stemmer
stemmer = PorterStemmer()

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Load the training data
df = pd.read_csv('mental_health.csv')

# Preprocess the text data
def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(cleaned_text)
    filtered_tokens = [word for word in tokens if word.lower() not in set(stopwords.words('english'))]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return ' '.join(stemmed_tokens)

df['preprocessed_text'] = df['text'].apply(preprocess_text)

# Fit the TF-IDF vectorizer with the preprocessed training text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_text'])

# Route for the main page
# @app.route('/')
# def index():
#     chat_message = "Hello There, Share how you feel to predict your likely mental health state"
#     return render_template('index.html', chat_message=chat_message)

# # Route to handle form submission and display the predictive result
# @app.route('/', methods=['POST'])
# def predict():
#     user_text = request.form['user_text']
    
#     # Preprocess user input
#     preprocessed_text = preprocess_text(user_text)
    
#     # Transform preprocessed text into TF-IDF features
#     tfidf_text = tfidf_vectorizer.transform([preprocessed_text])
    
#     # Make prediction using the loaded SVC model
#     prediction = svc_model.predict(tfidf_text)[0]
    
#     # Define the chat message based on the prediction
#     if prediction == 1:
#         chat_message = "You seem to be in a negative mental state."
#     else:
#         chat_message = "You seem to be in a positive mental state."
    
#     return render_template('index.html', chat_message=chat_message)
# Route for the main page
@app.route('/')
def index():
    prev_chat_message = "Hello There, <br> Share how you feel to predict your likely mental health state"
    return render_template('index.html', prev_chat_message=prev_chat_message)

# Route to handle form submission and display the predictive result
@app.route('/', methods=['POST'])
def predict():
    user_text = request.form['user_text']
    
    # Preprocess user input
    preprocessed_text = preprocess_text(user_text)
    
    # Transform preprocessed text into TF-IDF features
    tfidf_text = tfidf_vectorizer.transform([preprocessed_text])
    
    # Make prediction using the loaded SVC model
    prediction = svc_model.predict(tfidf_text)[0]
    
    # Define the chat message based on the prediction
    if prediction == 1:
        prediction_result = "Oops!😟 You seem to be in a negative mental state."
    else:
        prediction_result = "Yas!😁 You seem to be in a positive mental state."
    
    # Retrieve previous chat message from the template
    prev_chat_message = request.form['prev_chat_message']
    
    return render_template('index.html', user_text=user_text, prev_chat_message=prev_chat_message, prediction_result=prediction_result)


if __name__ == '__main__':
    app.run(debug=True)
