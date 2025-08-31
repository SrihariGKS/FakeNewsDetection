from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import string
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained Bi-LSTM model
model_path = "C:/Users/gksha/Fake_News_Detection/bilstm.h5"  # Ensure the path to your model file is correct
model = load_model(model_path)

# Load the tokenizer
tokenizer_path = "C:/Users/gksha/Fake_News_Detetokenizer.pkl"  # Path to saved tokenizer
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text.strip()

# Define max sequence length
max_length = 256  # Ensure this matches your training

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')  # Load HTML template for frontend

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the news text from the form
    news_text = request.form['news_text']
    
    # Preprocess the text
    cleaned_text = preprocess_text(news_text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding="post")

    # Make prediction
    prediction = model.predict(padded_sequence)
    label = "Fake" if prediction[0][0] < 0.5 else "Real"
    confidence = float(prediction[0][0]) if label == "Real" else 1 - float(prediction[0][0])

    # Render result
    return render_template('index.html', prediction=f"Prediction: {label} (Confidence: {confidence:.2%})")

if __name__ == '__main__':
    app.run(debug=True)