from flask import Flask, request, render_template
import joblib
import spacy
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('nlp_automl_model.pkl')

# Initialize Spacy model
nlp = spacy.load('en_core_web_sm')

# Preprocessing function
def preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the text from the form
        text = request.form['text']
        
        # Preprocess the text
        cleaned_text = preprocess(text)
        
        # Debug: Print the cleaned text
        print(f"Cleaned Text: {cleaned_text}")
        
        # Transform the text using the vectorizer
        transformed_text = model.named_steps['tfidf'].transform([cleaned_text])
        
        # Debug: Print the transformed text shape
        print(f"Transformed Text Shape: {transformed_text.shape}")
        
        # Predict the label
        prediction = model.named_steps['model'].predict(transformed_text)
        
        # Debug: Print the prediction
        print(f"Prediction: {prediction}")
        
        # Convert prediction to human-readable format
        prediction_text = 'Positive' if prediction[0] == 1 else 'Negative'
        
        return render_template('index.html', prediction=prediction_text, text=text)
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

