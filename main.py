from flask import Flask, request, jsonify, render_template
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon
nltk.download('vader_lexicon')

app = Flask(__name__)

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the input text from the request
    text = request.form.get('text', '')

    if not text:
        return jsonify({"error": "Text input is required"}), 400

    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(text)

    # Classify the sentiment
    sentiment = "neutral"
    if sentiment_scores['compound'] > 0.05:
        sentiment = "positive"
    elif sentiment_scores['compound'] < -0.05:
        sentiment = "negative"

    result = {
        "text": text,
        "sentiment": sentiment,
        "scores": sentiment_scores
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
