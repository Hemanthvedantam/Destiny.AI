# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from datetime import datetime
import joblib

app = Flask(__name__)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
def load_data():
    with open('data/destinations.json', 'r', encoding='utf-8') as f:
        return json.load(f)

destinations = load_data()

# Text preprocessing
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Prepare data for model
def prepare_data():
    destination_texts = []
    for dest in destinations:
        text = ' '.join(dest['tags']) + ' ' + dest['why_perfect'] + ' ' + ' '.join(dest['activities'])
        for review in dest['reviews']:
            text += ' ' + review
        destination_texts.append(preprocess_text(text))
    return destination_texts

# Train model
def train_model():
    destination_texts = prepare_data()
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(destination_texts)
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(tfidf_matrix)
    return vectorizer, kmeans, tfidf_matrix

# Load or train model
MODEL_FILE = 'model_artifacts.pkl'
if os.path.exists(MODEL_FILE):
    import joblib
    vectorizer, kmeans, tfidf_matrix = joblib.load(MODEL_FILE)
else:
    vectorizer, kmeans, tfidf_matrix = train_model()
    joblib.dump((vectorizer, kmeans, tfidf_matrix), MODEL_FILE)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = {
        'interests': request.form.getlist('interests'),
        'budget': request.form.get('budget'),
        'season': request.form.get('season'),
        'travelers': request.form.get('travelers'),
        'days': request.form.get('days')
    }
    
    # Prepare user input text
    user_text = ' '.join(user_input['interests']) + ' ' + user_input['budget'] + ' ' + user_input['season']
    user_text = preprocess_text(user_text)
    
    # Vectorize user input
    user_vector = vectorizer.transform([user_text])
    
    # Find similar destinations
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    similar_indices = similarities.argsort()[0][-4:][::-1]
    
    # Filter by budget and season
    filtered_destinations = []
    for idx in similar_indices:
        dest = destinations[idx]
        if dest['budget'].lower() == user_input['budget'].lower() and dest['season'].lower() == user_input['season'].lower():
            filtered_destinations.append(dest)
    
    # If not enough matches, relax constraints
    if len(filtered_destinations) < 3:
        for idx in similar_indices:
            dest = destinations[idx]
            if dest not in filtered_destinations and len(filtered_destinations) < 4:
                filtered_destinations.append(dest)
    
    return render_template('recommendations.html', destinations=filtered_destinations)

@app.route('/popular')
def popular():
    # Sort by popularity and rating
    sorted_destinations = sorted(destinations, key=lambda x: (x['popularity'], x['rating']), reverse=True)
    return render_template('popular.html', destinations=sorted_destinations[:20])

@app.route('/insights', methods=['GET', 'POST'])
def insights():
    if request.method == 'POST':
        dest_name = request.form.get('destination')
        destination = next((d for d in destinations if d['name'] == dest_name), None)
        
        if destination:
            # NLP analysis
            reviews = ' '.join(destination['reviews'])
            words = [word.lower() for word in re.findall(r'\b\w+\b', reviews) if word.lower() not in stopwords.words('english')]
            
            # Word frequency
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Ignore short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Top 10 words
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Sentiment analysis (simplified)
            positive_words = ['amazing', 'beautiful', 'love', 'recommend', 'excellent', 'unforgettable', 'wonderful', 'perfect']
            negative_words = ['crowded', 'expensive', 'hot', 'difficult', 'problem']
            
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            sentiment = {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': len(words) - positive_count - negative_count
            }
            
            return render_template('insights.html', 
                                  destinations=destinations,
                                  selected_dest=destination,
                                  top_words=top_words,
                                  sentiment=sentiment)
    
    return render_template('insights.html', destinations=destinations)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # Log contact (in a real app, you'd store this in a database)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('contact_log.txt', 'a') as f:
            f.write(f"{timestamp} - {name} ({email}): {message}\n")
        
        return render_template('contact_success.html')
    
    return render_template('contact.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    action = request.form.get('action')
    if action == 'recommendations':
        return redirect(url_for('index'))
    elif action == 'popular':
        return redirect(url_for('popular'))
    elif action == 'insights':
        return redirect(url_for('insights'))
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)