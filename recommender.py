# recommender.py
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_model():
    return joblib.load('model_artifacts.pkl')

def get_recommendations(user_input, destinations):
    vectorizer, kmeans, tfidf_matrix = load_model()
    
    # Preprocess user input
    user_text = ' '.join(user_input['interests']) + ' ' + user_input['budget'] + ' ' + user_input['season']
    user_vector = vectorizer.transform([user_text])
    
    # Find similar destinations
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    similar_indices = similarities.argsort()[0][-4:][::-1]
    
    # Filter results
    recommendations = []
    for idx in similar_indices:
        dest = destinations[idx]
        if dest['budget'].lower() == user_input['budget'].lower() and dest['season'].lower() == user_input['season'].lower():
            recommendations.append(dest)
    
    return recommendations[:4]