from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import uuid
import os
import ast

app = Flask(__name__)

# Load the books dataset
books_df = pd.read_csv('BooksDataset.csv')

# Prepare data for the recommendation model
def prepare_data():
    def parse_genres(category_str):
        try:
            genres = ast.literal_eval(category_str)
            if isinstance(genres, list) and all(isinstance(g, str) for g in genres):
                return genres
            elif isinstance(genres, str):
                return [genres]
            else:
                return []
        except:
            return []

    books_df['genres'] = books_df['Category'].fillna('[]').apply(parse_genres)

    genre_list = ['Fiction', 'Non-Fiction', 'Mystery', 'Science Fiction', 'Fantasy']
    for genre in genre_list:
        books_df[f'genre_{genre}'] = books_df['genres'].apply(lambda x: 1 if genre in x else 0)

    books_df['publication_year'] = pd.to_numeric(books_df['Publish Date (Year)'], errors='coerce')

    numerical_features = ['Price Starting With ($)', 'publication_year']
    for feature in numerical_features:
        books_df[feature] = (books_df[feature] - books_df[feature].min()) / (books_df[feature].max() - books_df[feature].min())

    features = [col for col in books_df.columns if col.startswith('genre_')] + ['Price Starting With ($)', 'publication_year']
    X = books_df[features].fillna(0)
    y = books_df['Price Starting With ($)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    return model, books_df, genre_list

model, books_df, genre_list = prepare_data()

# Homepage
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint to get recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    user_answers = request.json['answers']

    user_profile = {
        'read_frequency': user_answers[0],
        'genre_preferences': user_answers[1:6],
        'length_importance': user_answers[6],
        'rating_importance': user_answers[7],
        'english_preference': user_answers[8],
        'author_importance': user_answers[9],
        'recency_preference': user_answers[10],
        'publication_year_importance': user_answers[11],
        'foreign_preference': user_answers[12],
        'reviews_importance': user_answers[13],
        'format_preference': user_answers[14]
    }

    books_df['user_score'] = 0

    for i, genre in enumerate(genre_list):
        books_df['user_score'] += books_df[f'genre_{genre}'] * user_profile['genre_preferences'][i]

    books_df['user_score'] += books_df['Price Starting With ($)'] * user_profile['length_importance']
    books_df['user_score'] += books_df['publication_year'] * user_profile['recency_preference']

    books_df['user_score'] = (books_df['user_score'] - books_df['user_score'].min()) / (books_df['user_score'].max() - books_df['user_score'].min())

    books_df['final_score'] = 0.7 * books_df['user_score'] + 0.3 * books_df['Price Starting With ($)']

    recommended_books = books_df.sort_values('final_score', ascending=False).head(10)
    return jsonify(recommended_books[['Title', 'final_score', 'genres', 'Price Starting With ($)']].to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)