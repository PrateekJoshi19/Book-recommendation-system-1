import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re
import matplotlib.pyplot as plt
from PIL import Image
import base64
import requests
from io import BytesIO
import random
import os
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from pathlib import Path

class BookRecommendationSystem:
    def __init__(self):
        self.books_df = None
        self.ratings_df = None
        self.tfidf_matrix = None
        self.svd_model = None
        self.knn_model = None
        
    def load_data(self, books_path):
        """
        Load books data
        """
        print("Loading dataset...")
        
        # Load books data
        self.books_df = pd.read_csv(books_path)
        print(f"Books dataset loaded with {self.books_df.shape[0]} entries")
        
        # Generate synthetic ratings if none are provided
        self.generate_synthetic_ratings()
        
        # Display sample data
        print("\nSample books data:")
        print(self.books_df.head())
        
    def generate_synthetic_ratings(self, num_users=100, min_ratings_per_user=5, max_ratings_per_user=20):
        """
        Generate synthetic ratings for testing when real user ratings aren't available
        """
        print("Generating synthetic ratings for testing...")
        
        # Create a book_id column if not exists
        if 'book_id' not in self.books_df.columns:
            self.books_df['book_id'] = self.books_df.index
        
        # Generate random user IDs
        user_ids = list(range(1, num_users + 1))
        
        # Create empty ratings dataframe
        ratings_data = []
        
        # For each user, generate random ratings
        for user_id in user_ids:
            # Decide how many books this user will rate
            num_ratings = random.randint(min_ratings_per_user, max_ratings_per_user)
            
            # Select random books
            if num_ratings < len(self.books_df):
                books_to_rate = self.books_df.sample(num_ratings)
            else:
                books_to_rate = self.books_df
            
            # Generate ratings
            for _, book in books_to_rate.iterrows():
                # Generate a rating between 1 and 5
                rating = random.randint(1, 5)
                
                # Add to ratings data
                ratings_data.append({
                    'user_id': user_id,
                    'book_id': book['book_id'],
                    'rating': rating
                })
        
        # Create ratings dataframe
        self.ratings_df = pd.DataFrame(ratings_data)
        print(f"Generated {len(ratings_data)} synthetic ratings for {num_users} users")
        
    def preprocess_data(self):
        """
        Preprocess the data: clean, handle missing values, extract features
        """
        print("Preprocessing data...")
        
        # Clean book titles
        if 'Title' in self.books_df.columns:
            self.books_df['Title'] = self.books_df['Title'].str.strip()
        
        # Handle missing values
        for col in self.books_df.columns:
            if self.books_df[col].dtype == 'object':
                self.books_df[col] = self.books_df[col].fillna('')
            else:
                self.books_df[col] = self.books_df[col].fillna(0)
        
        # Create content features by combining relevant text columns
        # Adapt this to your specific column names
        content_columns = ['Title', 'Authors', 'Description', 'Category']
        content_columns = [col for col in content_columns if col in self.books_df.columns]
        
        if content_columns:
            self.books_df['content_features'] = self.books_df[content_columns].apply(
                lambda x: ' '.join(x.astype(str)), axis=1
            )
            
            # Clean the content features
            self.books_df['content_features'] = self.books_df['content_features'].apply(
                lambda x: re.sub(r'[^\w\s]', '', x.lower())
            )
        
        # Check and handle duplicate book IDs if any
        if 'book_id' in self.books_df.columns:
            if self.books_df.duplicated(subset=['book_id']).any():
                print("Warning: Duplicate book IDs found. Keeping the first occurrence.")
                self.books_df = self.books_df.drop_duplicates(
                    subset=['book_id'], 
                    keep='first'
                )
        else:
            # Create a book_id column if not exists
            self.books_df['book_id'] = self.books_df.index
        
        # Extract price values (remove $ and convert to float)
        if 'Price Starting With ($)' in self.books_df.columns:
            self.books_df['Price'] = self.books_df['Price Starting With ($)'].replace('[\$,]', '', regex=True).astype(float)
            
        # Create a publish_date column if year and month columns exist
        if 'Publish Date (Year)' in self.books_df.columns and 'Publish Date (Month)' in self.books_df.columns:
            self.books_df['publish_date'] = self.books_df['Publish Date (Year)'].astype(str) + '-' + \
                                           self.books_df['Publish Date (Month)'].astype(str)
        
        print("Data preprocessing completed")
        
    def build_content_based_model(self):
        """
        Build a content-based recommendation model using TF-IDF and cosine similarity
        """
        print("Building content-based recommendation model...")
        
        # Check if content features are available
        if 'content_features' not in self.books_df.columns:
            print("Error: Content features not found. Run preprocess_data() first.")
            return
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = tfidf.fit_transform(self.books_df['content_features'])
        print(f"TF-IDF matrix created with shape: {self.tfidf_matrix.shape}")
        
        print("Content-based model built successfully")
        
    def get_content_based_recommendations(self, book_id=None, title=None, top_n=10):
        """
        Get content-based recommendations for a specific book
        """
        # Validate input
        if self.tfidf_matrix is None:
            print("Error: Content-based model not built. Run build_content_based_model() first.")
            return pd.DataFrame()
        
        # Find the book index
        book_index = None
        
        if book_id is not None:
            if book_id in self.books_df['book_id'].values:
                book_index = self.books_df[self.books_df['book_id'] == book_id].index[0]
        elif title is not None:
            # Try to find the book by title (case-insensitive)
            matching_books = self.books_df[self.books_df['Title'].str.lower() == title.lower()]
            if not matching_books.empty:
                book_index = matching_books.index[0]
            else:
                # Try partial match
                matching_books = self.books_df[self.books_df['Title'].str.lower().str.contains(title.lower())]
                if not matching_books.empty:
                    book_index = matching_books.index[0]
        
        if book_index is None:
            print(f"Error: Book not found in the dataset.")
            return pd.DataFrame()
        
        # Calculate cosine similarity
        book_vector = self.tfidf_matrix[book_index:book_index+1]
        cosine_similarities = cosine_similarity(book_vector, self.tfidf_matrix).flatten()
        
        # Get top similar books (excluding the input book)
        similar_indices = cosine_similarities.argsort()[::-1][1:top_n+1]
        similar_books = self.books_df.iloc[similar_indices].copy()
        
        # Add similarity scores
        similar_books['similarity_score'] = cosine_similarities[similar_indices]
        
        return similar_books
    
    def build_collaborative_filtering_models(self):
        """
        Build collaborative filtering models: SVD for item-based and KNN for user-based
        """
        print("Building collaborative filtering models...")
        
        # Check if ratings data is available
        if self.ratings_df is None or self.ratings_df.empty:
            print("Error: Ratings data not available.")
            return
        
        # Configure Surprise reader
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings_df[['user_id', 'book_id', 'rating']], reader)
        
        # Split data
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        
        # Build SVD model (Item-based)
        print("Training SVD model (Item-based collaborative filtering)...")
        self.svd_model = SVD(n_factors=50, random_state=42)
        self.svd_model.fit(trainset)
        
        # Build KNN model (User-based)
        print("Training KNN model (User-based collaborative filtering)...")
        self.knn_model = KNNBasic(k=30, sim_options={'name': 'pearson', 'user_based': True})
        self.knn_model.fit(trainset)
        
        print("Collaborative filtering models built successfully")
        
    def get_svd_recommendations(self, user_id, top_n=10):
        """
        Get recommendations using the SVD model (item-based collaborative filtering)
        """
        if self.svd_model is None:
            print("Error: SVD model not built. Run build_collaborative_filtering_models() first.")
            return pd.DataFrame()
        
        # Get books the user has already rated
        user_rated_books = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['book_id'].values)
        
        # Get all books that the user hasn't rated
        candidate_books = [book for book in self.books_df['book_id'].values if book not in user_rated_books]
        
        # Predict ratings for all candidate books
        predictions = []
        for book_id in candidate_books:
            predicted_rating = self.svd_model.predict(user_id, book_id).est
            predictions.append((book_id, predicted_rating))
        
        # Sort by predicted rating and get top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_book_ids = [book_id for book_id, _ in predictions[:top_n]]
        
        # Get recommended books
        recommended_books = self.books_df[self.books_df['book_id'].isin(top_book_ids)].copy()
        
        # Add predicted ratings
        pred_ratings = {book_id: rating for book_id, rating in predictions[:top_n]}
        recommended_books['predicted_rating'] = recommended_books['book_id'].map(pred_ratings)
        
        return recommended_books.sort_values('predicted_rating', ascending=False)
    
    def get_knn_recommendations(self, user_id, top_n=10):
        """
        Get recommendations using the KNN model (user-based collaborative filtering)
        """
        if self.knn_model is None:
            print("Error: KNN model not built. Run build_collaborative_filtering_models() first.")
            return pd.DataFrame()
        
        # Get books the user has already rated
        user_rated_books = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['book_id'].values)
        
        # Get all books that the user hasn't rated
        candidate_books = [book for book in self.books_df['book_id'].values if book not in user_rated_books]
        
        # Predict ratings for all candidate books
        predictions = []
        for book_id in candidate_books:
            predicted_rating = self.knn_model.predict(user_id, book_id).est
            predictions.append((book_id, predicted_rating))
        
        # Sort by predicted rating and get top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_book_ids = [book_id for book_id, _ in predictions[:top_n]]
        
        # Get recommended books
        recommended_books = self.books_df[self.books_df['book_id'].isin(top_book_ids)].copy()
        
        # Add predicted ratings
        pred_ratings = {book_id: rating for book_id, rating in predictions[:top_n]}
        recommended_books['predicted_rating'] = recommended_books['book_id'].map(pred_ratings)
        
        return recommended_books.sort_values('predicted_rating', ascending=False)
    
    def get_category_based_recommendations(self, category, top_n=10):
        """
        Get recommendations based on a specific category/genre
        """
        if 'Category' not in self.books_df.columns:
            print("Error: Category column not found in the dataset.")
            return pd.DataFrame()
        
        # Filter books by category (case-insensitive)
        category_books = self.books_df[self.books_df['Category'].str.lower() == category.lower()]
        
        if category_books.empty:
            # Try partial match
            category_books = self.books_df[self.books_df['Category'].str.lower().str.contains(category.lower())]
            
        if category_books.empty:
            print(f"Error: No books found for category '{category}'.")
            return pd.DataFrame()
        
        # Sort by publish date (newest first) if available
        if 'publish_date' in self.books_df.columns:
            category_books = category_books.sort_values('publish_date', ascending=False)
        
        return category_books.head(top_n)
    
    def get_popularity_based_recommendations(self, top_n=10):
        """
        Get recommendations based on popularity (rating counts and average rating)
        """
        if self.ratings_df is None or self.ratings_df.empty:
            print("Error: Ratings data not available.")
            return pd.DataFrame()
        
        # Calculate popularity metrics
        popularity = self.ratings_df.groupby('book_id').agg(
            rating_count=('rating', 'count'),
            avg_rating=('rating', 'mean')
        ).reset_index()
        
        # Calculate a popularity score
        popularity['popularity_score'] = popularity['avg_rating'] * np.log1p(popularity['rating_count'])
        
        # Sort by popularity score
        popularity = popularity.sort_values('popularity_score', ascending=False)
        
        # Get top N popular book IDs
        top_book_ids = popularity.head(top_n)['book_id'].values
        
        # Get book details
        popular_books = self.books_df[self.books_df['book_id'].isin(top_book_ids)].copy()
        
        # Add popularity metrics
        popular_books = popular_books.merge(popularity, on='book_id')
        
        return popular_books.sort_values('popularity_score', ascending=False)

    def get_ensemble_recommendations(self, user_id, book_id=None, title=None, top_n=10, weights=(0.3, 0.3, 0.4)):
        """
        Get ensemble recommendations combining content-based, user-based, and item-based filtering
        weights: tuple of (content_weight, user_based_weight, item_based_weight)
        """
        recommendations = {}
        
        # Get content-based recommendations if book_id or title is provided
        if (book_id is not None or title is not None) and weights[0] > 0:
            content_recs = self.get_content_based_recommendations(book_id=book_id, title=title, top_n=top_n*2)
            if not content_recs.empty:
                for _, row in content_recs.iterrows():
                    book = row['book_id']
                    score = row['similarity_score'] * weights[0]
                    recommendations[book] = recommendations.get(book, 0) + score
        
        # Get user-based (KNN) recommendations
        if weights[1] > 0:
            user_recs = self.get_knn_recommendations(user_id, top_n=top_n*2)
            if not user_recs.empty:
                for _, row in user_recs.iterrows():
                    book = row['book_id']
                    score = (row['predicted_rating'] / 5.0) * weights[1]
                    recommendations[book] = recommendations.get(book, 0) + score
        
        # Get item-based (SVD) recommendations
        if weights[2] > 0:
            item_recs = self.get_svd_recommendations(user_id, top_n=top_n*2)
            if not item_recs.empty:
                for _, row in item_recs.iterrows():
                    book = row['book_id']
                    score = (row['predicted_rating'] / 5.0) * weights[2]
                    recommendations[book] = recommendations.get(book, 0) + score
        
        # Sort and get top recommendations
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Get book details
        top_book_ids = [book_id for book_id, _ in sorted_recs]
        recommended_books = self.books_df[self.books_df['book_id'].isin(top_book_ids)].copy()
        
        # Add ensemble scores
        ensemble_scores = {book_id: score for book_id, score in sorted_recs}
        recommended_books['ensemble_score'] = recommended_books['book_id'].map(ensemble_scores)
        
        return recommended_books.sort_values('ensemble_score', ascending=False)

# Create the Streamlit app
def create_streamlit_app():
    st.set_page_config(
        page_title="Book Recommendation System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize the recommendation system
    recommender = BookRecommendationSystem()
    
    # Add title and description
    st.title("📚 Book Recommendation System")
    st.markdown("""
    Find your next favorite book with our intelligent recommendation system. 
    Discover new books based on your preferences, similar books, or popular titles!
    """)
    
    # Sidebar for dataset loading and model building
    with st.sidebar:
        st.header("Settings")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload Books Dataset (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            with open("temp_books.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load the dataset
            if st.button("Load Dataset"):
                with st.spinner("Loading dataset..."):
                    recommender.load_data("temp_books.csv")
                    recommender.preprocess_data()
                    st.session_state['data_loaded'] = True
                    st.success("Dataset loaded successfully!")
        
        # Option to use demo dataset
        if st.button("Use Demo Dataset"):
            if os.path.exists("demo_books.csv"):
                with st.spinner("Loading demo dataset..."):
                    recommender.load_data("demo_books.csv")
                    recommender.preprocess_data()
                    st.session_state['data_loaded'] = True
                    st.success("Demo dataset loaded successfully!")
            else:
                st.error("Demo dataset not found. Please upload a dataset.")
        
        # Build models
        if 'data_loaded' in st.session_state and st.session_state['data_loaded']:
            if st.button("Build Recommendation Models"):
                with st.spinner("Building content-based model..."):
                    recommender.build_content_based_model()
                    st.session_state['content_model_built'] = True
                
                with st.spinner("Building collaborative filtering models..."):
                    recommender.build_collaborative_filtering_models()
                    st.session_state['collab_models_built'] = True
                
                st.success("All models built successfully!")
    
    # Main area with tabs
    if 'data_loaded' in st.session_state and st.session_state['data_loaded']:
        tabs = st.tabs(["Browse Books", "Get Recommendations", "User Profile", "About"])
        
        # Tab 1: Browse Books
        with tabs[0]:
            st.header("Browse Books")
            
            # Show dataset statistics
            st.subheader("Dataset Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Books", len(recommender.books_df))
            with col2:
                if 'Category' in recommender.books_df.columns:
                    st.metric("Categories", recommender.books_df['Category'].nunique())
                else:
                    st.metric("Categories", "N/A")
            with col3:
                if 'Publisher' in recommender.books_df.columns:
                    st.metric("Publishers", recommender.books_df['Publisher'].nunique())
                else:
                    st.metric("Publishers", "N/A")
            
            # Category filter
            if 'Category' in recommender.books_df.columns:
                categories = ['All'] + sorted(recommender.books_df['Category'].unique().tolist())
                selected_category = st.selectbox("Filter by Category", categories)
            
            # Price range filter
            if 'Price' in recommender.books_df.columns:
                min_price = float(recommender.books_df['Price'].min())
                max_price = float(recommender.books_df['Price'].max())
                price_range = st.slider("Price Range", min_price, max_price, (min_price, max_price))
            
            # Apply filters
            filtered_df = recommender.books_df.copy()
            
            if 'Category' in recommender.books_df.columns and selected_category != 'All':
                filtered_df = filtered_df[filtered_df['Category'] == selected_category]
            
            if 'Price' in recommender.books_df.columns:
                filtered_df = filtered_df[(filtered_df['Price'] >= price_range[0]) & 
                                         (filtered_df['Price'] <= price_range[1])]
            
            # Display books
            st.subheader(f"Showing {len(filtered_df)} Books")
            
            # Book display
            books_per_row = 3
            for i in range(0, len(filtered_df), books_per_row):
                cols = st.columns(books_per_row)
                for j in range(books_per_row):
                    if i + j < len(filtered_df):
                        book = filtered_df.iloc[i + j]
                        with cols[j]:
                            st.subheader(book['Title'])
                            st.markdown(f"**Author:** {book['Authors'] if 'Authors' in book else 'Unknown'}")
                            st.markdown(f"**Category:** {book['Category'] if 'Category' in book else 'Unknown'}")
                            if 'Price' in book:
                                st.markdown(f"**Price:** ${book['Price']:.2f}")
                            st.markdown("---")
        
        # Tab 2: Get Recommendations
        with tabs[1]:
            st.header("Get Book Recommendations")
            
            # Check if models are built
            models_ready = ('content_model_built' in st.session_state and 
                           'collab_models_built' in st.session_state)
            
            if not models_ready:
                st.warning("Please build the recommendation models first using the sidebar.")
            else:
                # Recommendation method selection
                rec_method = st.radio(
                    "Choose Recommendation Method",
                    ["Based on a Book", "Based on User Profile", "Based on Category", "Popular Books", "Ensemble (Combined)"]
                )
                
                if rec_method == "Based on a Book":
                    # Book-based recommendations
                    st.subheader("Find Similar Books")
                    book_titles = [''] + sorted(recommender.books_df['Title'].tolist())
                    selected_title = st.selectbox("Select a Book", book_titles)
                    
                    if selected_title and st.button("Get Similar Books"):
                        with st.spinner("Finding similar books..."):
                            similar_books = recommender.get_content_based_recommendations(title=selected_title)
                            
                            if not similar_books.empty:
                                st.success(f"Here are books similar to '{selected_title}'")
                                
                                # Display similar books
                                for i, (_, book) in enumerate(similar_books.iterrows()):
                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        st.markdown(f"### {i+1}.")
                                    with col2:
                                        st.subheader(book['Title'])
                                        st.markdown(f"**Author:** {book['Authors'] if 'Authors' in book else 'Unknown'}")
                                        st.markdown(f"**Category:** {book['Category'] if 'Category' in book else 'Unknown'}")
                                        st.markdown(f"**Similarity Score:** {book['similarity_score']:.2f}")
                                        if 'Description' in book and book['Description']:
                                            with st.expander("Description"):
                                                st.write(book['Description'])
                                        st.markdown("---")
                            else:
                                st.error("No similar books found.")
                
                elif rec_method == "Based on User Profile":
                    # User-based recommendations
                    st.subheader("Recommendations Based on Your Profile")
                    user_id = st.number_input("Enter User ID", min_value=1, max_value=100, value=1)
                    
                    if st.button("Get Recommendations"):
                        with st.spinner("Finding books you might like..."):
                            user_recs = recommender.get_knn_recommendations(user_id)
                            
                            if not user_recs.empty:
                                st.success(f"Here are books recommended for User {user_id}")
                                
                                # Display recommended books
                                for i, (_, book) in enumerate(user_recs.iterrows()):
                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        st.markdown(f"### {i+1}.")
                                    with col2:
                                        st.subheader(book['Title'])
                                        st.markdown(f"**Author:** {book['Authors'] if 'Authors' in book else 'Unknown'}")
                                        st.markdown(f"**Category:** {book['Category'] if 'Category' in book else 'Unknown'}")
                                        st.markdown(f"**Predicted Rating:** {book['predicted_rating']:.2f}/5.0")
                                        if 'Description' in book and book['Description']:
                                            with st.expander("Description"):
                                                st.write(book['Description'])
                                        st.markdown("---")
                            else:
                                st.error("No recommendations found for this user.")
                
                elif rec_method == "Based on Category":
                    # Category-based recommendations
                    st.subheader("Browse Books by Category")
                    
                    if 'Category' in recommender.books_df.columns:
                        categories = sorted(recommender.books_df['Category'].unique().tolist())
                        selected_category = st.selectbox("Select a Category", categories)
                        
                        if selected_category and st.button("Browse Category"):
                            with st.spinner(f"Finding top books in {selected_category}..."):
                                category_books = recommender.get_category_based_recommendations(selected_category)
                                
                                if not category_books.empty:
                                    st.success(f"Top books in the '{selected_category}' category")
                                    
                                    # Display category books
                                    for i, (_, book) in enumerate(category_books.iterrows()):
                                        col1, col2 = st.columns([1, 3])
                                        with col1:
                                            st.markdown(f"### {i+1}.")
                                        with col2:
                                            st.subheader(book['Title'])
                                            st.markdown(f"**Author:** {book['Authors'] if 'Authors' in book else 'Unknown'}")
                                            if 'Publisher' in book:
                                                st.markdown(f"**Publisher:** {book['Publisher']}")
                                            if 'Price' in book:
                                                st.markdown(f"**Price:** ${book['Price']:.2f}")
                                            if 'Description' in book and book['Description']:
                                                with st.expander("Description"):
                                                    st.write(book['Description'])
                                            st.markdown("---")
                                else:
                                    st.error(f"No books found in the '{selected_category}' category.")
                    else:
                        st.error("Category information not available in the dataset.")
                
                elif rec_method == "Popular Books":
                    # Popularity-based recommendations
                    st.subheader("Most Popular Books")
                    
                    if st.button("Show Popular Books"):
                        with st.spinner("Finding most popular books..."):
                            popular_books = recommender.get_popularity_based_recommendations()
                            
                            if not popular_books.empty:
                                st.success("Here are the most popular books")
                                
                                # Display popular books
                                for i, (_, book) in enumerate(popular_books.iterrows()):
                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        st.markdown(f"### {i+1}.")
                                    with col2:
                                        st.subheader(book['Title'])
                                        st.markdown(f"**Author:** {book['Authors'] if 'Authors' in book else 'Unknown'}")
                                        st.markdown(f"**Category:** {book['Category'] if 'Category' in book else 'Unknown'}")
                                        st.markdown(f"**Average Rating:** {book['avg_rating']:.2f}/5.0 ({book['rating_count']} ratings)")
                                        if 'Description' in book and book['Description']:
                                            with st.expander("Description"):
                                                st.write(book['Description'])
                                        st.markdown("---")
                            else:
                                st.error("Could not determine popular books.")
                
                elif rec_method == "Ensemble (Combined)":
                    # Ensemble recommendations
                    st.subheader("Smart Recommendations (Combined Methods)")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        user_id = st.number_input("Enter User ID", min_value=1, max_value=100, value=1)
                
if __name__ == "__main__":
    create_streamlit_app()
