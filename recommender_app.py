import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Movie Recommender System", layout="wide")

# Sample movie dataset
movie_data = pd.DataFrame({
    "Movie ID": [1, 2, 3, 4],
    "Title": ["Inception", "Titanic", "Avatar", "The Matrix"],
    "Genre": ["Sci-Fi", "Romance", "Sci-Fi", "Action"]
})

# Sample user-item matrix with movies
ratings_matrix = np.array([
    [5, 3, np.nan, 1],  # User 1
    [4, np.nan, 4, 1],  # User 2
    [1, 1, np.nan, 5],  # User 3
    [np.nan, 5, 4, 4]   # User 4
])
movie_titles = movie_data["Title"].tolist()

# App title
st.title("🍿 Netflix-Style Movie Recommender System")
st.write(
    """
    Welcome to your personalized movie recommender system! This tool predicts movies you might enjoy 
    based on your ratings and preferences, similar to how Netflix recommends content.
    """
)

# Show current movie dataset
st.subheader("Available Movies")
st.write(movie_data)

# Sidebar for user input
st.sidebar.header("Rate Movies")
st.sidebar.write("Rate the movies you've watched. Leave the slider at **0.0** if you haven't watched a movie.")

# Input user ratings for movies
new_user_ratings = []
for i, movie in enumerate(movie_titles):
    # Sliders for user ratings (Default value is 0.0, representing "not rated")
    rating = st.sidebar.slider(
        f"{movie} ({movie_data['Genre'][i]})", 
        min_value=0.0, 
        max_value=5.0, 
        step=0.5, 
        value=0.0,  # Default value
        format="%.1f"
    )
    new_user_ratings.append(rating)

# Convert ratings to numpy array
new_user = np.array(new_user_ratings)

# Validation: Check if all ratings are 0.0
if np.all(new_user == 0.0):
    st.warning("Please rate at least one movie to get personalized recommendations.")
else:
    # Treat 0.0 ratings as missing (convert to NaN for prediction logic)
    new_user_with_nan = np.where(new_user == 0.0, np.nan, new_user)

    # Add the new user to the ratings matrix
    ratings_with_new_user = np.vstack([ratings_matrix, new_user_with_nan])

    # Replace NaN with 0 for similarity calculations
    filled_ratings = np.nan_to_num(ratings_with_new_user)

    # Calculate cosine similarity
    similarity = cosine_similarity(filled_ratings)

    # Predict ratings
    def predict_ratings(ratings, similarity):
        mean_ratings = np.nanmean(ratings, axis=1, keepdims=True)
        ratings_diff = ratings - mean_ratings
        ratings_diff = np.nan_to_num(ratings_diff)  # Replace NaN with 0

        predictions = mean_ratings + similarity @ ratings_diff / np.abs(similarity).sum(axis=1, keepdims=True)
        return predictions

    predicted_ratings = predict_ratings(ratings_with_new_user, similarity)
    new_user_predictions = predicted_ratings[-1]

    # Most similar user
    similarity_to_new_user = similarity[-1, :-1]  # Exclude self-similarity
    most_similar_user_idx = np.argmax(similarity_to_new_user)
    most_similar_user_score = similarity_to_new_user[most_similar_user_idx]

    # Display recommendations
    st.subheader("🎯 Personalized Recommendations")
    top_recommendations = [
        (movie_titles[i], new_user_predictions[i]) for i in range(len(new_user_predictions)) if new_user_with_nan[i] != new_user_with_nan[i]  # Check for NaN
    ]
    top_recommendations = sorted(top_recommendations, key=lambda x: x[1], reverse=True)[:3]

    if top_recommendations:
        st.write("Based on your preferences, you might enjoy these movies:")
        for movie, pred_rating in top_recommendations:
            st.write(f"- **{movie}** (Predicted Rating: {pred_rating:.2f})")
    else:
        st.write("You have already rated all available movies!")

    # Display most similar user
    st.subheader("🧑‍🤝‍🧑 Closest Match")
    st.write(
        f"The new user is most similar to **User {most_similar_user_idx + 1}** "
        f"with a similarity score of **{most_similar_user_score:.2f}**."
    )

    # Similarity heatmap
    st.subheader("🔍 User Similarity Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        similarity, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
        xticklabels=[f"User {i+1}" for i in range(len(filled_ratings))],
        yticklabels=[f"User {i+1}" for i in range(len(filled_ratings))]
    )
    plt.title("User Similarity Matrix")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.write(
    """
    **About**: This tool helps businesses and users leverage collaborative filtering to make personalized recommendations. 
    Created with ❤️ using [Streamlit](https://streamlit.io).
    """
)
