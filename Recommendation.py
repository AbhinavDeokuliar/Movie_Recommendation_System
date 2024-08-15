import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
# Assuming the dataset is in a CSV file with columns: user_id, movie_id, rating
data = pd.read_csv('movie_ratings.csv')

# Create user-item matrix
user_item_matrix = data.pivot_table(index='user_id', columns='movie_id', values='rating')

# Fill NaN values with 0
user_item_matrix.fillna(0, inplace=True)

# Compute user similarity matrix using cosine similarity
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function to predict ratings
def predict_ratings(user_id, movie_id):
    # Get the similarity scores for the user
    user_sim_scores = user_similarity_df[user_id]

    # Get the ratings for the movie by all users
    movie_ratings = user_item_matrix[movie_id]

    # Compute the weighted sum of ratings
    weighted_sum = np.dot(user_sim_scores, movie_ratings)

    # Compute the sum of the similarity scores
    sum_of_sim_scores = np.sum(user_sim_scores)

    # Predict the rating
    if sum_of_sim_scores == 0:
        return 0
    else:
        return weighted_sum / sum_of_sim_scores

# Function to recommend top N movies for a user
def recommend_movies(user_id, N=5):
    # Get the list of movies the user has not rated
    unrated_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] == 0].index.tolist()

    # Predict ratings for the unrated movies
    predicted_ratings = {movie_id: predict_ratings(user_id, movie_id) for movie_id in unrated_movies}

    # Sort the movies by predicted rating in descending order
    recommended_movies = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)

    # Return the top N recommended movies
    return recommended_movies[:N]

# Example usage
user_id = 1
recommended_movies = recommend_movies(user_id, N=5)
print(f"Top 5 recommended movies for user {user_id}: {recommended_movies}")