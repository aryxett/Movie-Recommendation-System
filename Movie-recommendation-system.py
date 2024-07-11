Title of Project
Movie Recommendation System for "Kalki 2898 AD"

Objective
To develop a recommendation system that suggests movies similar to "Kalki 2898 AD" using collaborative filtering techniques.

Data Source
The data for this project is sourced from the MovieLens dataset, which includes user ratings, movie metadata, and user information.

Import Library
python
Copy code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
Import Data
python
Copy code
# Load the datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
Describe Data
python
Copy code
# Display the first few rows of the datasets
print(movies.head())
print(ratings.head())

# Get a summary of the datasets
print(movies.info())
print(ratings.info())

# Get descriptive statistics
print(ratings.describe())
Data Visualization
python
Copy code
# Distribution of movie ratings
plt.figure(figsize=(8, 6))
sns.histplot(ratings['rating'], bins=20, kde=False)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Number of ratings per movie
ratings_per_movie = ratings.groupby('movieId').size()
plt.figure(figsize=(8, 6))
sns.histplot(ratings_per_movie, bins=50, kde=False)
plt.title('Number of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Count')
plt.show()
Data Preprocessing
python
Copy code
# Merge movies and ratings datasets
data = pd.merge(ratings, movies, on='movieId')

# Handle missing values (if any)
data = data.dropna()
Define Target Variable (y) and Feature Variables (X)
For a recommendation system, we typically use collaborative filtering methods rather than defining explicit target and feature variables. We use user-item interaction matrices.

Train Test Split
python
Copy code
# Load data into Surprise library's format
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split data into training and testing sets
trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)
Modeling
python
Copy code
# Initialize the SVD model
model = SVD()

# Train the model
model.fit(trainset)
Model Evaluation
python
Copy code
# Make predictions on the test set
predictions = model.test(testset)

# Calculate evaluation metrics
mse = accuracy.mse(predictions)
rmse = accuracy.rmse(predictions)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
Prediction
To recommend movies similar to "Kalki 2898 AD," we need to first identify the movie ID for "Kalki 2898 AD" in the MovieLens dataset. Let's assume the movie ID is 1 for this example.

python
Copy code
# Define the movie ID for "Kalki 2898 AD"
kalki_movie_id = 1  # Replace with the actual movie ID

# Function to get top N similar movies
def get_similar_movies(movie_id, model, data, top_n=10):
    # Get the list of movie IDs
    movie_ids = data['movieId'].unique()
    
    # Predict ratings for all movies for a generic user
    predictions = [model.predict(0, movie_id).est for movie_id in movie_ids]
    
    # Create a DataFrame with movie IDs and predicted ratings
    pred_df = pd.DataFrame({'movieId': movie_ids, 'predicted_rating': predictions})
    
    # Get the top N similar movies
    similar_movies = pred_df.sort_values(by='predicted_rating', ascending=False).head(top_n)
    
    return similar_movies

# Get top 10 movies similar to "Kalki 2898 AD"
similar_movies = get_similar_movies(kalki_movie_id, model, data)
print(similar_movies)
Explanation
Results and Insights:

The SVD model, a collaborative filtering algorithm, was used to build the recommendation system.
The RMSE metric was used to evaluate the model's performance, with a lower RMSE indicating better prediction accuracy.
The model can recommend movies similar to "Kalki 2898 AD" by predicting user ratings for other movies and identifying the top similar movies.
Visualizations of the rating distribution and the number of ratings per movie provide insights into user behavior and movie popularity, which can inform further refinements to the recommendation system.