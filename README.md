# Movie-Recommendation-System
### Movie Recommendation System for "Kalki 2898 AD"

#### Project Objective
The goal of this project is to develop a recommendation system that suggests movies similar to "Kalki 2898 AD." This system aims to enhance user experience by offering personalized movie recommendations based on user preferences and past viewing history.

#### Data Source
The project uses the MovieLens dataset, which includes extensive information about user ratings, movie metadata, and user profiles. This dataset is widely used for building and evaluating movie recommendation systems.

#### Methodology
The project follows the YBI Foundation's methodology, which includes the following steps:

1. **Data Import and Description**
   - Import the MovieLens dataset and explore its structure.
   - Understand the types of data available, such as user ratings, movie details, and user demographics.

2. **Data Visualization**
   - Visualize the distribution of movie ratings to understand user rating behavior.
   - Plot the number of ratings per movie to identify popular movies.

3. **Data Preprocessing**
   - Merge the ratings and movies datasets to create a comprehensive dataset.
   - Handle any missing values to ensure the dataset is clean and ready for analysis.

4. **Collaborative Filtering Approach**
   - Use collaborative filtering techniques to recommend movies.
   - Apply Singular Value Decomposition (SVD), a popular algorithm for collaborative filtering, to the user-item interaction matrix.

5. **Model Training and Evaluation**
   - Split the data into training and testing sets to evaluate the model's performance.
   - Train the SVD model on the training set and evaluate it using metrics like Root Mean Squared Error (RMSE) on the test set.

6. **Generating Recommendations**
   - Predict user ratings for movies that they haven't seen yet.
   - Identify and recommend movies that are similar to "Kalki 2898 AD."

7. **Explanation of Results**
   - Interpret the model's recommendations and explain how similar movies to "Kalki 2898 AD" are identified.
   - Provide insights into the model's performance and potential areas for improvement.

### Results and Insights
- The recommendation system effectively uses the SVD algorithm to identify movies similar to "Kalki 2898 AD."
- The RMSE metric indicates the accuracy of the model's predictions.
- Visualizations provide insights into user rating behavior and movie popularity.

### Conclusion
This project demonstrates the development of a movie recommendation system focused on "Kalki 2898 AD." By leveraging collaborative filtering techniques and evaluating model performance, the system provides personalized movie suggestions, enhancing the user experience by recommending movies similar to "Kalki 2898 AD." Future improvements could include incorporating additional features like movie genres or user demographics to further enhance recommendation accuracy.
