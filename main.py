from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

data = Dataset.load_builtin('ml-100k')

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

sim_options = {'name': 'cosine', 'user_based': False}
model = KNNBasic(sim_options=sim_options)

model.fit(trainset)

predictions = model.test(testset)

accuracy.rmse(predictions)

movies = pd.read_csv('./ml-100k/u.item', sep='|', encoding='latin-1', header=None, names=['movieId', 'title'], usecols=[0, 1])

user_id = str(1)
movie_recommendations = []
for movie_id in range(1, 6):
    prediction = model.predict(user_id, str(movie_id))
    movie_name = movies.loc[movies['movieId'] == int(movie_id), 'title'].values[0]
    movie_recommendations.append((movie_name, prediction.est))

top_recommendations = sorted(movie_recommendations, key=lambda x: x[1], reverse=True)[:3]
print("Top 3 Movie Recommendations:")
for movie_name, score in top_recommendations:
    print(f"Movie Name: {movie_name}, Predicted Rating: {score}")
