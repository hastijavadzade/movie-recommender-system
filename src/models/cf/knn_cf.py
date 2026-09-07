import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def build_user_item_matrix(ratings):

    user_item = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)
    return user_item


def compute_item_similarity(user_item_matrix, k=100):

    # transpose so rows = movies, cols = users
    item_vectors = user_item_matrix.T

    # cosine similarity between movies
    sim_matrix = cosine_similarity(item_vectors)

    sim_df = pd.DataFrame(
        sim_matrix, 
        index=item_vectors.index, 
        columns=item_vectors.index
    )
    
    # keep only top-k neighbours for each movie
    topk_sim = {}
    for movie in sim_df.index:
        # sort neighbours (skip itself)
        sims = sim_df[movie].drop(movie).sort_values(ascending=False).head(k)
        topk_sim[movie] = list(sims.items())
    
    return topk_sim

def recommend_item_knn(user_id, ratings, topk_sim, n, movies_df):

    user_ratings = ratings[ratings['userId'] == user_id]
    rated_movies = user_ratings['movieId'].values
    
    scores = {}

    # loop over rated movies
    for _, row in user_ratings.iterrows():
        movie, rating = row['movieId'], row['rating']
        
        if movie not in topk_sim:
            continue
        
        for sim_movie, sim_score in topk_sim[movie]:
            if sim_movie in rated_movies:
                continue  # skip already seen
            scores[sim_movie] = scores.get(sim_movie, 0) + sim_score * rating # this is just a recommendation score, not a predicted rating

    # rank by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

    if movies_df is not None:
        return [(movies_df.loc[movies_df['movieId']==mid, 'original_title'].values[0], score) 
                for mid, score in ranked]

    return ranked
