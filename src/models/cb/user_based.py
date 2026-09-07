from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import ast

def build_user_profile(user_id, ratings, matrix, movie_indices):
    user_data = ratings[ratings['userId'] == user_id]
    if user_data.empty: # cold start
        return None
    
    # Center ratings
    mean_rating = user_data['rating'].mean()
    user_data = user_data.copy()
    user_data['weight'] = user_data['rating'] - mean_rating
    
    # Weighted sum of movie vectors
    vectors = []
    for _, row in user_data.iterrows():
        
        if row['movieId'] in movie_indices:
            idx = movie_indices.get_loc(row['movieId'])   # row position in matrix
            vec = matrix[idx]
            if hasattr(vec, "toarray"):
                vec = vec.toarray().ravel()
            vectors.append(vec * row['weight'])

    if not vectors:
        return None
    
    profile = np.mean(vectors, axis=0)
    return profile

def recommend_cb(user_id, ratings, movie_matrix, movie_ids, movie_indices, movie_norms, k=10):
    profile = build_user_profile(user_id, ratings, movie_matrix, movie_indices)
    if profile is None:
        return []

    # Cosine similarity
    profile_norm = np.linalg.norm(profile)
    sim_scores = movie_matrix @ profile / (movie_norms * profile_norm + 1e-8) # implementing cosine similarity manually for faster resulst 

    # Remove movies the user has already seen
    seen = set(ratings[ratings['userId'] == user_id]['movieId'])
    mask = np.array([mid not in seen for mid in movie_ids])
    sim_scores = sim_scores[mask]
    candidate_ids = movie_ids[mask]

    if len(sim_scores) <= k:
        topk_ids = candidate_ids[np.argsort(-sim_scores)]
    else:
        topk_idx = np.argpartition(-sim_scores, k)[:k]
        topk_ids = candidate_ids[topk_idx[np.argsort(-sim_scores[topk_idx])]]

    return topk_ids.tolist()

# def explain_recommendation(user_id, rec_movie, ratings, movies):
#     user_movies = ratings[ratings['userId'] == user_id].merge(movies, on="movieId")
#     liked = user_movies[user_movies['rating'] >= user_movies['rating'].mean()]
    
#     explanation = []
#     for col in ['genres', 'cast', 'director']:
#         overlap = set(rec_movie[col].explode()) & set(liked[col].explode().unique())
#         if overlap:
#             explanation.append(f"shares {col}: {', '.join(overlap)}")
    
#     return "; ".join(explanation) if explanation else "similar themes"


def to_list(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass

        return [x.strip() for x in value.split(",") if x.strip()]

    return []


def explain_recommendation(user_id, rec_movie, ratings, movies):
    user_movies = ratings[ratings['userId'] == user_id].merge(
        movies,
        on="movieId"
    )

    if user_movies.empty:
        return "Recommended based on your previous movie ratings."

    mean_rating = user_movies['rating'].mean()
    liked = user_movies[user_movies['rating'] >= mean_rating]

    explanations = []

    for col in ['genres', 'cast', 'director']:
        rec_values = set(to_list(rec_movie.get(col)))

        liked_values = set()
        for value in liked[col]:
            liked_values.update(to_list(value))

        overlap = rec_values & liked_values

        if overlap:
            items = ", ".join(list(overlap)[:3])
            explanations.append(f"similar {col}: {items}")

    if explanations:
        return " • ".join(explanations)

    return "Recommended based on your previous movie ratings."

def recommend_unified(user_id, ratings, movies, matrix, movie_ids, movie_indices, movie_norms, n=10, selected_genres=None):

    # Cold start with genre input
    if selected_genres:
        subset = movies[movies['genres'].apply(lambda g: any(genre in g for genre in selected_genres))]
        if not subset.empty:
            return subset.sample(min(n, len(subset)))[['original_title', 'genres', 'cast', 'director']]
        
    # Check if user has ratings
    user_ratings = ratings[ratings['userId'] == user_id]
    if not user_ratings.empty:
        # Personalized recommendation
        return recommend_cb(user_id, ratings, matrix, movie_ids, movie_indices, movie_norms, k=10)

    # Popularity fallback
    return movies.sort_values('vote_count', ascending=False).head(n)[['original_title', 'genres', 'cast', 'director']]
