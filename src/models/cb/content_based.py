from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from sklearn.decomposition import TruncatedSVD

def compute_tf_matrix(movies):
    tf = TfidfVectorizer(max_features=5000, stop_words='english')
    tf_matrix = tf.fit_transform(movies['metatags'])

    return tf_matrix

def compute_cv_matrix(movies):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    cv_matrix = cv.fit_transform(movies['metatags'])

    return cv_matrix

def compute_truncatedSVD_matrix(tf_matrix):
    # reducing dimensionality with truncated SVD
    svd = TruncatedSVD(n_components=200)   # keep top 200 latent features
    reduced_matrix = svd.fit_transform(tf_matrix)

    return reduced_matrix

def recommend_movie(movies, title, matrix):

    title = title.lower()

    # Find index (first match)
    matches = movies[movies['title_lower'].str.contains(title)]
    if matches.empty:
        return ["No match found."]
    idx = matches.index[0]
    
    similar_list = cosine_similarity(matrix[idx:idx+1], matrix).flatten()
    
    # Sort by similarity
    sim_indices = similar_list.argsort()[::-1]

    # Remove the movie itself
    sim_indices = [i for i in sim_indices if i != idx]

    # Take top 10 results
    sim_indices = sim_indices[:10]
    
    return movies['movieId'].iloc[sim_indices].tolist()