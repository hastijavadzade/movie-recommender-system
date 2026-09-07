def compute_movie_stats(ratings, movies):
    # Compute average rating and number of ratings per movie.
    # Returns a DataFrame with movieId, avg_rating, num_ratings, and title.
    movie_stats = ratings.groupby('movieId').agg(
        avg_rating=('rating','mean'),
        num_ratings=('rating','count')
    ).reset_index()

    # Merge with movie titles for readability
    movie_stats = movie_stats.merge(movies[['movieId','original_title']], on='movieId', how='left')
    movie_stats = movie_stats.dropna(subset=['original_title'])

    return movie_stats

def top_avg_rating(movie_stats, min_ratings=50, k=10):
    # Sort by average rating (with a minimum rating count threshold to avoid noise)
    q_movies = movie_stats.copy().loc[movie_stats['num_ratings'] >= min_ratings]
    top_avg = q_movies.sort_values('avg_rating', ascending=False).head(k)
    # top_avg = movie_stats.query("num_ratings >= @min_ratings").sort_values('avg_rating', ascending=False).head(k)

    return top_avg[['original_title','avg_rating','num_ratings']]


def top_weighted_rating(movie_stats):
    # Compute global average rating (C) and threshold m
    C = movie_stats['avg_rating'].mean()
    m = movie_stats['num_ratings'].quantile(0.80)

    q_movies = movie_stats.copy().loc[movie_stats['num_ratings'] >= m]

    def weighted_rating(x, m=m, C=C):
        v = x['num_ratings']
        R = x['avg_rating']
        return (v/(v+m) * R) + (m/(v+m) * C)

    q_movies['wr'] = q_movies.apply(weighted_rating, axis=1)

    # Top movies by WR
    q_movies = q_movies.sort_values('wr', ascending=False).head(10)

    return q_movies[['original_title','wr','num_ratings','avg_rating']]

def top_weighted_rating_tmdb(movies):
    # Compute global average rating (C) and threshold m
    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(0.80)

    q_movies = movies.copy().loc[movies['vote_count'] >= m]

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(v+m) * C)

    q_movies['wr'] = q_movies.apply(weighted_rating, axis=1)

    # Top movies by WR
    q_movies = q_movies.sort_values('wr', ascending=False).head(10)

    return q_movies[['original_title','wr','vote_count','vote_average']]
