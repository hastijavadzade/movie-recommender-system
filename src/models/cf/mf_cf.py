from surprise import Dataset, Reader, SVD
import pickle

def train_mf(ratings):

    reader = Reader()
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()

    model = SVD(random_state=42)
    model.fit(trainset)

    return model

def recommend_mf(model, user_id, movies_df, ratings_df, n=10):

    # model = train_mf(ratings_df)

    seen = set(ratings_df[ratings_df['userId'] == user_id]['movieId']) # already watched movies

    candidates = movies_df[~movies_df['movieId'].isin(seen)].copy() # not watched

    candidates['pred_rating'] = candidates['movieId'].apply(
        lambda x: model.predict(user_id, x).est
    )

    # Top-N recommendations
    return candidates.sort_values('pred_rating', ascending=False).head(n)
