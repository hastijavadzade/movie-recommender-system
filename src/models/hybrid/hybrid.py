import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def _minmax(series: pd.Series) -> pd.Series:

    if series is None or series.empty:
        return pd.Series(dtype=float)
    vals = series.values.reshape(-1, 1)
    scaled = MinMaxScaler().fit_transform(vals).reshape(-1)
    return pd.Series(scaled, index=series.index)

def blend_scores(scores_dict: dict, weights: dict) -> pd.Series:

    normed = {k: _minmax(v) for k, v in scores_dict.items() if v is not None}
    all_items = pd.Index([])
    for s in normed.values():
        all_items = all_items.union(s.index)
    final = pd.Series(0.0, index=all_items)
    for name, s in normed.items():
        w = weights.get(name, 0.0)
        final = final.add(s.reindex(all_items, fill_value=0.0) * w, fill_value=0.0)
    return final.sort_values(ascending=False)


def mf_scores_for_user(model, user_id: int, candidates: pd.Index) -> pd.Series:
    preds = [model.predict(uid=user_id, iid=int(mid)).est for mid in candidates]
    return pd.Series(preds, index=candidates)

def content_scores_for_user(user_profile_vec, item_matrix, item_index: pd.Index, movie_norms=None) -> pd.Series:
    profile_norm = np.linalg.norm(user_profile_vec)
    if profile_norm == 0:
        return pd.Series(0.0, index=item_index)

    sims = item_matrix @ user_profile_vec / (movie_norms * profile_norm + 1e-8)  # implementing cosine similarity manually for faster resulst
    return pd.Series(sims, index=item_index)
    # sims = cosine_similarity(user_profile_vec.reshape(1, -1), item_matrix).ravel()
    # return pd.Series(sims, index=item_index)

def recommend_hybrid(user_id, movies_df, ratings_df, movie_norms,
                     mf_model=None,
                     content_item_matrix=None,
                     content_item_index=None,
                     user_profile_vec=None,
                     weights={'mf': 0.7, 'content': 0.3},
                     n=10):
    seen = set(ratings_df.loc[ratings_df.userId == user_id, 'movieId'])
    candidates = pd.Index(set(movies_df['movieId']) - seen)

    mf_series = None
    if mf_model is not None:
        mf_series = mf_scores_for_user(mf_model, user_id, candidates)

    content_series = None
    if user_profile_vec is not None and content_item_matrix is not None:
        mask = content_item_index.isin(candidates)

        content_series = content_scores_for_user(
            user_profile_vec,
             content_item_matrix[mask],
            content_item_index[mask],
            movie_norms[mask]
        )

    if (mf_series is None) and (content_series is None):
        return movies_df[['movieId','original_title']].head(n)

    final = blend_scores({'mf': mf_series, 'content': content_series}, weights).head(n)

    # out = movies_df.set_index('movieId').loc[final.index, ['original_title','genres','cast','director']].copy()
    # out = movies_df.loc[final.index, ['movieId']].copy()

    # out['hybrid_score'] = final.values
    return final.index.tolist()