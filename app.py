import gradio as gr
import pandas as pd
import pickle
import numpy as np
from scipy.sparse import load_npz
import os
import ast

from src.models.cb.content_based import recommend_movie
from src.models.hybrid.hybrid import recommend_hybrid
from src.models.cb.user_based import build_user_profile, explain_recommendation


# =========================================================
# Load artifacts
# =========================================================

BASE_DIR = os.path.dirname(__file__)

movies = pd.read_csv(
    os.path.join(BASE_DIR, "data/processed/movies_clean.csv")
)

ratings_small = pd.read_csv(
    os.path.join(BASE_DIR, "data/raw/ratings_small.csv")
)

with open(
    os.path.join(BASE_DIR, "src/models/cf/mf_model.pkl"), "rb"
) as f:
    mf_model = pickle.load(f)

tf_matrix = load_npz(
    os.path.join(BASE_DIR, "src/models/cb/tf_matrix.npz")
)

content_item_index = pd.Index(movies["movieId"])

movies["title_lower"] = movies["original_title"].str.lower()

movie_matrix_dense = tf_matrix.toarray()
movie_norms = np.linalg.norm(movie_matrix_dense, axis=1)


# =========================================================
# Helper functions
# =========================================================

def format_value(value):

    if isinstance(value, list):
        return ", ".join(map(str, value))

    if pd.isna(value):
        return ""

    return str(value)


def get_movie(movie_id):

    result = movies[movies["movieId"] == movie_id]

    if result.empty:
        return None

    return result.iloc[0]


def create_movie_card(movie, explanation=None):

    title = format_value(movie["original_title"])
    genres = ", ".join(ast.literal_eval(movie["genres"])) 
    director = format_value(movie["director"])
    cast = ", ".join(ast.literal_eval(movie["cast"]))
    poster_url = format_value(movie.get("poster_url", ""))

    if explanation is None:
        explanation = "Recommended based on similar movie content."

    card = f"""
    <div style="
        width: 220px;
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 12px;
        margin: 8px;
        display: inline-block;
        vertical-align: top;
        background: white;
    ">

        <img
            src="{poster_url}"
            style="
                width: 100%;
                height: 300px;
                object-fit: cover;
                border-radius: 8px;
            "
        >

        <h3 style="margin-bottom: 8px;">
            {title}
        </h3>

        <p>
            <b>Genres:</b> {genres}
        </p>

        <p>
            <b>Cast:</b> {cast}
        </p>

        <p>
            <b>Director:</b> {director}
        </p>

    </div>
    """

    return card


def create_results_html(movie_ids, user_id=None):

    cards = []

    for movie_id in movie_ids:

        movie = get_movie(movie_id)

        if movie is None:
            continue

        # explanation = None

        # if user_id is not None:

        #     try:
        #         explanation = explain_recommendation(
        #             user_id,
        #             movie,
        #             ratings_small,
        #             movies
        #         )

        #     except Exception:
        #         explanation = (
        #             "Recommended based on your previous movie ratings."
        #         )

        cards.append(
            create_movie_card(
                movie
                # explanation
            )
        )

    if not cards:
        return "<p>No recommendations found.</p>"

    return f"""
    <div style="
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
    ">
        {''.join(cards)}
    </div>
    """


# =========================================================
# Recommend by title
# =========================================================

def recommend_by_title(title):

    if not title or not title.strip():

        return "<p>Please enter a movie title.</p>"

    try:

        recommended_ids = recommend_movie(
            movies,
            title,
            tf_matrix
        )

    except Exception as e:

        return f"<p>Error: {e}</p>"

    return create_results_html(
        recommended_ids
    )


# =========================================================
# Recommend by user
# =========================================================

def recommend_by_user(user_id):

    if user_id is None:

        return "<p>Please enter a user ID.</p>"

    try:

        user_id = int(user_id)

    except Exception:

        return "<p>User ID must be a number.</p>"

    profile = build_user_profile(
        user_id,
        ratings_small,
        tf_matrix,
        content_item_index
    )

    recommended_ids = recommend_hybrid(
        user_id,
        movies,
        ratings_small,
        movie_norms,
        mf_model,
        tf_matrix,
        content_item_index,
        profile,
        weights={
            "mf": 0.7,
            "content": 0.3
        },
        n=10
    )

    return create_results_html(
        recommended_ids,
        user_id=user_id
    )


# =========================================================
# Gradio UI
# =========================================================

with gr.Blocks() as demo:

    gr.Markdown(
        """
        # 🎬 Movie Recommender System

        Get movie recommendations using
        **Content-Based Filtering**, **Collaborative Filtering**
        and a **Hybrid Recommender**.
        """
    )

    # =====================================================
    # By Title
    # =====================================================

    with gr.Tab("By Title"):

        gr.Markdown(
            "Enter a movie title to find movies with similar content."
        )

        inp_title = gr.Textbox(
            label="Movie Title",
            placeholder="e.g. Batman"
        )

        btn_title = gr.Button("Recommend")

        out_title = gr.HTML()

        btn_title.click(
            fn=recommend_by_title,
            inputs=inp_title,
            outputs=out_title
        )

    # =====================================================
    # By User ID
    # =====================================================

    with gr.Tab("By User ID"):

        gr.Markdown(
            "Enter a user ID to receive personalized recommendations."
        )

        inp_uid = gr.Number(
            label="User ID",
            precision=0
        )

        btn_user = gr.Button("Recommend")

        out_user = gr.HTML()

        btn_user.click(
            fn=recommend_by_user,
            inputs=inp_uid,
            outputs=out_user
        )


demo.launch()