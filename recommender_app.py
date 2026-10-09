import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Page configuration
st.set_page_config(page_title="Movie Recommender System", page_icon="🍿", layout="wide")

# Sample movie catalogue
movie_data = pd.DataFrame({
    "Movie ID": range(1, 11),
    "Title": ["Inception", "Titanic", "Avatar", "The Matrix", "Interstellar",
              "The Notebook", "John Wick", "La La Land", "Mad Max: Fury Road", "Pride & Prejudice"],
    "Genre": ["Sci-Fi", "Romance", "Sci-Fi", "Action", "Sci-Fi",
              "Romance", "Action", "Romance", "Action", "Romance"],
})
movie_titles = movie_data["Title"].tolist()

# Sample user-item ratings matrix (NaN = not rated). Users have recognisable tastes.
ratings_matrix = np.array([
    #  Inc  Tit  Ava  Mat  Int  Note Wick LaLa  Max  P&P
    [5.0, 3.0, np.nan, 4.5, 5.0, np.nan, 3.5, np.nan, 4.0, 2.0],  # User 1: sci-fi fan
    [4.0, np.nan, 4.0, 5.0, np.nan, 1.5, 5.0, 2.0, 4.5, np.nan],  # User 2: action and sci-fi
    [1.0, 5.0, np.nan, 1.5, 2.0, 5.0, np.nan, 4.5, 1.0, 5.0],  # User 3: romance fan
    [np.nan, 4.5, 3.5, 2.0, np.nan, 4.0, 1.5, 5.0, np.nan, 4.5],  # User 4: mostly romance
    [4.5, 2.0, 5.0, np.nan, 4.5, np.nan, 3.0, 2.5, 4.0, 1.5],  # User 5: sci-fi blockbusters
    [3.0, np.nan, 2.5, 4.5, 3.0, 2.0, 5.0, np.nan, 5.0, np.nan],  # User 6: action fan
])
user_labels = [f"User {i + 1}" for i in range(len(ratings_matrix))]

DEFAULT_RATINGS = {"Inception": 5.0, "The Matrix": 4.5, "Titanic": 1.5}


def centered_cosine(ratings, shrink=2.0):
    """Cosine similarity on mean-centred ratings (unrated items count as neutral).

    Each user's mean is shrunk towards the global mean, so someone who has only given a couple
    of 5-star ratings still reads as "likes these" instead of "average".
    """
    counts = (~np.isnan(ratings)).sum(axis=1, keepdims=True)
    sums = np.nansum(ratings, axis=1, keepdims=True)
    means = (sums + shrink * np.nanmean(ratings)) / (counts + shrink)
    centred = np.nan_to_num(ratings - means)
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = centred / norms
    return unit @ unit.T, means.ravel()


def predict_for_user(ratings, sim, means, u, k=3):
    """Predict user u's unrated items from the k most similar users who rated each item."""
    preds, reasons = {}, {}
    for i in range(ratings.shape[1]):
        if not np.isnan(ratings[u, i]):
            continue
        raters = [v for v in range(ratings.shape[0]) if v != u and not np.isnan(ratings[v, i]) and sim[u, v] > 0]
        raters = sorted(raters, key=lambda v: sim[u, v], reverse=True)[:k]
        if not raters:
            preds[i] = float(np.nanmean(ratings[:, i]))
            reasons[i] = "No similar user rated it, so this is its average rating."
            continue
        w = np.array([sim[u, v] for v in raters])
        dev = np.array([ratings[v, i] - means[v] for v in raters])
        preds[i] = float(np.clip(means[u] + (w @ dev) / w.sum(), 0.5, 5.0))
        top = raters[0]
        reasons[i] = f"{user_labels[top]} ({sim[u, top]:.0%} similar to you) rated it {ratings[top, i]:.1f}."
    return preds, reasons


# App title
st.title("🍿 Netflix-Style Movie Recommender System")
st.markdown(
    "Rate a few movies in the sidebar and the app recommends what to watch next using "
    "**user-based collaborative filtering**: it finds viewers whose taste matches yours and "
    "borrows their opinions on the movies you have not seen. A few ratings are pre-filled so you "
    "see results straight away."
)

with st.expander("How it works"):
    st.markdown(
        """
1. **Ratings matrix:** each row is a user, each column a movie, blanks are unrated.
2. **Similarity:** every user's ratings are centred on their own average (so a harsh and a generous
   rater can still agree, and that average is pulled towards the global mean when a user has rated
   only a few movies), then compared with **cosine similarity**.
3. **Prediction:** for each movie you have not rated, take up to 3 of the most similar users who
   rated it and add their weighted above/below-average opinion to your own average rating.
4. **Ranking:** unrated movies are sorted by predicted rating.

This is the classic neighbourhood approach behind early Netflix and Amazon recommenders. It runs on
a small built-in sample so it works without any downloads.
"""
    )

# Sidebar for user input
st.sidebar.header("Rate movies")
st.sidebar.write("Leave a slider at **0.0** if you haven't watched that movie.")
for movie in movie_titles:
    st.session_state.setdefault(f"rate_{movie}", DEFAULT_RATINGS.get(movie, 0.0))
if st.sidebar.button("Clear all ratings"):
    for movie in movie_titles:
        st.session_state[f"rate_{movie}"] = 0.0

new_user_ratings = []
for i, movie in enumerate(movie_titles):
    rating = st.sidebar.slider(
        f"{movie} ({movie_data['Genre'][i]})", min_value=0.0, max_value=5.0, step=0.5,
        format="%.1f", key=f"rate_{movie}",
    )
    new_user_ratings.append(rating)
new_user = np.array(new_user_ratings)

if np.all(new_user == 0.0):
    st.warning("Please rate at least one movie to get personalised recommendations.")
    st.subheader("Available movies")
    st.dataframe(movie_data, width="stretch", hide_index=True)
    st.stop()

new_user_with_nan = np.where(new_user == 0.0, np.nan, new_user)
ratings_with_new_user = np.vstack([ratings_matrix, new_user_with_nan])
all_labels = user_labels + ["You"]
similarity, means = centered_cosine(ratings_with_new_user)
you = len(ratings_with_new_user) - 1
preds, reasons = predict_for_user(ratings_with_new_user, similarity, means, you)

left, right = st.columns([3, 2])
with left:
    st.subheader("🎯 Personalised recommendations")
    if not preds:
        st.info("You have already rated every movie in the catalogue.")
    else:
        rec = pd.DataFrame({
            "Title": [movie_titles[i] for i in preds],
            "Genre": [movie_data["Genre"][i] for i in preds],
            "Predicted rating": [round(p, 2) for p in preds.values()],
            "Why": [reasons[i] for i in preds],
        }).sort_values("Predicted rating", ascending=False)
        top3 = rec.head(3).reset_index(drop=True)
        cols = st.columns(len(top3))
        for col, (_, row) in zip(cols, top3.iterrows()):
            col.metric(f"{row['Title']}", f"{row['Predicted rating']:.1f} ★", row["Genre"], delta_color="off")
        st.dataframe(rec, width="stretch", hide_index=True)

with right:
    st.subheader("🧑‍🤝‍🧑 Your closest matches")
    sims = pd.DataFrame({"User": user_labels, "Similarity": similarity[you, :-1]}).sort_values("Similarity")
    fig_s = px.bar(sims, x="Similarity", y="User", orientation="h", color="Similarity",
                   color_continuous_scale="RdBu", range_color=[-1, 1])
    fig_s.update_layout(height=320, margin=dict(t=10, b=10), coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig_s, width="stretch")
    best = sims.iloc[-1]
    st.caption(f"Your taste is closest to **{best['User']}** (similarity {best['Similarity']:.2f}, range −1 to 1).")

tab_heat, tab_matrix = st.tabs(["🔍 User similarity heatmap", "📋 Ratings matrix"])
with tab_heat:
    fig_h = px.imshow(np.round(similarity, 2), x=all_labels, y=all_labels, text_auto=True,
                      color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
    fig_h.update_layout(height=480)
    st.plotly_chart(fig_h, width="stretch")
with tab_matrix:
    matrix = pd.DataFrame(ratings_with_new_user, index=all_labels, columns=movie_titles)
    st.dataframe(matrix.style.format("{:.1f}", na_rep="–"), width="stretch")

# Footer
st.markdown("---")
st.caption(
    "**About:** collaborative filtering helps businesses personalise catalogues, from streaming to retail. "
    "Built with Streamlit, NumPy and Plotly."
)
