import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import ast
import streamlit as st
import requests
import difflib

import os

# Load the datasets
# Using relative path
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# movies = pd.read_csv(r"C:/Users/DELL/Downloads/recommendation_system_data/tmdb_5000_movies.csv")
# credits = pd.read_csv(r"C:/Users/DELL/Downloads/recommendation_system_data/ tmdb_5000_credits.csv")
# Merge the datasets on 'id'
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres',
                 'keywords', 'cast', 'crew', 'budget',
                 'original_language', 'popularity',
                 'production_companies',
                 'production_countries', 'release_date',
                 'revenue', 'runtime', 'spoken_languages',
                 'status', 'vote_average', 'tagline',
                 'vote_count']]

# Check for null values
print(movies.isnull().sum())

print(movies.columns)

# Drop rows with null values
movies.dropna(inplace=True)

# Verify no null values remain
print(movies.isnull().sum())


# Preprocess textual data
def convert_to_list(x):
    try:
        return [i['name'] for i in ast.literal_eval(x)]
    except:
        return []


def clean_data(x):
    if isinstance(x, str):
        return x.lower().replace(" ", "")
    elif isinstance(x, list):
        return [i.replace(" ", "").lower() for i in x]
    else:
        return ''


# Apply cleaning and conversion to relevant columns
movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].apply(convert_to_list).apply(clean_data)
movies['keywords'] = movies['keywords'].apply(convert_to_list).apply(clean_data)
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:3]).apply(clean_data)
movies['crew'] = movies['crew'].apply(
    lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director']).apply(clean_data)
movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
movies['production_companies'] = movies['production_companies'].apply(convert_to_list).apply(clean_data)
movies['production_countries'] = movies['production_countries'].apply(convert_to_list).apply(clean_data)
movies['spoken_languages'] = movies['spoken_languages'].apply(convert_to_list).apply(clean_data)
movies['tagline'] = movies['tagline'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Combine textual features into a single string
movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'] + movies['overview'] + movies[
    'production_companies'] + movies['production_countries'] + movies['spoken_languages'] + movies['tagline']

# Tokenize the 'tags' column
movies['tags'] = movies['tags'].apply(lambda x: ' '.join(x))

# Normalize numerical features
scaler = MinMaxScaler()
movies['budget'] = scaler.fit_transform(movies[['budget']])
movies['popularity'] = scaler.fit_transform(movies[['popularity']])
movies['revenue'] = scaler.fit_transform(movies[['revenue']])
movies['runtime'] = scaler.fit_transform(movies[['runtime']])
movies['vote_average'] = scaler.fit_transform(movies[['vote_average']])
movies['vote_count'] = scaler.fit_transform(movies[['vote_count']])

# Combine textual and numerical features
movies['combined_features'] = movies['tags'] + ' ' + \
                              movies['budget'].astype(str) + ' ' + \
                              movies['popularity'].astype(str) + ' ' + \
                              movies['revenue'].astype(str) + ' ' + \
                              movies['runtime'].astype(str) + ' ' + \
                              movies['vote_average'].astype(str) + ' ' + \
                              movies['vote_count'].astype(str)

# Vectorize the combined features
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['combined_features']).toarray()

# Compute cosine similarity
similarity = cosine_similarity(vector)


def fetch_movie_details(movie_id):
    api_key = 'c796b6b547bc29be54231d5e5ba61bf2'
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    response = requests.get(url)
    data = response.json()
    return {
        "poster_path": data.get('poster_path'),
        "title": data.get('title'),
        "overview": data.get('overview'),
        "genre": data.get('genre'),
        "release_date": data.get('release_date'),
        "runtime": data.get('runtime'),
        "vote_average": data.get('vote_average'),
        "budget": data.get('budget'),
        "popularity": data.get('popularity'),
        "Revenue": data.get('revenue'),
        "cast": data.get('cast'),
        "crew": data.get('crew'),
        "keywords": data.get('keywords'),

    }


def get_poster_url(poster_path):
    return f"https://image.tmdb.org/t/p/w500{poster_path}"


# Function to recommend movies
def recommend(movie_title, num_recommendations):
    movie_index = movies[movies['title'] == movie_title].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recommendations]

    recommended_movies = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        details = fetch_movie_details(movie_id)
        details['cast'] = movies.iloc[i[0]]['cast']
        details['crew'] = movies.iloc[i[0]]['crew']
        recommended_movies.append(details)
    return recommended_movies


def set_background_image(color=None):
    st.markdown(f"""
            <style>
            .stApp {{
                background-color: {color};
            }}
            </style>
            """, unsafe_allow_html=True)


def main():
    st.markdown("<h1 style='color: #FFD700;'>Enhanced Movie Recommender System</h1>", unsafe_allow_html=True)

    set_background_image(color="black")

    # List of movie titles from your dataset
    movie_titles = movies['title'].tolist()

    st.markdown("<p style='color: #FFD700;'>Enter movie name</p>", unsafe_allow_html=True)

    # Accept user input
    user_input = st.text_input("")

    # Slider to choose the number of recommendations
    st.markdown("<p style='color: #FFD700;'>Number of movie to recommend</p>", unsafe_allow_html=True)
    num_recommendations = st.slider("", 1, 10, 5)

    # Recommend movies when the button is pressed
    if st.button("Recommend"):
        # Use fuzzy matching to find the closest movie title
        closest_match = difflib.get_close_matches(user_input, movie_titles, n=1, cutoff=0.6)

        if closest_match:
            selected_movie = closest_match[0]  # Closest matching movie title
            st.markdown(f"<span style='color: #FFD700;'>Did you mean: {selected_movie}?</span>",
                        unsafe_allow_html=True)

            # Call the recommend function with the selected movie
            recommendations = recommend(selected_movie, num_recommendations)
            st.markdown(f"<span style='color: #FFD700;'>Top {num_recommendations} Recommended Movies:</span>",
                        unsafe_allow_html=True)

            # Display movies in the left column, and their details in the right column
            for movie in recommendations:
                st.markdown(f"<h3 style='color: #FFD700;'>{movie['title']}</h3>", unsafe_allow_html=True)  # Gold color
                st.image(get_poster_url(movie['poster_path']), width=120)
                # Display the movie title first, in bold and larger text
                # Then display the other details
                st.markdown(f"<span style='color: #FFFFFF;'>**Overview:** {movie['overview']}</span>",
                            unsafe_allow_html=True)
                st.markdown(f"<span style='color: #FFFFFF;'>**Release Date:** {movie['release_date']}</span>",
                            unsafe_allow_html=True)
                st.markdown(f"<span style='color: #FFFFFF;'>**Runtime:** {movie['runtime']} minutes</span>",
                            unsafe_allow_html=True)
                st.markdown(f"<span style='color: #FFFFFF;'>**Vote Average:** {movie['vote_average']}/10</span>",
                            unsafe_allow_html=True)
                st.markdown(f"<span style='color: #FFFFFF;'>**Popularity:** {movie['popularity']}</span>",
                            unsafe_allow_html=True)
                st.write("---")
        else:
            st.markdown(f"<span style='color: red;'>Sorry, no matching movie found.</span>",
                        unsafe_allow_html=True)


# Entry point for Streamlit app
if __name__ == "__main__":
    main()
