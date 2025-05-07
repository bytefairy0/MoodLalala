import pandas as pd
import ast
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

def main():
    # Apply custom CSS for styling (keeping existing CSS as it's already in the code)
    add_custom_css()

    st.title("✨ MoodLalala: Your Mood-Based Recommender 🎬📚")
    print("🎯 MAIN STARTED")
    st.write("✅ App loading... please wait")
    try:
        # Read the files in Pandas
        books_df = pd.read_csv("books_cleaned.csv")
        movies_df = pd.read_csv("movies_metadata_reduced.csv", low_memory=False)

        # Cleaning and Preparing the Data
        # Limiting to 5000 rows
        movies_df = movies_df.head(5000)
        movies_df = movies_df[['title', 'genres', 'runtime', 'overview', 'vote_average']]

        # Renaming columns
        movies_df = movies_df.rename(columns={
            'genres' : 'genre',
            'overview' : 'description',
            'runtime': 'duration',
            'vote_average': 'rating'
        })

        movies_df['genre'] = movies_df['genre'].apply(extract_genre_names)

        # Dropping Rows with Missing Info
        movies_df = movies_df.dropna(subset=['title', 'genre', 'description', 'duration'])
        movies_df['description'] = movies_df['description'].fillna('')
        movies_df['rating'] = pd.to_numeric(movies_df['rating'], errors='coerce')
        movies_df['duration'] = pd.to_numeric(movies_df['duration'], errors='coerce')

        # Only drop rows with missing duration, keep rows with missing ratings
        movies_df = movies_df.dropna(subset=['duration'])
        
        # Fill NaN ratings with "Not rated" text
        movies_df['rating'] = movies_df['rating'].fillna("Not rated")

        # Resetting Index
        movies_df.reset_index(drop=True, inplace=True)

        books_df = books_df.rename(columns={'Name': 'title', 'Genre':'genre'}) # renaming a column
        # Clean the book titles
        books_df['title'] = books_df['title'].apply(lambda x:x.split('(')[0].strip())

        # Getting Book Info from Google Books API
        # Note: API key is fetched from environment variables
        # api_key = os.getenv("GOOGLE_BOOKS_API_KEY", "")
        # books_df[['description', 'pages', 'rating']] = books_df['title'].apply(
        #     lambda x: pd.Series(fetchBookInfo(x, api_key)))

        books_df['description'] = books_df['description'].fillna('')

        # Creating duration columns in Books (converting page number into time)
        books_df['pages'] = pd.to_numeric(books_df['pages'], errors='coerce')
        books_df = books_df.dropna(subset=['pages'])
        books_df['duration'] = books_df['pages'] * 2 # Estimated reading time

        # Streamlit Code for Interface
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("Find the perfect book or movie based on your current mood!")
            
        # Mood Input
        user_mood = st.text_input("💭 How are you feeling today?")

        # Genre Input
        genre_list = pd.concat([
            books_df['genre'], 
            movies_df['genre']
        ]).dropna().astype(str).str.strip().unique().tolist()
        user_genre = st.selectbox("🎭 Choose a Genre", options=genre_list)

        # Time Input
        user_time = st.slider("⏱️ Max time you can spend (in minutes):", 10, 300, 60)

        # Get Recommendations Button
        if st.button("🎁 Get Recommendations!"):
            if not user_mood:
                st.warning("Please enter your mood to get recommendations!")
                return
                
            with st.spinner('Fetching data... please wait'):
                books = recommend_books_by_mood(user_mood, books_df, top_n=3)
                movies = recommend_movies_by_mood(user_mood, movies_df, top_n=3)

                if 'duration' not in books.columns and 'pages' in books.columns:
                    books['duration'] = books['pages'] * 2

                filtered_books = filter_results(books_df.head(300), user_genre, user_time)
                filtered_movies = filter_results(movies_df.head(300), user_genre, user_time)

                # 🎯 Mood-based recommendations in expandable sections - NOT expanded by default
                with st.expander("📚 Books Based on Mood", expanded=False):
                    display_recommendations_as_cards(books, "book", "mood")

                with st.expander("🎬 Movies Based on Mood", expanded=False):
                    display_recommendations_as_cards(movies, "movie", "mood")

                # 📚 Genre and Time-Based Recommendations in expandable sections - NOT expanded by default
                with st.expander("📚 Genre + Time Based Book Picks", expanded=False):
                    display_recommendations_as_cards(filtered_books, "book", "genre")

                with st.expander("🎥 Genre + Time Based Movie Picks", expanded=False):
                    display_recommendations_as_cards(filtered_movies, "movie", "genre")

                # Display a random recommendation for bonus in an expandable section - NOT expanded by default
                with st.expander("🎁 Bonus Recommendations", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    if not books.empty:
                        random_book = books.sample(1).iloc[0]
                        with col1:
                            display_bonus_card(random_book, "book")

                    if not movies.empty:
                        random_movie = movies.sample(1).iloc[0]
                        with col2:
                            display_bonus_card(random_movie, "movie")

    except Exception as e:
        st.error(f"❌ Something broke: {e}")


def extract_genre_names(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return ', '.join([g['name'] for g in genres if 'name' in g])
    except:
        return 'Unknown'

def fetchBookInfo(title, api_key):
    # Getting info from Google Books API (not scraping kez goodreads block scra)
    try:
        url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{title}&key={api_key}"
        response = requests.get(url)
        data = response.json()

        if 'items' in data:
            book = data['items'][0]['volumeInfo'] # get first book in results list and focus on volumeInfo section
            description = book.get('description', 'No description available')
            pages = book.get('pageCount', 'Unknown')
            rating = book.get('averageRating', 'Not rated')

            return description, pages, rating
        else:
            return 'No description', 'Unknown', 'Not rated'

    except Exception as e:
        print(f"Error fetching for {title}: {e}")
        return 'Error fetching', 'Error', 'Error' # Error response in case of something breaks
  
def recommend_books_by_mood(user_input, df, top_n=7):
    tfidf = TfidfVectorizer(stop_words='english')

    # Fit on all descriptions
    tfidf_matrix = tfidf.fit_transform(df['description'])

    # Transform user input into same vector space
    user_vec = tfidf.transform([user_input])

    # Calculate similarity scores
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()

    # Attach scores to DataFrame
    result_df = df.copy()
    result_df['similarity'] = similarity_scores

    # Return top matches
    return result_df.sort_values(by='similarity', ascending=False)[['title', 'genre', 'description', 'pages', 'rating']].head(top_n)

def recommend_movies_by_mood(user_input, df, top_n=7):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    user_vec = tfidf.transform([user_input])
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    result_df = df.copy()
    result_df['similarity'] = similarity_scores
    return result_df.sort_values(by='similarity', ascending=False)[['title', 'genre', 'description', 'duration', 'rating']].head(top_n)

def filter_results(df, selected_genre=None, max_time=None):
    # Create a copy to avoid modifying original
    filtered_df = df.copy()
    
    if 'genre' not in filtered_df.columns or 'duration' not in filtered_df.columns:
        return pd.DataFrame()  # return empty safely if needed

    if selected_genre:
        # Convert both to lowercase for case-insensitive comparison
        filtered_df = filtered_df[filtered_df['genre'].str.lower().str.contains(selected_genre.lower(), na=False)]
    if max_time:
        filtered_df = filtered_df[filtered_df['duration'] <= max_time]
    return filtered_df.head(3)  # Reduced from 5 to 3


def enhanced_layout(title, dataframe, icon=None):
    """
    Displays a prettier section with optional icon + bold title.
    """
    st.markdown("---")
    title_str = f"### {icon} {title}" if icon else f"### {title}"
    st.markdown(title_str)
    st.dataframe(dataframe)


def add_custom_css():
    st.markdown(
        """
        <style>
        /* Custom font and background */
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #1e1e1e;
            color: #f0f0f0;
        }

        /* Styling for headers and subheaders */
        .streamlit-expanderHeader {
            color: #4CAF50;
            font-weight: bold;
            font-size: 24px;
        }

        h1, h2, h3, h4 {
            color: #f0f0f0;
            font-weight: 500;
        }

        /* Styling buttons */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px 30px;
            border: none;
            cursor: pointer;
            transition: 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #45a049;
        }

        /* Add a simple spinner animation */
        .stSpinner>div {
            background-color: #0d6efd;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s infinite linear;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Styling for recommendation cards */
        .recommendation-card {
            background-color: #333;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #4CAF50;
        }

        .book-card {
            border-left: 4px solid #4CAF50;
        }

        .movie-card {
            border-left: 4px solid #9c27b0;
        }

        .card-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 8px;
            color: #fff;
        }

        .card-genre {
            font-size: 14px;
            color: #aaa;
            margin-bottom: 5px;
        }

        .card-rating {
            font-size: 14px;
            color: #ffc107;
            margin-bottom: 5px;
        }

        .card-details {
            font-size: 14px;
            color: #ddd;
            margin-top: 8px;
        }

        .expander-content {
            padding: 10px;
            background-color: #2a2a2a;
            border-radius: 5px;
            margin-top: 5px;
        }

        .bonus-card {
            background-color: #2c3e50;
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #e74c3c;
            margin-bottom: 15px;
        }

        /* Adjust dataframe styling */
        .stDataFrame {
            font-size: 14px;
            border-collapse: collapse;
            width: 100%;
        }
        .stDataFrame th, .stDataFrame td {
            padding: 8px 15px;
            border: 1px solid #444;
        }
        .stDataFrame th {
            background-color: #4CAF50;
            color: white;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


def display_recommendations_as_cards(df, item_type, section_type):
    """
    Display recommendations as cards instead of a dataframe
    
    Parameters:
    df (DataFrame): DataFrame containing recommendations
    item_type (str): 'book' or 'movie' to determine styling and what fields to show
    section_type (str): The section where this card appears (for unique keys)
    """
    if df.empty:
        st.warning(f"No {item_type}s found matching your criteria 😢")
        return
    
    # Limit to fewer recommendations (3 instead of 5)
    df = df.head(3)
    
    # Counter to make keys unique even for same titles
    item_count = 0
    
    for _, item in df.iterrows():
        # Create a container for each item
        st.markdown("---")
        
        # Format the title and rating for display
        title = item['title']
        rating = item.get('rating', 'Not rated')
        if isinstance(rating, (int, float)):
            rating_display = f"{rating:.1f} ⭐"
        else:
            rating_display = f"{rating} ⭐"
            
        # Display title and rating
        st.markdown(f"### {'📚' if item_type == 'book' else '🎬'} **{title}** ({rating_display})")
        
        # Create columns for content
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display item details
            st.markdown(f"**Genre:** {item.get('genre', 'Unknown')}")
            
            # Show appropriate duration info based on type
            if item_type == "book":
                pages = item.get('pages', 'Unknown')
                duration = item.get('duration', 'Unknown')
                st.markdown(f"**Pages:** {pages}")
                st.markdown(f"**Est. Reading Time:** {duration} minutes")
            else:
                duration = item.get('duration', 'Unknown')
                st.markdown(f"**Duration:** {duration} minutes")
        
        with col2:
            # Show description
            st.markdown("**Description:**")
            description = item.get('description', 'No description available')
            if description:
                # Show a truncated description to keep the card compact
                if len(description) > 200:
                    st.markdown(description[:200] + "...")
                    
                    # Create a unique key for this button by combining all information
                    unique_key = f"{section_type}_{item_type}_{title}_{item_count}"
                    
                    # Add a details button with unique key
                    if st.button(f"See full description for {title}", key=unique_key):
                        st.markdown(description)
                else:
                    st.markdown(description)
            else:
                st.markdown("No description available")
        
        # Increment counter for uniqueness
        item_count += 1


def display_bonus_card(item, item_type):
    """
    Display a bonus recommendation card
    
    Parameters:
    item (Series): Pandas Series containing item details
    item_type (str): 'book' or 'movie' to determine styling and fields
    """
    # Create content container with bonus styling
    st.markdown(
        f"""
        <div class="recommendation-card bonus-card">
            <div class="card-title">🎁 Bonus {item_type.capitalize()} Pick: {item['title']}</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Display item details
    st.markdown(f"**Genre:** {item.get('genre', 'Unknown')}")
    
    # Rating with star emoji
    rating = item.get('rating', 'Not rated')
    if isinstance(rating, (int, float)):
        st.markdown(f"**Rating:** {rating:.1f} ⭐")
    else:
        st.markdown(f"**Rating:** {rating} ⭐")
    
    # Show appropriate info based on type
    if item_type == "book":
        pages = item.get('pages', 'Unknown')
        duration = item.get('duration', 'Unknown')
        st.markdown(f"**Pages:** {pages}")
        st.markdown(f"**Est. Reading Time:** {duration} minutes")
    else:
        duration = item.get('duration', 'Unknown')
        st.markdown(f"**Duration:** {duration} minutes")
    
    # Show a preview of the description
    description = item.get('description', 'No description available')
    if description and len(description) > 150:
        preview = description[:150] + "..."
    else:
        preview = description
        
    st.markdown(f"**Preview:** {preview}")


if __name__ == "__main__":
    main()
