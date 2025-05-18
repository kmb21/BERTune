import pandas as pd
import pickle
import numpy as np
import sys
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix
from setup import clean_lyrics
from lyricsgenius import Genius

token = "6HdId8kO6n4quYWrSFJt6VukjY5SDeuWGch-F6UMnu7630izwDRU6t6i1TysHcty"

def get_lyrics(artist_name, title):
    try:
        genius = Genius(token)
        song = genius.search_song(title, artist_name)
        if song is None:
            artist = genius.search_artist(artist_name, max_songs=3, sort='title')
            if artist:
                song = genius.search_song(title, artist.name)
        if song:
            return song.lyrics
        return None
    except Exception as e:
        print(f"Error fetching lyrics: {e}")
        return None


def open_pickle_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def build_lyrics_matrix(english_df, words_dict, titles_dict):
    """Build term-frequency matrix in COO format"""
    data = [] 
    rows = []
    cols = []
    
    for _, row in english_df.iterrows():
        title = row['title']
        lyrics = row['cleaned_lyrics']
        row_num = titles_dict[title]
        
        word_counts = Counter(lyrics.split())
        for word, count in word_counts.items():
            if word in words_dict:
                data.append(count)
                rows.append(row_num)
                cols.append(words_dict[word])
    
    return coo_matrix((data, (rows, cols)), 
                     shape=(len(titles_dict), len(words_dict)), 
                     dtype=np.int32).tocsr()



def main():
    words_dict = open_pickle_file('/scratch/dburger1/proj/words_dict.pkl')
    titles_dict = open_pickle_file('/scratch/dburger1/proj/titles_dict.pkl')
    
    english_df = pd.read_pickle("/scratch/dburger1/proj/english_songs.pkl")

    index_to_title = {v: k for k, v in titles_dict.items()}


    matrix = build_lyrics_matrix(english_df, words_dict, titles_dict)
    

    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(matrix)
    tfidf_matrix = tfidf_transformer.transform(matrix)

    print("Type 0 to exit")
    while True:
        name = input('What artist are you listening to? ').strip().lower().title()
        song_title = input('What song are you listening to? ').strip().lower().title()

        if name == '0' or song_title == '0':
            sys.exit(0)
            
        lyrics = get_lyrics(name, song_title)
        if lyrics:
            break
        print(f"Couldn't find lyrics for {song_title} by {name}. Please try again.")
    print(lyrics)
    return
    query_str = lyrics
    print("Lyrics found, processing...")
#     query_str = """
# What would I do without your smart mouth?
# Drawing me in, and you kicking me out
# You've got my head spinning, no kidding, I can't pin you down
# What's going on in that beautiful mind?
# I'm on your magical mystery ride
# And I'm so dizzy, don't know what hit me, but I'll be alright
# My head's under water
# But I'm breathing fine
# You're crazy and I'm out of my mind
# 'Cause all of me
# Loves all of you
# Love your curves and all your edges
# All your perfect imperfections
# Give your all to me
# I'll give my all to you
# You're my end and my beginning
# Even when I lose, I'm winning
# 'Cause I give you all of me
# And you give me all of you, oh-oh
# How many times do I have to tell you?
# Even when you're crying, you're beautiful too
# The world is beating you down, I'm around through every mood
# You're my downfall, you're my muse
# My worst distraction, my rhythm and blues
# I can't stop singing, it's ringing in my head for you
# My head's under water
# But I'm breathing fine
# You're crazy and I'm out of my mind
# 'Cause all of me
# Loves all of you
# Love your curves and all your edges
# All your perfect imperfections
# Give your all to me
# I'll give my all to you
# You're my end and my beginning
# Even when I lose, I'm winning
# 'Cause I give you all of me
# And you give me all of you, oh-oh
# Give me all of you, oh
# Cards on the table, we're both showing hearts
# Risking it all, though it's hard
# 'Cause all of me
# Loves all of you
# Love your curves and all your edges
# All your perfect imperfections
# Give your all to me
# I'll give my all to you
# You're my end and my beginning
# Even when I lose, I'm winning
# 'Cause I give you all of me
# And you give me all of you
# I give you all of me
# And you give me all of you, oh-oh
# """

    query_terms = clean_lyrics(query_str).split()
    print(query_terms)
    num_rows, num_cols = matrix.shape
    query_vec = np.zeros((1, num_cols), dtype=np.int32)

    query_counts = Counter(query_terms)

    unknown_words = []
    for word, count in query_counts.items():
        if word in words_dict:
            query_vec[0, words_dict[word]] = count
        else:
            unknown_words.append(word)

    if unknown_words:
        print(f"Unknown words: {', '.join(unknown_words)}")

    tfidf_query = tfidf_transformer.transform(query_vec)


    similarities = cosine_similarity(tfidf_query, tfidf_matrix).flatten()
    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    print(f"\nTop 5 songs similar to the query:")
    for i in range(min(5, len(ranked))):
        row_index, cos_sim = ranked[i]
        title = index_to_title.get(row_index, "Unknown")
        print(f"{i+1}. {title} (score: {cos_sim:.3f})")

if __name__ == "__main__":
    main()