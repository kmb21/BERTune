import os
os.system('clear')
print("importing libraries, please wait...")
import io
import contextlib
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
from lyricsgenius import Genius
from setup import clean_lyrics
from query import open_pickle_file, get_lyrics
from pathlib import Path
from sklearn.cluster import KMeans
import numpy as np
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from rich.console import Console
from rich.table import Table
from rich import box
print("")

token = "6HdId8kO6n4quYWrSFJt6VukjY5SDeuWGch-F6UMnu7630izwDRU6t6i1TysHcty"
genius = Genius(token)

def display_playlist(title, matches, index_to_title, artist_lookup_fn, N, artist, method):
    
    table = Table(
        title=f"🎶 A playlist of {N} songs based on {title} by {artist}! 🎶 \n ~ Using {method} ~",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_lines=True,
        title_style="bold magenta",
        header_style="bold cyan"
    )

    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Song Title", style="italic")
    table.add_column("Artist", style="green")

    for rank, idx in enumerate(matches, start=1):
        song_title = index_to_title.get(idx, "Error: Unknown")
        artist = artist_lookup_fn(song_title)
        table.add_row(str(rank), song_title, artist)

    console = Console()
    console.print(table)

def get_lyrics_by_indices(indices, df):
    """
    Given a list of indices and a DataFrame, return a list of lyrics for those indices.
    """
    lyrics_list = []
    for idx in indices:
        try:
            lyrics = df.iloc[idx]['cleaned_lyrics']
            lyrics_list.append(lyrics)
        except IndexError:
            lyrics_list.append("[Lyrics not found]")
    return lyrics_list

def find_artist_by_title(title):
    """
    Use Genius API to find the primary artist for a given song title.
    """
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            song = genius.search_song(title)
            if song and song.artist:
                return song.artist
    except Exception as e:
        print(f"Error fetching artist for '{title}': {e}")
    return "Error: Unknown"



def find_closest_songs_global(query_vec, embedding_matrix, N, query_title, index_to_title):
    """
    Given a query vector, find the closest songs globally using cosine similarity.
    """
    sims = cosine_similarity(query_vec.reshape(1, -1), embedding_matrix).flatten()
    ranked_indices = np.argsort(sims)[::-1]

    filtered_indices = []
    query_title_clean = query_title.strip().lower()
    for idx in ranked_indices:
        song_title = str(index_to_title.get(idx, "")).strip().lower()
        if song_title != query_title_clean:
            filtered_indices.append(idx)    
    return filtered_indices[:N]


def find_closest_songs_in_cluster(query_vec, embedding_matrix, cluster_labels, model, N, method, query_title, index_to_title):
    """
    Given a query vector, find the closest songs in the same Kmeans or GMM cluster using cosine similarity.
    """
    if method == "kmeans":
        cluster_id = model.predict(query_vec.reshape(1, -1))[0]
    elif method == "gmm":
        cluster_id = model.predict(query_vec.reshape(1, -1))[0]
    else:
        raise ValueError("method must be 'kmeans' or 'gmm'")

    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    cluster_vectors = embedding_matrix[cluster_indices]

    sims = cosine_similarity(query_vec.reshape(1, -1), cluster_vectors).flatten()
    ranked_cluster_indices = cluster_indices[np.argsort(sims)[::-1]]

    filtered_indices = []
    query_title_clean = query_title.strip().lower()

    for idx in ranked_cluster_indices:
        song_title = str(index_to_title.get(idx, "")).strip().lower()
        if song_title != query_title_clean:
            filtered_indices.append(idx)   
            
             
    return filtered_indices[:N]


def embed_song_from_lyrics(lyrics_str, model):
    """
    Given a string of lyrics, compute a BERT embedding by averaging line embeddings.
    """
    lines = [line.strip() for line in lyrics_str.strip().split("\n") if line.strip()]

    if not lines:
        raise ValueError("No valid lines found in lyrics.")

    line_embeddings = model.encode(lines)
    song_embedding = line_embeddings.mean(axis=0)

    return song_embedding

def main():
    print("Loading data, please wait...")
    words_dict = open_pickle_file('/scratch/dburger1/proj/words_dict.pkl')
    titles_dict = open_pickle_file('/scratch/dburger1/proj/titles_dict.pkl')
    index_to_title = open_pickle_file('/scratch/dburger1/proj/index_to_title.pkl')
    english_df = pd.read_pickle("/scratch/dburger1/proj/english_songs.pkl")
    gmm_labels = np.load('saved_cluster/gmm_labels.npy')
    gmm_model = open_pickle_file('saved_cluster/gmm_model.pkl')
    kmeans_labels = np.load('saved_cluster/kmeans_labels.npy')
    kmeans_model = open_pickle_file('saved_cluster/kmeans_model.pkl')
    embedding_matrix = np.load('/scratch/dburger1/proj/embedding_matrix.npy')
    english_df = pd.read_pickle("/scratch/dburger1/proj/english_songs.pkl")
    os.system('clear')



    print(r" _               _        __  __       _       _     ")
    print(r"| |   _   _ _ __(_) ___  |  \/  | __ _| |_ ___| |__  ")
    print(r"| |  | | | | '__| |/ __| | |\/| |/ _` | __/ __| '_ \ ")
    print(r"| |__| |_| | |  | | (__  | |  | | (_| | || (__| | | |")
    print(r"|_____\__,_|_|  |_|\___| |_|  |_|\__,_|\__\___|_| |_|")
    print(r"      |___/                                          ")
    print("\n")
    
    print("Let's Make a Playlist From Almost 3 Million Options!")
    print("Type 0 to exit")
    while True:
        N = input("How many songs would you like to see? (default is 10): ").strip()
        try:
           N = int(N)
        except ValueError:
           print("Invalid input. Next time please enter a number.")
           N = 10
        name = input('What artist are you listening to?: ').strip().lower().title()
        song_title = input('What song are you listening to?: ').strip().lower().title()

        if name == '0' or song_title == '0':
            sys.exit(0)
            
        lyrics = get_lyrics(name, song_title)
        if lyrics:
            break
        print(f"Couldn't find lyrics for {song_title} by {name}. Please try again.")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    lyrics = clean_lyrics(lyrics)
    embedded_lyrics = embed_song_from_lyrics(lyrics, model)
        
    gmm_matches = find_closest_songs_in_cluster(embedded_lyrics, embedding_matrix, gmm_labels, gmm_model, N, 'gmm', song_title, index_to_title)
    kmeans_matches = find_closest_songs_in_cluster(embedded_lyrics, embedding_matrix, kmeans_labels, kmeans_model, N, 'kmeans', song_title, index_to_title)
    cosine_matches = find_closest_songs_global(embedded_lyrics, embedding_matrix, N, song_title, index_to_title)    
    
    gmm_lyrics = get_lyrics_by_indices(gmm_matches, english_df)
    kmeans_lyrics = get_lyrics_by_indices(kmeans_matches, english_df)
    cosine_lyrics = get_lyrics_by_indices(cosine_matches, english_df)

    os.system('clear')
    print(r" _               _        __  __       _       _     ")
    print(r"| |   _   _ _ __(_) ___  |  \/  | __ _| |_ ___| |__  ")
    print(r"| |  | | | | '__| |/ __| | |\/| |/ _` | __/ __| '_ \ ")
    print(r"| |__| |_| | |  | | (__  | |  | | (_| | || (__| | | |")
    print(r"|_____\__,_|_|  |_|\___| |_|  |_|\__,_|\__\___|_| |_|")
    print(r"      |___/                                          ")

    
    print("\n")
    print("Fetching artists and formatting matches, please wait...")
    print("\n")
    display_playlist(song_title, gmm_matches, index_to_title, find_artist_by_title, N, name, 'GMM Clusters')
    print("\n")
    print("Fetching artists and formatting matches, please wait...")
    print("\n")
    display_playlist(song_title, kmeans_matches, index_to_title, find_artist_by_title, N, name, 'K-Means Clusters')
    print("\n")
    print("Fetching artists and formatting matches, please wait...")
    print("\n")
    display_playlist(song_title, cosine_matches, index_to_title, find_artist_by_title, N, name, 'Global Cosine Similarity')
    
    
    
    
    
main()