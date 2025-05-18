# BERTune

BERTune is a lyric-based song recommendation system. It leverages BERT embeddings to capture the semantic meaning of song lyrics and recommends similar songs using clustering techniques (K-Means and Gaussian Mixture Models) as well as direct cosine similarity.

## Overview

The project aims to provide users with personalized song playlists based on the lyrical content of a song they like. By inputting an artist and a song title, users receive a list of N similar songs. The similarity is determined by analyzing a large corpus of song lyrics, embedding them into a vector space, and then finding neighbors using different methodologies.

## Features

*   **Lyric-Based Recommendations**: Suggests songs with similar lyrical themes and content.
*   **BERT Embeddings**: Utilizes `sentence-transformers` (specifically `all-MiniLM-L6-v2`) to generate dense vector representations of song lyrics.
*   **Multiple Recommendation Strategies**:
    *   **K-Means Clustering**: Finds songs within the same cluster as the query song.
    *   **Gaussian Mixture Model (GMM) Clustering**: Finds songs within the same GMM component.
    *   **Global Cosine Similarity**: Finds the most similar songs globally across the entire dataset.
*   **Lyrics Fetching**: Integrates with the LyricsGenius API to fetch lyrics for user-provided songs.
*   **Interactive CLI**: A user-friendly command-line interface built with `rich` for querying and displaying playlists.
*   **Cluster Visualization**: t-SNE plots are generated to visualize the song clusters (see `cluster_imgs/` directory for examples).
*   **Data Preprocessing**: Includes scripts for cleaning lyrics, filtering for English songs, and building necessary vocabulary and an index-to-title mapping.

## Project Structure

The repository is organized as follows:

*   `main.py`: The main script to run the interactive song recommendation system.
*   `embedder.py`: Script to generate BERT embeddings from processed lyrics and save them.
*   `cluster.py`: Script for performing K-Means and GMM clustering on the embeddings, generating plots, and saving cluster models.
*   `setup.py`: Script for initial data loading, cleaning (removing tags, non-ASCII), language detection, and creation of vocabulary/title dictionaries.
*   `query.py`: Contains utility functions for fetching lyrics via LyricsGenius and an alternative TF-IDF based querying mechanism.
*   `playground.py`: A utility script for downloading the initial dataset from Hugging Face Hub.
*   `cluster_imgs/`: Directory containing visualizations (e.g., `k_means.png`, `gmm_cluster.png`) and text files (`clusters.txt`) listing top songs for different cluster configurations (k values).
*   `saved_cluster/`: Directory where trained K-Means and GMM models (`.pkl`) and their corresponding labels (`.npy`) are stored.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kmb21/BERTune.git
    cd BERTune
    ```

2.  **Install dependencies:**
    Key Python libraries used include:
    *   `sentence-transformers`
    *   `pandas`
    *   `numpy`
    *   `scikit-learn`
    *   `matplotlib`
    *   `lyricsgenius`
    *   `rich`
    *   `langdetect`

    It's recommended to set up a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install sentence-transformers pandas numpy scikit-learn matplotlib lyricsgenius rich langdetect
    ```

3.  **LyricsGenius API Token:**
    You will need a LyricsGenius API token. Obtain one from [Genius API Management](http://genius.com/api-clients). The token is hardcoded in `main.py` and `query.py`. You should replace the placeholder token:
    ```python
    token = "YOUR_LYRICSGENIUS_API_TOKEN"
    ```

4.  **Data Acquisition and Preparation:**
    *   The scripts use hardcoded paths (e.g., `/scratch/dburger1/proj/`). You will need to adjust these paths to your local environment throughout the scripts (`playground.py`, `setup.py`, `embedder.py`, `cluster.py`, `main.py`).
    *   **Download Data**: `playground.py` is set up to download `song_lyrics_min.csv` from `amishshah/song_lyrics` on Hugging Face Hub. Modify the `cache_dir` in `playground.py` and run it.
        ```bash
        python playground.py
        ```
        This script also saves a gzipped CSV (`bigsample.csv.gz`).
    *   **Preprocess Data**: Run `setup.py` to clean lyrics, filter for English songs, and create necessary data files (`big.pkl`, `words_dict.pkl`, `titles_dict.pkl`).
        ```bash
        python setup.py
        ```
    *   **Generate Embeddings**: Run `embedder.py` to compute BERT embeddings for the lyrics and save `embedding_matrix.npy` and `index_to_title.pkl`. This step requires a CUDA-enabled GPU for efficient processing as configured (`device='cuda'`).
        ```bash
        python embedder.py
        ```
    *   **Perform Clustering**: Run `cluster.py` to perform K-Means and GMM clustering. This will save cluster models (`kmeans_model.pkl`, `gmm_model.pkl`) and labels (`kmeans_labels.npy`, `gmm_labels.npy`) in the `saved_cluster/` directory. It also generates t-SNE plots and cluster summaries in `cluster_imgs/`. You can adjust `num_clusters` in `cluster.py`.
        ```bash
        python cluster.py
        ```

## Usage

Once the setup and data preparation steps are complete, you can run the main application:

```bash
python main.py
```

The script will display an ASCII art title and then prompt you for:
1.  The number of songs you'd like in the playlist (default is 10).
2.  The artist's name.
3.  The song title.

After fetching and embedding the lyrics for your input song, the system will display three playlists of recommended songs based on:
*   GMM Clustering
*   K-Means Clustering
*   Global Cosine Similarity

Each playlist is presented in a formatted table using `rich`, showing the rank, song title, and artist.

## Key Functionalities

*   **`setup.py`**:
    *   Loads raw song data (title, lyrics).
    *   `clean_lyrics()`: Removes structural tags (e.g., `[Verse]`) and non-printable characters.
    *   `is_english()`: Uses `langdetect` to filter for English songs.
    *   Extracts unique vocabulary and song titles, creating mappings (`words_dict.pkl`, `titles_dict.pkl`).
    *   Saves the processed English songs DataFrame (`big.pkl`).

*   **`embedder.py`**:
    *   Loads the processed DataFrame (`big.pkl`).
    *   Uses `SentenceTransformer('all-MiniLM-L6-v2')` to encode cleaned lyrics.
    *   Saves the resulting `embedding_matrix.npy` and an `index_to_title.pkl` mapping.

*   **`cluster.py`**:
    *   Loads `embedding_matrix.npy` and `index_to_title.pkl`.
    *   Performs K-Means clustering (`sklearn.cluster.KMeans`).
    *   Performs Gaussian Mixture Model clustering (`sklearn.mixture.GaussianMixture`).
    *   `reduce_matrix()`: Uses t-SNE (`sklearn.manifold.TSNE`) for 2D visualization.
    *   `plot_clusters()` & `plot_gmm_clusters()`: Generate and save scatter plots of clusters with song titles annotated. Results are saved in `cluster_imgs/k=<num_clusters>/`.
    *   Cluster models and labels are saved to `saved_cluster/`.
    *   Includes an `plot_elbow_curve()` function for K-Means experimentation.

*   **`main.py`**:
    *   Loads pre-computed embeddings, cluster models, and mappings.
    *   `get_lyrics()` (from `query.py`): Fetches lyrics for the user's input song using `lyricsgenius`.
    *   `embed_song_from_lyrics()`: Embeds the fetched lyrics using the same Sentence Transformer model.
    *   `find_closest_songs_global()`: Finds N most similar songs using cosine similarity against the entire embedding matrix.
    *   `find_closest_songs_in_cluster()`: Predicts the cluster for the query song (K-Means or GMM) and finds the N most similar songs within that cluster using cosine similarity.
    *   `display_playlist()`: Uses `rich.table.Table` to present the recommendations.
    *   `find_artist_by_title()`: Uses LyricsGenius to look up artist names for recommended songs.

*   **`query.py`**:
    *   Primarily provides `get_lyrics()` for fetching lyrics.
    *   Also contains functions (`build_lyrics_matrix`, commented out TF-IDF logic in `main()`) for an alternative TF-IDF based similarity search, which is not the primary method used in `main.py`.

## Output Examples

The `cluster_imgs/` directory contains examples of outputs from the clustering process:
*   For various values of `k` (number of clusters), `clusters.txt` files list the top 5 songs closest to the centroid of each K-Means cluster and the mean of each GMM component.
*   PNG images (e.g., `k_means.png`, `gmm_cluster.png`) visually represent these clusters after t-SNE dimensionality reduction.

The `saved_cluster/` directory stores the trained `KMeans` and `GaussianMixture` model objects (pickled) and their corresponding label assignments for each song in the dataset (NumPy arrays). This allows `main.py` to quickly assign new songs to clusters and retrieve pre-computed cluster information.
