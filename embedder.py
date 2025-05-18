from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    df = pd.read_pickle("/scratch/dburger1/proj/big.pkl")
    df = df.reset_index(drop=True)
    df = df[df['cleaned_lyrics'].apply(lambda x: isinstance(x, str) and x.strip() != "")]

    lyrics_texts = df['cleaned_lyrics'].tolist()
    titles = df['title'].tolist()

    embeddings = model.encode(
        lyrics_texts,
        batch_size=32,
        show_progress_bar=True
    )

    embedding_matrix = np.vstack(embeddings)
    np.save("/scratch/dburger1/proj/embedding_matrix.npy", embedding_matrix)

    index_to_title = {i: title for i, title in enumerate(titles)}
    with open("/scratch/dburger1/proj/index_to_title.pkl", "wb") as f:
        pickle.dump(index_to_title, f)

main()
