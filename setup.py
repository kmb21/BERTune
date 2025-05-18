import pandas as pd
import pickle
import re

from langdetect import detect
from collections import defaultdict

def make_dictionary_count(data):
    dict_map = {}
    count = 0
    for item in data:
        dict_map[item] = count
        count += 1
    return dict_map

def clean_lyrics(text):
    """Remove song structure tags and clean text"""
    if not isinstance(text, str):
        return ""
    # Removes [Verse], [Chorus] etc. and non-printable chars
    text = re.sub(r'\[.*?\]|[\x00-\x1F]', ' ', text)
    return ' '.join(text.split())

def is_english(text):
    """Check if text is English using langdetect"""
    try:
        return detect(text) == 'en'
    except:
        return False
    
def process_df(df):
    """Process DataFrame in one pass"""
    df['cleaned_lyrics'] = df['lyrics'].apply(clean_lyrics)
    
    mask = df['cleaned_lyrics'].apply(is_english)
    return df[mask]

def extract_vocab(cleaned_lyrics_series, min_word_length=3):
    """Extract vocabulary from cleaned lyrics"""
    words = set()
    
    for lyrics in cleaned_lyrics_series:
        for word in lyrics.split():
            words.add(word)
    return words

def extract_titles(titles_series):
    """Extract unique titles"""
    return set(titles_series.unique())


def save_pickle(filename, data):
    f = open(filename, 'wb')  # open a file (write, binary mode)
    pickle.dump(data, f)          # save the list to the file
    f.close()

def main():
    df = pd.read_csv("/scratch/dburger1/proj/bigsample.csv.gz", 
                    compression='gzip',
                    usecols=['title', 'lyrics']) 
    
    english_df = process_df(df)
    print(f"Found {len(english_df)} English songs")
    
    english_df.to_pickle("/scratch/dburger1/proj/big.pkl")
    
    
    song_vocab = extract_vocab(english_df['cleaned_lyrics'])
    print(f"Unique words: {len(song_vocab)}")
    words_dict = make_dictionary_count(song_vocab)
    save_pickle("words_dict.pkl",words_dict)
    
    titles = extract_titles(english_df['title'])
    print(f"Unique titles: {len(titles)}")
    titles_dict = make_dictionary_count(titles)
    save_pickle("titles_dict.pkl",titles_dict)
    
    
    
    





if __name__ == "__main__":
    main()
    
    
    
