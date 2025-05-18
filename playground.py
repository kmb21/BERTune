import pandas as pd
from huggingface_hub import hf_hub_download


file_path = hf_hub_download(
    repo_id="amishshah/song_lyrics",
    filename="song_lyrics_min.csv",
    repo_type="dataset",
    cache_dir="/scratch/dburger1/proj/CS91S-25"
)

# df = pd.read_csv(file_path)
# df.to_csv('/scratch/mkumbon1/CS91S-25/song_lyrics_min.csv.gz', compression='gzip', index=False)

df = pd.read_csv(file_path)

df.to_csv(
    '/scratch/dburger1/proj/bigsample.csv.gz', 
    compression='gzip', 
    index=False
)

print(f"Saved test subset with {len(test_df)} rows")
print("Done!")

"""
After each new data file added, run:
chmod 666 filename
chmod -R a+rwx /scratch/dburger1/proj/
to make sure that data is accessible by both of us
"""