from pathlib import Path
from sklearn.cluster import KMeans
import numpy as np
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def open_pickle_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def reduce_matrix(embeddings):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)
    return reduced

def plot_gmm_clusters(index_to_title, embedding_matrix, reduced_matrix, gmm_labels, gmm_model, k_size, subfolder):
   
    plt.figure(figsize=(12, 9))
    plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], c=gmm_labels, cmap='tab10', alpha=0.6)
    plt.title(f"t-SNE of GMM Clusters (K={k_size})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)

    output_path = subfolder/"clusters.txt"
    with open(output_path, 'a') as f:
        f.write(f"\n=== GMM Clusters for k={k_size} ===\n")

    labels_per_cluster=5
    for cluster_id in np.unique(gmm_labels):
        indices = np.where(gmm_labels == cluster_id)[0]
        cluster_vectors = embedding_matrix[indices]
        center = gmm_model.means_[cluster_id]  # GMM cluster mean

        distances = euclidean_distances(cluster_vectors, center.reshape(1, -1)).flatten()
        sorted_indices = indices[np.argsort(distances)]

        for rank, song_idx in enumerate(sorted_indices[:labels_per_cluster]):
            x, y = reduced_matrix[song_idx]
            title = index_to_title.get(song_idx, "Unknown")
            with open(output_path, 'a') as f:
                f.write(f"GMM Cluster {cluster_id + 1} Rank {rank + 1}: {title}")
            plt.text(x, y, f"{title}", fontsize=6, ha='center', va='center',
                     bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))

    plt.tight_layout()
    return plt


def plot_clusters(index_to_title, embedding_matrix, reduced_matrix, cluster_labels, kmeans, k_size, subfolder):
    plt.figure(figsize=(12, 9))
    plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6)
    plt.title(f"t-SNE of Clusters (K={str(k_size)})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    
    output_path = subfolder/"clusters.txt"
    with open(output_path, 'a') as f:
        f.write(f"\n=== K-Means Clusters for k={k_size} ===\n")

    for cluster_id in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_vectors = embedding_matrix[indices]
        center = kmeans.cluster_centers_[cluster_id]
        
        distances = euclidean_distances(cluster_vectors, center.reshape(1, -1)).flatten()
        closest_idx_in_cluster = indices[np.argmin(distances)]
        
        sorted_indices = indices[np.argsort(distances)]
        labels_per_cluster = 5
        for rank, song_idx in enumerate(sorted_indices[:labels_per_cluster]):
            x, y = reduced_matrix[song_idx]
            title = index_to_title.get(song_idx, "Unknown")
            with open(output_path, 'a') as f:
                f.write(f"Cluster {cluster_id + 1} Rank {rank +1}: {title}")
            plt.text(x, y, f"{title}", fontsize=6, ha='center', va='center',
                     bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))
            
    plt.tight_layout()
    return plt
    

def plot_elbow_curve(embedding_matrix, max_k=20):
    inertias = []
    k_values = range(2, max_k + 1)

    print("Calculating SSE for different K values...")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embedding_matrix)
        inertias.append(kmeans.inertia_)
        print(f"K={k}, Inertia={kmeans.inertia_:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cluster_imgs/elbow_plot.png", dpi=300)
    plt.show()

def tests():
    #embeddings
    unique_rows = np.unique(embedding_matrix, axis=0)
    print(f"Total embeddings: {embedding_matrix.shape[0]}")
    print(f"Unique embeddings: {unique_rows.shape[0]}")
    # Check mean and standard deviation across dimensions
    mean_std = embedding_matrix.std(axis=0).mean()
    print(f"Average dimension-wise stddev: {mean_std:.5f}") 
    random_labels = np.random.randint(0, num_clusters, size=len(cluster_labels))
    print("Random vs KMeans ARI:", adjusted_rand_score(random_labels, cluster_labels))
    ari = adjusted_rand_score(cluster_labels, gmm_labels)
    nmi = normalized_mutual_info_score(cluster_labels, gmm_labels)

    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Info: {nmi:.4f}")


def main():
    num_clusters = 8
    #Loading BERT embeddings
    embedding_matrix = np.load('/scratch/dburger1/proj/embedding_matrix.npy')
    
    with open('/scratch/dburger1/proj/index_to_title.pkl', 'rb') as f:
        index_to_title = pickle.load(f)
    print("Retrieved Matrix Embedding")
    #plot_elbow_curve(embedding_matrix, max_k=15)
    folder = Path("cluster_imgs/")
    folder.mkdir(exist_ok=True)
    subfolder = folder/Path(f"k={str(num_clusters)}")
    subfolder.mkdir(exist_ok=True)
    save = Path("saved_cluster/")
    print("beginning GMM cluster")
    gmm = GaussianMixture(num_clusters, covariance_type='full', init_params="random_from_data", random_state=42)
    print("created GMM model")
    gmm_labels = gmm.fit_predict(embedding_matrix)
    print("created GMM labels")
    
    print("creating Kmeans model")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    print("creating Kmeans labels")
    cluster_labels = kmeans.fit_predict(embedding_matrix)
    print("Created Clusters!")

    
    np.save(save/"kmeans_labels.npy", cluster_labels)
    np.save(save / "gmm_labels.npy", gmm_labels)
    with open(save / "kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open(save / "gmm_model.pkl", "wb") as f:
        pickle.dump(gmm, f)
    print("saved models successfully")
    
    matrix = reduce_matrix(embedding_matrix)
    print("Created Reduced Matrix")
    #print(matrix)
    
    plot = plot_clusters(index_to_title, embedding_matrix, matrix, cluster_labels, kmeans, num_clusters, subfolder)
    plot.savefig(subfolder/f"k_means.png")
    print("created kmeans plot")
    gmm_plot = plot_gmm_clusters(index_to_title, embedding_matrix, matrix, gmm_labels, gmm, num_clusters, subfolder)
    gmm_plot.savefig(subfolder/f"gmm_cluster.png")
    print("created gmm plot")
    print("Success!")
    return  
main()



