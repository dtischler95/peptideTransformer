from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import umap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import manifold, metrics
from sklearn.cluster import KMeans
from warnings import filterwarnings
from collections import Counter
from transformers import BertTokenizer, BertModel
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import embeddings


def encode_peptides(sequence_file: str,
                    tokenizer_and_model: [BertTokenizer, BertModel] or None = None,
                    batch_size: int = 32,
                    to_file_path: str or None = None) -> [np.ndarray, list]:
    if tokenizer_and_model is None:
        # Load the pre-trained model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd",
                                                  clean_up_tokenization_spaces=True)
        model = BertModel.from_pretrained("../../peptideBERT")
    else:
        tokenizer, model = tokenizer_and_model

    model.eval()

    df = pd.read_csv(sequence_file, sep=';')
    # Prepare peptides
    peptides_prepared = [' '.join(pep) for pep in df['sequence'].to_list()]

    # Generate embeddings in batches
    progress = 0
    embeddings = []
    for i in range(0, len(peptides_prepared), batch_size):
        batch_peptides = peptides_prepared[i:i + batch_size]
        enc = tokenizer(batch_peptides, return_tensors="pt", padding='max_length', truncation=True, max_length=36)
        outputs = model(**enc)
        batch_embeddings = outputs.pooler_output.detach().numpy()
        embeddings.append(batch_embeddings)
        progress += len(batch_peptides)
        print(f"[Embedding] Progress: {progress}/{len(peptides_prepared)}")

    # Concatenate all batch embeddings
    embeddings = np.vstack(embeddings)

    if to_file_path is not None:
        # save embeddings to npz file
        np.savez_compressed(f"{to_file_path}/embeddings.npz", embeddings)
        np.savez_compressed(f"{to_file_path}/labels.npz", df['label'].to_list())

    return embeddings, df['label'].to_list()


def perform_tsne_clustering(sequence_embeddings, sequence_labels, plot_path="../../plots"):
    # Apply t-SNE
    print("[t-SNE] Applying t-SNE...")
    tsne = TSNE(n_components=2,
                perplexity=15,
                init="random",
                method="exact",
                max_iter=1000,
                random_state=42,
                verbose=0)
    tsne_results = tsne.fit_transform(sequence_embeddings)
    print("[t-SNE] Done.")

    print("[KMeans] Applying KMeans clustering...")
    # Apply KMeans clustering on t-SNE results
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(tsne_results)

    print("[KMeans] Done.")


    plot_clusters(tsne_results, kmeans_labels, sequence_labels, plot_path=plot_path)

    return tsne_results, kmeans_labels


def plot_clusters(tsne_results, kmeans_labels, ground_truth_labels, plot_path=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot with KMeans labels
    ax1.scatter(tsne_results[:, 0], tsne_results[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.5)
    ax1.set_title("t-SNE Clustering with KMeans Labels")
    ax1.set_xlabel("t-SNE Component 1")
    ax1.set_ylabel("t-SNE Component 2")

    # Plot with Ground Truth labels
    ax2.scatter(tsne_results[:, 0], tsne_results[:, 1], c=ground_truth_labels, cmap='viridis', alpha=0.5)
    ax2.set_title("t-SNE Clustering with Ground Truth Labels")
    ax2.set_xlabel("t-SNE Component 1")
    ax2.set_ylabel("t-SNE Component 2")

    if plot_path:
        plt.savefig(f"{plot_path}/tsne_test.pdf")
        plt.clf()
    else:
        plt.show()


def tmp_main(as_npz: bool = False):

    if as_npz:
        embedding = np.load("../../data/out/embeddings.npz")['arr_0']
        labels = np.load("../../data/out/labels.npz")['arr_0']
    else:
        embedding, labels = encode_peptides(sequence_file="../../data/train_data/whitelab_hemo_data.csv",
                                            batch_size=32,
                                            to_file_path="../../data/out")


    tsne_results, kmeans_labels = perform_tsne_clustering(embedding, labels, plot_path="../../plots")


# Example usage
if __name__ == "__main__":
    tmp_main(as_npz=False)
