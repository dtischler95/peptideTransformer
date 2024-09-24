import polars as pl
import torch
import umap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn import manifold, metrics
from sklearn.cluster import KMeans
from warnings import filterwarnings
from collections import Counter

from transformers import BertTokenizer, BertModel



def amino_acid_frequency_comparison(
    sequences_0, sequences_1, plot_path=""
):
    def _tokenize(sequences) -> dict:
        counts = Counter()
        for seq in sequences:
            counts += Counter(seq)
        return counts

    counts_0 = _tokenize(sequences_0)
    counts_1 = _tokenize(sequences_1)

    tick_labels = sorted(set(counts_0.keys())) #| set(counts_1.keys()))
    vec_0 = np.array([counts_0[k] for k in tick_labels])
    vec_0 = vec_0 / vec_0.sum()

    vec_1 = np.array([counts_1[k] for k in tick_labels])
    vec_1 = vec_1 / vec_1.sum()

    x = np.arange(len(tick_labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    fig.suptitle("Amino acid frequency comparison", fontsize=12)
    ax1.bar(x - 0.2, vec_0, 0.4, label="Negativ")
    ax1.bar(x + 0.2, vec_1, 0.4, label="Positiv")
    plt.xticks(x, tick_labels)
    ax1.set_ylabel("AA Frequency")
    ax1.legend()
    ax1.set_xticks(x, tick_labels)
    ax1.set_xlabel("Amino Acids")

    # Train - gen
    ax2.bar(x, vec_0 - vec_1, alpha=0.75)
    ax2.set_ylabel("Difference in Relative Frequency")
    ax2.set_xticks(x, tick_labels)
    ax2.set_xlabel("Amino Acids")
    ax2.set_title("Negativ - Positiv")

    if plot_path != "":
        plt.savefig(f"{plot_path}.pdf")
        plt.clf()
    else:
        plt.show()


def plot_density(x, y, labels, ax):
    return sns.kdeplot(
        x=x,
        y=y,
        linewidths=1,
        ax=ax,
        hue=labels,
        legend=False,
        palette={1: "darkgrey", 0: "lawngreen"},
    )


# Positive 0 = helix, Negative 1 = beta
def seperate_points(data, labels):
    positive, negative = [], []
    for label in range(len(labels)):
        if labels[label] == 1:
            positive.append(data[label])
        else:
            negative.append(data[label])
    return np.array(positive), np.array(negative)


def plot_pca(pca, pca_fit, labels, plot_path=""):
    pos, neg = seperate_points(pca_fit, labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    ax1.scatter(
        x=pos[:, 0],
        y=pos[:, 1],
        c="lawngreen",
        alpha=0.7,
        label=f"Negativ: {labels.count(0)}",
        s=10
    )
    ax1.scatter(
        x=neg[:, 0],
        y=neg[:, 1],
        c="darkgrey",
        alpha=0.7,
        label=f"Positiv: {labels.count(1)}",
        s=10
    )
    """    plot_density(
        x=pca_fit[:, 0],
        y=pca_fit[:, 1],
        ax=ax1,
        labels=labels,
    )"""

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    ax1.set_title("PCA on embedding data")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")
    ax2.set_title("PCA explained variance ratio")
    ax2.set_ylabel("Explained variance ratio")
    ax2.set_xlabel("Principal components")
    plt.suptitle("PCA Plot analysis")
    ax2.plot(np.arange(30) + 1, pca.explained_variance_ratio_, "o-", linewidth=2)
    """ax1.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="blue",
        zorder=10,
    )  # Plot cluster centers"""

    if plot_path != "":
        plt.savefig(f"{plot_path}.pdf")
        plt.clf()
    else:
        plt.show()


def plot_tsne(tsne_fit, labels, plot_path=""):
    pos, neg = seperate_points(tsne_fit, labels)
    figr, ax = plt.subplots(1, 1)

    ax.scatter(
        x=pos[:, 0],
        y=pos[:, 1],
        c="lawngreen",
        alpha=0.7,
        label=f"Negativ: {labels.count(0)}",
        s=10
    )
    ax.scatter(
        x=neg[:, 0],
        y=neg[:, 1],
        c="darkgrey",
        alpha=0.7,
        label=f"Positiv: {labels.count(1)}",
        s=10
    )

    plot_density(
        x=tsne_fit[:, 0],
        y=tsne_fit[:, 1],
        ax=ax,
        labels=labels,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_title("TSNE on embedding data")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if plot_path != "":
        plt.savefig(f"{plot_path}.pdf")
        plt.clf()
    else:
        plt.show()


def plot_umap(data, labels, plot_path=""):
    reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
    umapped = reducer.fit_transform(data)
    pos, neg = seperate_points(umapped, labels)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(
        x=pos[:, 0],
        y=pos[:, 1],
        c="lawngreen",
        alpha=0.7,
        label=f"Negativ: {labels.count(0)}",
        s=10
    )
    ax.scatter(
        x=neg[:, 0],
        y=neg[:, 1],
        c="darkgrey",
        alpha=0.7,
        label=f"Positiv: {labels.count(1)}",
        s=10
    )

    plot_density(
        x=umapped[:, 0],
        y=umapped[:, 1],
        ax=ax,
        labels=labels,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_title("UMAP on embedding data")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if plot_path != "":
        plt.savefig(f"{plot_path}.pdf")
        plt.clf()
    else:
        plt.show()
    return umapped


def perform_culstering(data, labels):
    def cluster_metrics(data, labels, cluster_labels, tag="", show=False):
        print(
            f"[Clustering]: Positive class: {list(cluster_labels).count(1)} Negative class: {list(cluster_labels).count(0)}"
        )
        unique_labels = set(cluster_labels)
        if len(unique_labels) < 2:
            print(f"Skipping silhouette score calculation for {tag} clustering due to insufficient unique labels.")
            print(f"Unique labels: {unique_labels}")
            silhouette_score = None
        else:
            silhouette_score = metrics.silhouette_score(
                data, cluster_labels, sample_size=10, random_state=42
            )

        res = {
            "hom": metrics.homogeneity_score(labels, cluster_labels),
            "comp": metrics.completeness_score(labels, cluster_labels),
            "v_score": metrics.v_measure_score(labels, cluster_labels),
            "adj_rand_idx": metrics.adjusted_rand_score(labels, cluster_labels),
            "shil": silhouette_score,
            "mifs": metrics.mutual_info_score(labels, cluster_labels),
        }
        if show:
            print("----------------------------------------------------")
            print(f"\t Metrics Report for {tag} Clustering")
            print("----------------------------------------------------")
            print(f"Homogeneity:\t\t{res['hom']:4f}")
            print(f"Completness:\t\t{res['comp']:4f}")
            print(f"V-measure:\t\t{res['v_score']:4f}")
            print(f"Adjusted Rand-Index:\t\t{res['adj_rand_idx']:4f}")
            if silhouette_score is not None:
                print(f"Silhouette Score:\t\t{res['shil']:4f}")
            print(f"Mutual Information Score: {res['mifs']:4f}")
            print("----------------------------------------------------")
        return pl.from_dict(res)

    def run_pca(data, comps=30):
        print("[Clustering] Running PCA")
        pca = PCA(n_components=comps, random_state=42)
        return pca, pca.fit(data).transform(data)

    def run_tsne(data):
        print("[Clustering] Running TSNE")
        tsne = manifold.TSNE(
            n_components=2,
            perplexity=5.0,
            init="random",
            method="exact",
            n_iter=1000,
            random_state=42,
            verbose=0,
        )
        return tsne.fit_transform(data)

    def run_umap(data):
        print("[Clustering] Running UMAP")
        reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)

        return reducer.fit_transform(data)

    def clustering(data, labels):
        print("[Clustering] Running KMeans")
        kmeans = KMeans(n_clusters=2, max_iter=100, n_init=5, random_state=42)
        kmeans.fit(data)

        scores = cluster_metrics(data, labels, kmeans.labels_, tag="KMeans", show=True)
        # scores.write_csv(output_path, separator=";", include_header=True)
        return kmeans, kmeans.labels_


    pca, pca_fit = run_pca(data)
    tsne_fit = run_tsne(data)
    umap_fit = run_umap(data)

    print("[Clustering] Running KMeans for PCA")
    pca_kmeans, pca_kmeans_labels = clustering(pca_fit, labels)
    print("[Clustering] Running KMeans for TSNE")
    tsne_kmeans, tnse_kmeans_labels = clustering(tsne_fit, labels)
    print("[Clustering] Running KMeans for UMAP")
    umap_kmeans, umap_kmeans_labels = clustering(umap_fit, labels)
    try:
        print("[Clustering] Plotting PCA")
        plot_pca(pca, pca_fit, labels, plot_path="../data/out/pca_plot")
    except Exception as e:
        print(f"PCA Plotting failed: {e}")
    print("[Clustering] Plotting TSNE")
    plot_tsne(tsne_fit, labels, plot_path="../data/out/tsne_plot")
    print("[Clustering] Plotting UMAP")
    plot_umap(umap_fit, labels, plot_path="../data/out/umap_plot")



def read_apd3(apd_path, min_len=10, max_len=50):
    apd3 = (
        pl.read_csv(
            apd_path,
            separator=";",
            has_header=True,
        )
    )


    # helix = helix.filter(~pl.col("sequence").is_in(beta.get_column("sequence")))





    # Write Seqs once

    return apd3



def encode_peptides(peptides, batch_size: int = 32):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Transformer is using device: {device}")

    # Load the pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("./peptideBERT_model")
    model.eval()
    # Prepare peptides
    peptides_prepared = [' '.join(pep) for pep in peptides]

    # TODO PUSHEN!

    # Generate embeddings in batches
    progress = 0
    embeddings = []
    for i in range(0, len(peptides_prepared), batch_size):
        batch_peptides = peptides_prepared[i:i + batch_size]
        enc = tokenizer(batch_peptides, return_tensors="pt", padding='max_length', truncation=True, max_length=36)
        outputs = model(**enc)
        batch_embeddings = outputs.last_hidden_state.detach().numpy()
        embeddings.append(batch_embeddings)
        progress += len(batch_peptides)
        print(f"[Embedding] Progress: {progress}/{len(peptides_prepared)}")


    # Concatenate all batch embeddings
    embeddings = np.vstack(embeddings)

    return embeddings


# TODO later include 2rd class?
def main(file_path: str, batch_size: int):
    df = read_apd3(file_path)


    labels = df.get_column("label").to_list()
    seqs = df.get_column("sequence").to_list()

    embedded_seqs = encode_peptides(seqs, batch_size=batch_size)
    perform_culstering(embedded_seqs, labels)



def main_plot_amino(file_path: str):
    df = pd.read_csv(file_path, sep=';')
    neg_seqs = df[df['label'] == 1]['sequence'].to_list()
    pos_seqs = df[df['label'] == 0]['sequence'].to_list()
    amino_acid_frequency_comparison(neg_seqs, pos_seqs, plot_path="../data/hemo/amino_acid_freq_comparison")

if __name__ == "__main__":
    filterwarnings("ignore", category=UserWarning)
    main(file_path="../data/base_data/splitted_hemo_labeled.csv", batch_size=64)
    # main_plot_amino(file_path="../data/hemo/splitted_hemo_labeled.csv")

