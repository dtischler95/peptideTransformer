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
from transformers import BertTokenizer, BertModel

def plot_density(x, y, labels, ax):
    """
    Plot density of the data points in the scatter plot using seaborn kdeplot function.

    :param x: x-axis data
    :param y: y-axis data
    :param labels: labels of the data points
    :param ax: axis to plot on
    """
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
    """
    Seperate the data points based on their labels.

    :param data: data points
    :param labels: labels of the data points
    """
    positive, negative = [], []
    for label in range(len(labels)):
        if labels[label] == 1:
            positive.append(data[label])
        else:
            negative.append(data[label])
    return np.array(positive), np.array(negative)


def plot_pca(pca, pca_fit, labels, plot_path=""):
    """
    Plot PCA analysis of the data points.

    :param pca: PCA object
    :param pca_fit: PCA transformed data
    :param labels: labels of the data points
    :param plot_path: path to save the plot
    """
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
    """
    Plot TSNE analysis of the data points.

    :param tsne_fit: TSNE transformed data
    :param labels: labels of the data points
    :param plot_path: path to save the plot
    """
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
    """
    Plot UMAP analysis of the data points.

    :param data: data points
    :param labels: labels of the data points
    :param plot_path: path to save the plot
    """
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


def perform_clustering(embedded_sequences,
                       sequence_labels,
                       plot_path: str,
                       tag: str= ""):
    """
    Perform clustering analysis on the data points.
    Wrapper for the basic clustering analysis.

    :param embedded_sequences: data points
    :param sequence_labels: labels of the data points. Needs to be index paired with the data points
    :param plot_path: path to sav{plot_path
    :param tag: tag for the output files
    """
    def cluster_metrics(data, labels, cluster_labels, cluster_tag=""):
        """
        Print and return clustering metrics.

        :param data: data points
        :param labels: true labels
        :param cluster_labels: predicted labels
        :param cluster_tag: tag for prints
        """
        print(
            f"[Clustering]: Positive class: {list(cluster_labels).count(1)} Negative class: {list(cluster_labels).count(0)}"
        )
        unique_labels = set(cluster_labels)
        if len(unique_labels) < 2:
            print(f"Skipping silhouette score calculation for {cluster_tag} clustering due to insufficient unique labels.")
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

        print("----------------------------------------------------")
        print(f"\t Metrics Report for {cluster_tag} Clustering")
        print("----------------------------------------------------")
        print(f"Homogeneity:\t\t{res['hom']:4f}")
        print(f"Completness:\t\t{res['comp']:4f}")
        print(f"V-measure:\t\t{res['v_score']:4f}")
        print(f"Adjusted Rand-Index:\t\t{res['adj_rand_idx']:4f}")
        if silhouette_score is not None:
            print(f"Silhouette Score:\t\t{res['shil']:4f}")
        print(f"Mutual Information Score: {res['mifs']:4f}")
        print("----------------------------------------------------")


    def run_pca(data, comps=12):
        print("[Clustering] Running PCA")
        _pca = PCA(n_components=comps, random_state=42)
        return _pca, _pca.fit(data).transform(data)

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

        cluster_metrics(data, labels, kmeans.labels_, cluster_tag="KMeans")
        # scores.write_csv(output_path, separator=";", include_header=True)
        return kmeans, kmeans.labels_

    try:
        pca, pca_fit = run_pca(embedded_sequences)
    except Exception as e:
        print(f"PCA failed: {e}")
        pca = None
        pca_fit = None

    tsne_fit = run_tsne(embedded_sequences)
    umap_fit = run_umap(embedded_sequences)

    print("[Clustering] Running KMeans for PCA")
    pca_kmeans, pca_kmeans_labels = clustering(pca_fit, sequence_labels)
    print("[Clustering] Running KMeans for TSNE")
    tsne_kmeans, tnse_kmeans_labels = clustering(tsne_fit, sequence_labels)
    print("[Clustering] Running KMeans for UMAP")
    umap_kmeans, umap_kmeans_labels = clustering(umap_fit, sequence_labels)
    try:
        print("[Clustering] Plotting PCA")
        plot_pca(pca, pca_fit, sequence_labels, plot_path=f"{plot_path}/{tag}_pca_plot")
    except Exception as e:
        print(f"PCA Plotting failed: {e}")
    print("[Clustering] Plotting TSNE")
    plot_tsne(tsne_fit, sequence_labels, plot_path=f"{plot_path}/{tag}_tsne_plot")
    print("[Clustering] Plotting UMAP")
    plot_umap(umap_fit, sequence_labels, plot_path=f"{plot_path}/{tag}_umap_plot")

    try:
        print("[Clustering] Plotting PCA")
        plot_pca(pca, pca_fit, pca_kmeans_labels.tolist(), plot_path=f"{plot_path}/{tag}_kmeans_label_pca_plot")
    except Exception as e:
        print(f"PCA Plotting failed: {e}")
    print("[Clustering] Plotting TSNE")
    plot_tsne(tsne_fit, tnse_kmeans_labels.tolist(), plot_path=f"{plot_path}/{tag}_kmeans_label_tsne_plot")
    print("[Clustering] Plotting UMAP")
    plot_umap(umap_fit, umap_kmeans_labels.tolist(), plot_path=f"{plot_path}/{tag}_kmeans_label_umap_plot")



def encode_peptides(sequence_file: str,
                    device,
                    tokenizer_and_model: [BertTokenizer, BertModel] or None = None,
                    batch_size: int = 32) -> [np.ndarray, list]:


    if tokenizer_and_model is None:
        # Load the pre-trained model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False, clean_up_tokenization_spaces=True)
        model = BertModel.from_pretrained("Rostlab/prot_bert")


    else:
        tokenizer, model = tokenizer_and_model

    model.eval()

    df = pd.read_csv(sequence_file, sep=';')
    # Shuffle the data to ensure labels are mixed
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[:100]
    # Prepare peptides
    peptides_prepared = [' '.join(pep) for pep in df['sequence'].to_list()]

    # Generate embeddings in batches
    progress = 0
    embeddings = []
    for i in range(0, len(peptides_prepared), batch_size):
        batch_peptides = peptides_prepared[i:i + batch_size]
        enc = tokenizer(batch_peptides, return_tensors="pt", padding='max_length', truncation=True, max_length=36)

        enc = {key: value.to(device) for key, value in enc.items()}

        outputs = model(**enc, output_hidden_states=True)

        cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu()

        embeddings.append(cls_embeddings.detach().numpy())
        progress += len(batch_peptides)
        print(f"[Embedding] Progress: {progress}/{len(peptides_prepared)}")

    # Concatenate all batch embeddings
    embeddings = np.vstack(embeddings)

    return embeddings, df['label'].to_list()


# TODO later include 2rd class?
def cluster_model_embedding(file_path: str,
                            batch_size: int,
                            plot_path:str,
                            device,
                            tokenizer_and_model: [BertTokenizer, BertModel] or None = None):
    """
    Standalone Wrapper for clustering analysis if you run this file directly.
    Thought for if you already have a model, and you want to further analyze the data.
    A basic clustering will be done in the fine tune script.

    :param file_path: Path to the csv file containing the sequences and labels
    :param batch_size: Batch size for encoding the sequences
    :param plot_path: Path to save the plots
    :param device: Device to run the model on
    :param tokenizer_and_model: Tuple containing the tokenizer and model
    """

    # Generating a tag for output files to be unique
    data_tag = file_path.split("/")[-1].split(".")[0]
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    embedding, labels = encode_peptides(sequence_file=file_path,
                                        batch_size=batch_size,
                                        tokenizer_and_model=tokenizer_and_model,
                                        device=device)
    perform_clustering(embedded_sequences=embedding,
                       sequence_labels=labels,
                       tag=data_tag,
                       plot_path=plot_path)



if __name__ == "__main__":
    filterwarnings("ignore", category=UserWarning)
    cluster_model_embedding(file_path="../../data/train_data/whitelab_hemo_data.csv",
                            batch_size=64,
                            plot_path="../../plots",
                            tokenizer_and_model=None,
                            device=None)


