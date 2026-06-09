import logging
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel

"""
This code is an to my usecase adapted version of the code from Lukas Bayerle 
of the workgroup i work in for Prof. Dr. Franz Cemic.
"""


def seperate_points(data, labels):
    """
    Separate the data points based on their labels.

    :param data: data points
    :param labels: labels of the data points
    """
    positive, negative = [], []
    for label in range(len(labels)):
        if labels[label] == 1:
            positive.append(data[label])
        else:
            negative.append(data[label])
    return np.array(positive).reshape(-1, 2), np.array(negative).reshape(-1, 2)


def plot_pca(pca, pca_fit, labels=None, lengths=None, plot_path=""):
    """
    Plot PCA analysis of the data points.

    :param pca: PCA object (scikit-learn)
    :param pca_fit: PCA transformed data (n_samples, n_components)
    :param labels: class labels (optional)
    :param lengths: sequence lengths (optional) -> if set, color-codes by sequence length
    :param plot_path: path to save the plot (without extension)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    if lengths is not None:
        lengths = np.asarray(lengths, dtype=float)
        sc = ax1.scatter(
            pca_fit[:, 0], pca_fit[:, 1],
            c=lengths, cmap="viridis",
            s=12, alpha=0.85, edgecolors="none"
        )
        cb = fig.colorbar(sc, ax=ax1)
        cb.set_label("Sequence Length (AAs)")
        if labels is not None:
            ax1.legend([f"Positive: {int(np.sum(np.asarray(labels) == 1))}",
                        f"Negative: {int(np.sum(np.asarray(labels) == 0))}"],
                       frameon=True, loc="best")
    else:
        assert labels is not None, "Provide 'labels' for class plot or set 'lengths'."
        labels_arr = np.asarray(labels)
        pos = pca_fit[labels_arr == 1]
        neg = pca_fit[labels_arr == 0]
        ax1.scatter(pos[:, 0], pos[:, 1], c="lawngreen", alpha=0.7,
                    label=f"Positive: {int(np.sum(labels_arr == 1))}", s=12)
        ax1.scatter(neg[:, 0], neg[:, 1], c="darkgrey", alpha=0.7,
                    label=f"Negative: {int(np.sum(labels_arr == 0))}", s=12)
        ax1.legend(loc="best")

    ax1.set_title("PCA on Sequence Embeddings")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")

    # right side: explained variance
    ax2.set_title("PCA Explained Variance")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_xlabel("Principal Components")
    # show the first 10 components
    ncomps = min(10, len(pca.explained_variance_ratio_))
    ax2.plot(np.arange(ncomps) + 1, pca.explained_variance_ratio_[:ncomps],
             "o-", linewidth=2)

    plt.suptitle("PCA Analysis of Sequences")

    if plot_path:
        plt.savefig(f"{plot_path}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_tsne(tsne_fit, labels=None, lengths=None, plot_path="", title="t-SNE on Sequence Embeddings"):
    """
    tsne_fit : array (n,2) – t-SNE coordinates
    labels   : list/array of {0,1} (optional, for class plot)
    lengths  : list/array of sequence lengths (optional, for length plot)
               -> if set, color-codes by length; 'labels' used only for legend counts.
    plot_path: without extension; saves as PDF if set
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    if lengths is not None:
        lengths = np.asarray(lengths, dtype=int)
        sc = ax.scatter(
            tsne_fit[:, 0], tsne_fit[:, 1],
            c=lengths, s=12, alpha=0.85, cmap="viridis", edgecolors="none"
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Sequence Length (AAs)")
        # optional: additional info in legend
        if labels is not None:
            ax.legend([f"Positive: {int(np.sum(np.asarray(labels) == 1))}",
                       f"Negative: {int(np.sum(np.asarray(labels) == 0))}"],
                      frameon=True, loc="best")
    else:
        labels_arr = np.asarray(labels)
        pos = tsne_fit[labels_arr == 1]
        neg = tsne_fit[labels_arr == 0]
        ax.scatter(pos[:, 0], pos[:, 1], c="lawngreen", alpha=0.7,
                   label=f"Positive: {int(np.sum(labels_arr == 1))}", s=12)
        ax.scatter(neg[:, 0], neg[:, 1], c="darkgrey", alpha=0.7,
                   label=f"Negative: {int(np.sum(labels_arr == 0))}", s=12)
        ax.legend(loc="best")

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if plot_path:
        plt.savefig(f"{plot_path}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_umap(data, labels=None, lengths=None, plot_path=""):
    """
    Plot UMAP analysis of the data points.

    :param data: high-dimensional data points
    :param labels: class labels of the data points (optional)
    :param lengths: sequence lengths (optional) -> if set, color-codes by sequence length
    :param plot_path: path to save the plot (without extension)
    """
    reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
    umapped = reducer.fit_transform(data)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    if lengths is not None:
        lengths = np.asarray(lengths, dtype=float)
        sc = ax.scatter(
            umapped[:, 0],
            umapped[:, 1],
            c=lengths,
            cmap="viridis",
            s=12,
            alpha=0.85,
            edgecolors="none"
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Sequence Length (AAs)")

        # optional legend with label counts
        if labels is not None:
            ax.legend([f"Positive: {int(np.sum(np.asarray(labels) == 1))}",
                       f"Negative: {int(np.sum(np.asarray(labels) == 0))}"],
                      frameon=True, loc="best")

    else:

        labels_arr = np.asarray(labels)
        pos = umapped[labels_arr == 1]
        neg = umapped[labels_arr == 0]
        ax.scatter(pos[:, 0], pos[:, 1], c="lawngreen", alpha=0.7,
                   label=f"Positive: {int(np.sum(labels_arr == 1))}", s=12)
        ax.scatter(neg[:, 0], neg[:, 1], c="darkgrey", alpha=0.7,
                   label=f"Negative: {int(np.sum(labels_arr == 0))}", s=12)
        ax.legend(loc="best")

    ax.set_title("UMAP on Sequence Embeddings")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if plot_path:
        plt.savefig(f"{plot_path}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()

    return umapped


def perform_clustering(embedded_sequences,
                       plot_path: str,
                       logger: logging.Logger or None = None,
                       tag: str = ""):
    """
    Perform clustering analysis on the data points.
    Wrapper for the basic clustering analysis.

    :param embedded_sequences: data points
    :param plot_path: path to sav{plot_path
    :param logger or None: logger for logging messages
    :param tag: tag for the output files
    """

    def run_tsne(data):
        print("[Clustering] Running TSNE")
        tsne = manifold.TSNE(
            n_components=2,
            perplexity=5.0,
            init="random",
            method="exact",
            max_iter=1000,
            random_state=42,
            verbose=0,
        )
        return tsne.fit_transform(data)

    def run_umap(data):
        print("[Clustering] Running UMAP")
        reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)

        return reducer.fit_transform(data)

    def clustering(data):
        print("[Clustering] Running KMeans")
        kmeans = KMeans(n_clusters=2, max_iter=100, n_init=5, random_state=42)
        kmeans.fit(data)

        return kmeans, kmeans.labels_

    if logger:
        logger.info("[Clustering] Starting clustering analysis TSNE")
    tsne_fit = run_tsne(embedded_sequences)

    if logger:
        logger.info("[Clustering] Starting clustering analysis UMAP")
    umap_fit = run_umap(embedded_sequences)

    print("[Clustering] Running KMeans for TSNE")
    clustering(tsne_fit)
    print("[Clustering] Running KMeans for UMAP")
    clustering(umap_fit)

    print("[Clustering] Plotting TSNE")
    plot_tsne(tsne_fit, plot_path=f"{plot_path}{tag}_tsne_plot")

    print("[Clustering] Plotting UMAP")
    plot_umap(umap_fit, plot_path=f"{plot_path}{tag}_umap_plot")


def encode_peptides(sequence_file,
                    device,
                    plot_path: str,
                    sequence_max_length: int,
                    logger: logging.Logger or None = None,
                    scaler=None,
                    label_0_cluster_data: int = 500,
                    label_1_cluster_data: int = 500,
                    tokenizer_and_model: [BertTokenizer, BertModel] or None = None,
                    batch_size: int = 32
                    ):
    if tokenizer_and_model is None:
        # Load the pre-trained model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False,
                                                  clean_up_tokenization_spaces=True)
        model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")


    else:
        tokenizer, model = tokenizer_and_model

    model.eval()

    df = pd.DataFrame({'sequence': sequence_file.peptides, 'label': sequence_file.labels})
    # Shuffle the data to ensure labels are mixed
    df = df.sample(frac=1).reset_index(drop=False).rename(columns={'index': 'orig_idx'})

    # ------------
    # Change number of datapoints used for clustering here.
    df = reduce_data_points_for_clustering(df,
                                           label_0_data=label_0_cluster_data,
                                           label_1_data=label_1_cluster_data,
                                           plot_path=plot_path)
    # ------------
    seq_lens = [len(''.join(seq.split(' '))) for seq in df['sequence']]

    # Generate embeddings in batches
    progress = 0
    embeddings = []
    for i in range(0, len(df['sequence']), batch_size):
        batch_peptides = df['sequence'][i:i + batch_size].to_list()
        enc = tokenizer(batch_peptides, return_tensors="pt", padding='max_length', truncation=True,
                        max_length=sequence_max_length)

        enc = {key: value.to(device) for key, value in enc.items()}

        outputs = model(**enc, return_pooler_output=True)

        embeddings.append(outputs.cpu().detach().numpy())
        progress += len(batch_peptides)
        if logger:
            logger.info(f"[Embedding] Progress: {progress}/{len(df['sequence'])}")
        else:
            print(f"[Embedding] Progress: {progress}/{len(df['sequence'])}")

    # Concatenate all batch embeddings
    embeddings = np.vstack(embeddings)
    if scaler is None:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    else:
        embeddings = scaler.transform(embeddings)

    return embeddings, df['label'].to_list(), seq_lens, scaler


def cluster_model_embedding(file_path,
                            batch_size: int,
                            plot_path: str,
                            device,
                            sequence_max_length: int,
                            data_tag: str = "test_run",
                            label_0_cluster_data: int = 500,
                            label_1_cluster_data: int = 500,
                            logger: logging.Logger or None = None,
                            scaler=None,
                            tokenizer_and_model: [BertTokenizer, BertModel] or None = None):
    """
    Standalone Wrapper for clustering analysis if you run this file directly.
    Thought for if you already have a model, and you want to further analyze the data.
    A basic clustering will be done in the fine tune script.

    :param file_path: Path to the csv file containing the sequences and labels
    :param batch_size: Batch size for encoding the sequences
    :param plot_path: Path to save the plots
    :param device: Device to run the model on
    :param sequence_max_length: Maximum length of the sequences
    :param data_tag: Tag for the output files
    :param label_0_cluster_data: Amount of data points for label 0
    :param label_1_cluster_data: Amount of data points for label 1
    :param tokenizer_and_model: Tuple containing the tokenizer and model
    """

    # Generating a tag for output files to be unique

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    embedding, labels, seq_lens, scaler = encode_peptides(sequence_file=file_path,
                                                          batch_size=batch_size,
                                                          tokenizer_and_model=tokenizer_and_model,
                                                          device=device,
                                                          sequence_max_length=sequence_max_length,
                                                          label_0_cluster_data=label_0_cluster_data,
                                                          label_1_cluster_data=label_1_cluster_data,
                                                          logger=logger,
                                                          scaler=scaler,
                                                          plot_path=plot_path)

    perform_clustering(embedded_sequences=embedding,
                       logger=logger,
                       tag=data_tag,
                       plot_path=plot_path)

    return scaler


def reduce_data_points_for_clustering(df: pd.DataFrame,
                                      plot_path: str,
                                      label_0_data: int = 500,
                                      label_1_data: int = 500) -> pd.DataFrame:
    """
    This function is meant to reduce to points that should get clustered to reduce a way to cluttered plot
    it should get roundabout egal 0 and 1 data points, but id would be good maybe to alter the ratio between them for
    further analysis.

    -> I should add an optional drop duplicate in case function is used outside of main logic

    :param df: pd.DataFrame: The DataFrame that should get reduced
    :param label_0_data: int: The amount of data points for label 0
    :param label_1_data: int: The amount of data points for label 1

    :return: pd.DataFrame: The reduced DataFrame
    """
    if len(df.index) < 1000:
        return df
    # ensure the new df's are shuffled
    df_0 = df[df['label'] == 0].sample(frac=1)
    df_1 = df[df['label'] == 1].sample(frac=1)

    result_df = pd.concat([df_0[:label_0_data], df_1[:label_1_data]]).sample(frac=1)
    result_df.to_csv(f"{plot_path}data_used_for_clustering.csv", sep=';',
                     index=False)

    return result_df


if __name__ == "__main__":
    ...
