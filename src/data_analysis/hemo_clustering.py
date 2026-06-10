import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel


def _scatter_embedding(ax, fit, labels):
    """
    Scatter a 2D embedding onto `ax`, split into positive/negative classes by `labels`.

    :param ax: matplotlib Axes to draw on
    :param fit: array (n, 2) of 2D coordinates
    :param labels: class labels of {0, 1}
    """
    labels_arr = np.asarray(labels)
    pos = fit[labels_arr == 1]
    neg = fit[labels_arr == 0]
    ax.scatter(pos[:, 0], pos[:, 1], c="lawngreen", alpha=0.7,
               label=f"Positive: {int(np.sum(labels_arr == 1))}", s=12)
    ax.scatter(neg[:, 0], neg[:, 1], c="darkgrey", alpha=0.7,
               label=f"Negative: {int(np.sum(labels_arr == 0))}", s=12)
    ax.legend(loc="best")

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")


def plot_pca(pca, pca_fit, labels, plot_path=""):
    """
    Plot PCA analysis of the data points.

    :param pca: PCA object (scikit-learn)
    :param pca_fit: PCA transformed data (n_samples, n_components)
    :param labels: class labels of {0, 1}
    :param plot_path: path to save the plot (without extension)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    _scatter_embedding(ax1, pca_fit, labels)
    ax1.set_title("PCA on Sequence Embeddings")

    # right side: explained variance
    ax2.set_title("PCA Explained Variance")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_xlabel("Principal Components")
    ncomps = min(10, len(pca.explained_variance_ratio_))
    ax2.plot(np.arange(ncomps) + 1, pca.explained_variance_ratio_[:ncomps],
             "o-", linewidth=2)

    plt.suptitle("PCA Analysis of Sequences")
    _save_or_show(fig, plot_path)


def plot_tsne(tsne_fit, labels, plot_path="", title="t-SNE on Sequence Embeddings"):
    """
    Plot a t-SNE embedding of the data points.

    :param tsne_fit: array (n, 2) of t-SNE coordinates
    :param labels: class labels of {0, 1}
    :param plot_path: path to save the plot (without extension); saves as PDF if set
    :param title: plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    _scatter_embedding(ax, tsne_fit, labels)
    ax.set_title(title)
    _save_or_show(fig, plot_path)


def plot_umap(data, labels, plot_path=""):
    """
    Run UMAP on `data` and plot the resulting 2D embedding.

    :param data: high-dimensional data points
    :param labels: class labels of {0, 1}
    :param plot_path: path to save the plot (without extension)
    :return: the 2D UMAP embedding
    """
    reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
    umapped = reducer.fit_transform(data)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    _scatter_embedding(ax, umapped, labels)
    ax.set_title("UMAP on Sequence Embeddings")
    _save_or_show(fig, plot_path)

    return umapped


def _save_or_show(fig, plot_path: str):
    if plot_path:
        fig.savefig(f"{plot_path}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def perform_clustering(embedded_sequences,
                        plot_path: str,
                        labels,
                        logger: Optional[logging.Logger] = None,
                        tag: str = ""):
    """
    Run TSNE and UMAP on `embedded_sequences` and save the resulting plots.

    :param embedded_sequences: data points
    :param plot_path: path prefix to save the plots under
    :param labels: class labels of {0, 1} for color-coding the plots
    :param logger: optional logger for progress messages
    :param tag: tag appended to the output file names
    """

    def _log(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    def run_tsne(data):
        _log("[Clustering] Running TSNE")
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
        _log("[Clustering] Running UMAP")
        reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
        return reducer.fit_transform(data)

    tsne_fit = run_tsne(embedded_sequences)
    #umap_fit = run_umap(embedded_sequences)

    _log("[Clustering] Plotting TSNE")
    plot_tsne(tsne_fit, labels=labels, plot_path=f"{plot_path}{tag}_tsne_plot")

    # _log("[Clustering] Plotting UMAP")
    # plot_umap(umap_fit, labels=labels, plot_path=f"{plot_path}{tag}_umap_plot")


def encode_peptides(sequence_file,
                     device,
                     plot_path: str,
                     sequence_max_length: int,
                     logger: Optional[logging.Logger] = None,
                     scaler=None,
                     label_0_cluster_data: int = 500,
                     label_1_cluster_data: int = 500,
                     tokenizer_and_model: Optional[Tuple[BertTokenizer, BertModel]] = None,
                     batch_size: int = 32
                     ):
    if tokenizer_and_model is None:
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False,
                                                    clean_up_tokenization_spaces=True)
        model = BertModel.from_pretrained("Rostlab/prot_bert_bfd").to(device)
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
    with torch.no_grad():
        for i in range(0, len(df['sequence']), batch_size):
            batch_peptides = df['sequence'][i:i + batch_size].to_list()
            enc = tokenizer(batch_peptides, return_tensors="pt", padding='max_length', truncation=True,
                            max_length=sequence_max_length)
            enc = {key: value.to(device) for key, value in enc.items()}

            outputs = model(**enc, return_pooler_output=True)

            embeddings.append(outputs.cpu().numpy())
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
                             logger: Optional[logging.Logger] = None,
                             scaler=None,
                             tokenizer_and_model: Optional[Tuple[BertTokenizer, BertModel]] = None):
    """
    Standalone wrapper for embedding + clustering analysis, e.g. for further
    analysis of an already-trained model. A basic clustering is also run as
    part of the fine-tuning script.

    :param file_path: dataset object holding `.peptides` and `.labels`
    :param batch_size: batch size for encoding the sequences
    :param plot_path: path prefix to save the plots under
    :param device: device to run the model on
    :param sequence_max_length: maximum length of the sequences
    :param data_tag: tag appended to the output file names
    :param label_0_cluster_data: number of data points to sample for label 0
    :param label_1_cluster_data: number of data points to sample for label 1
    :param logger: optional logger for progress messages
    :param scaler: pre-fitted StandardScaler (optional, fit a new one if None)
    :param tokenizer_and_model: pre-loaded (tokenizer, model) pair (optional)
    :return: the (fitted) scaler used to normalize the embeddings
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    Path(plot_path).mkdir(parents=True, exist_ok=True)

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
                        labels=labels,
                        logger=logger,
                        tag=data_tag,
                        plot_path=plot_path)

    return scaler


def reduce_data_points_for_clustering(df: pd.DataFrame,
                                       plot_path: str,
                                       label_0_data: int = 500,
                                       label_1_data: int = 500) -> pd.DataFrame:
    """
    Subsample `df` to roughly `label_0_data` / `label_1_data` rows per class,
    to keep clustering plots from becoming too cluttered.

    :param df: the DataFrame to subsample
    :param plot_path: path prefix; the subsampled rows are also saved here as a CSV
    :param label_0_data: number of rows to keep for label 0
    :param label_1_data: number of rows to keep for label 1
    :return: the subsampled DataFrame
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
