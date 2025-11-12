import logging
import torch
import umap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import manifold, metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel

"""
This code is an to my usecase adapted version of the code from Lukas Bayerle 
of the workgroup i work in for Prof. Dr. Franz Cemic.
"""


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
        palette={0: "darkgrey", 1: "lawngreen"},
    )


# Positive 0 = helix, Negative 1 = beta
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
    # return np.array(positive), np.array(negative)
    return np.array(positive).reshape(-1, 2), np.array(negative).reshape(-1, 2)


def plot_pca22(pca, pca_fit, labels, plot_path=""):
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
        label=f"Positive: {labels.count(1)}",
        s=10
    )
    ax1.scatter(
        x=neg[:, 0],
        y=neg[:, 1],
        c="darkgrey",
        alpha=0.7,
        label=f"Negative: {labels.count(0)}",
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
    ax1.set_title("PCA auf Sequenz-Embeddings")
    ax1.set_xlabel("Komponente 1")
    ax1.set_ylabel("Komponente 2")
    ax2.set_title("Erklärte Varianz der PCA")
    ax2.set_ylabel("Anteil erklärte Varianz")
    ax2.set_xlabel("Hauptkomponenten")
    plt.suptitle("PCA-Analyse der Sequenzen")
    ax2.plot(np.arange(10) + 1, pca.explained_variance_ratio_, "o-",
             linewidth=2)  # needs to be the same as pca comps TODO move to param
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


def plot_pca(pca, pca_fit, labels=None, lengths=None, plot_path=""):
    """
    Plot PCA analysis of the data points.

    :param pca: PCA object (scikit-learn)
    :param pca_fit: PCA transformed data (n_samples, n_components)
    :param labels: class labels (optional)
    :param lengths: sequence lengths (optional) -> wenn gesetzt, wird nach Länge eingefärbt
    :param plot_path: path to save the plot (ohne Endung)
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
        cb.set_label("Sequenzlänge (AAs)")
        # optional: Labelzahlen in Legende
        if labels is not None:
            ax1.legend([f"Positive: {int(np.sum(np.asarray(labels)==1))}",
                        f"Negative: {int(np.sum(np.asarray(labels)==0))}"],
                       frameon=True, loc="best")
    else:
        assert labels is not None, "Für Klassenplot 'labels' übergeben oder 'lengths' setzen."
        labels_arr = np.asarray(labels)
        pos = pca_fit[labels_arr == 1]
        neg = pca_fit[labels_arr == 0]
        ax1.scatter(pos[:, 0], pos[:, 1], c="lawngreen", alpha=0.7,
                    label=f"Positive: {int(np.sum(labels_arr==1))}", s=12)
        ax1.scatter(neg[:, 0], neg[:, 1], c="darkgrey", alpha=0.7,
                    label=f"Negative: {int(np.sum(labels_arr==0))}", s=12)
        ax1.legend(loc="best")

    ax1.set_title("PCA auf Sequenz-Embeddings")
    ax1.set_xlabel("Komponente 1")
    ax1.set_ylabel("Komponente 2")

    # rechte Seite: erklärte Varianz
    ax2.set_title("Erklärte Varianz der PCA")
    ax2.set_ylabel("Anteil erklärte Varianz")
    ax2.set_xlabel("Hauptkomponenten")
    # zeige z. B. die ersten 10 Komponenten
    ncomps = min(10, len(pca.explained_variance_ratio_))
    ax2.plot(np.arange(ncomps) + 1, pca.explained_variance_ratio_[:ncomps],
             "o-", linewidth=2)

    plt.suptitle("PCA-Analyse der Sequenzen")

    if plot_path:
        plt.savefig(f"{plot_path}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_tsne(tsne_fit, labels=None, lengths=None, plot_path="", title="TSNE auf Sequenz-Embeddings"):
    """
    tsne_fit : array (n,2) – t-SNE-Koordinaten
    labels   : Liste/Array aus {0,1} (optional, für Klassen-Plot)
    lengths  : Liste/Array Sequenzlängen (optional, für Längen-Plot)
               -> Wenn gesetzt, wird nach Länge eingefärbt und 'labels' nur für Legendenzahlen genutzt.
    plot_path: ohne Endung; speichert als PDF wenn gesetzt
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    if lengths is not None:
        lengths = np.asarray(lengths, dtype=int)
        sc = ax.scatter(
            tsne_fit[:, 0], tsne_fit[:, 1],
            c=lengths, s=12, alpha=0.85, cmap="viridis", edgecolors="none"
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Sequenzlänge (AAs)")
        # optional: zusätzliche Info in der Legende
        if labels is not None:
            ax.legend([f"Positive: {int(np.sum(np.asarray(labels) == 1))}",
                       f"Negative: {int(np.sum(np.asarray(labels) == 0))}"],
                      frameon=True, loc="best")
    else:
        # klassischer Klassen-Plot
        assert labels is not None, "Für Klassen-Plot 'labels' übergeben oder 'lengths' setzen."
        labels_arr = np.asarray(labels)
        pos = tsne_fit[labels_arr == 1]
        neg = tsne_fit[labels_arr == 0]
        ax.scatter(pos[:, 0], pos[:, 1], c="lawngreen", alpha=0.7,
                   label=f"Positive: {int(np.sum(labels_arr == 1))}", s=12)
        ax.scatter(neg[:, 0], neg[:, 1], c="darkgrey", alpha=0.7,
                   label=f"Negative: {int(np.sum(labels_arr == 0))}", s=12)
        ax.legend(loc="best")

    ax.set_title(title)
    ax.set_xlabel("Komponente 1")
    ax.set_ylabel("Komponente 2")

    if plot_path:
        plt.savefig(f"{plot_path}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_tsne22(tsne_fit, labels, plot_path=""):
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
        label=f"Positive: {labels.count(1)}",
        s=10
    )
    ax.scatter(
        x=neg[:, 0],
        y=neg[:, 1],
        c="darkgrey",
        alpha=0.7,
        label=f"Negative: {labels.count(0)}",
        s=10
    )

    # plot_density(
    #     x=tsne_fit[:, 0],
    #     y=tsne_fit[:, 1],
    #     ax=ax,
    #     labels=labels,
    # )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_title("TSNE auf Sequenz-Embeddings")
    ax.set_xlabel("Komponente 1")
    ax.set_ylabel("Komponente 2")

    if plot_path != "":
        plt.savefig(f"{plot_path}.pdf")
        plt.clf()
    else:
        plt.show()


def plot_umap22(data, labels, plot_path=""):
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
        label=f"Positive: {labels.count(1)}",
        s=10
    )
    ax.scatter(
        x=neg[:, 0],
        y=neg[:, 1],
        c="darkgrey",
        alpha=0.7,
        label=f"Negative: {labels.count(0)}",
        s=10
    )


    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_title("UMAP auf Sequenz-Embeddings")
    ax.set_xlabel("Komponente 1")
    ax.set_ylabel("Komponente 2")

    if plot_path != "":
        plt.savefig(f"{plot_path}.pdf")
        plt.clf()
    else:
        plt.show()
    return umapped


def plot_umap(data, labels=None, lengths=None, plot_path=""):
    """
    Plot UMAP analysis of the data points.

    :param data: high-dimensional data points
    :param labels: class labels of the data points (optional)
    :param lengths: sequence lengths (optional) -> wenn gesetzt, wird nach Länge eingefärbt
    :param plot_path: path to save the plot (ohne Endung)
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
        cb.set_label("Sequenzlänge (AAs)")

        # optional Legende mit Labelzahlen
        if labels is not None:
            ax.legend([f"Positive: {int(np.sum(np.asarray(labels)==1))}",
                       f"Negative: {int(np.sum(np.asarray(labels)==0))}"],
                      frameon=True, loc="best")

    else:
        # klassischer Klassenplot wie bisher
        assert labels is not None, "Für Klassenplot 'labels' übergeben oder 'lengths' setzen."
        labels_arr = np.asarray(labels)
        pos = umapped[labels_arr == 1]
        neg = umapped[labels_arr == 0]
        ax.scatter(pos[:, 0], pos[:, 1], c="lawngreen", alpha=0.7,
                   label=f"Positive: {int(np.sum(labels_arr==1))}", s=12)
        ax.scatter(neg[:, 0], neg[:, 1], c="darkgrey", alpha=0.7,
                   label=f"Negative: {int(np.sum(labels_arr==0))}", s=12)
        ax.legend(loc="best")

    ax.set_title("UMAP auf Sequenz-Embeddings")
    ax.set_xlabel("Komponente 1")
    ax.set_ylabel("Komponente 2")

    if plot_path:
        plt.savefig(f"{plot_path}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()

    return umapped


def perform_clustering(embedded_sequences,
                       sequence_labels,
                       plot_path: str,
                       seq_lens,
                       logger: logging.Logger or None = None,
                       tag: str = ""):
    """
    Perform clustering analysis on the data points.
    Wrapper for the basic clustering analysis.

    :param embedded_sequences: data points
    :param sequence_labels: labels of the data points. Needs to be index paired with the data points
    :param plot_path: path to sav{plot_path
    :param logger: logger for logging messages
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
            print(
                f"Skipping silhouette score calculation for {cluster_tag} clustering due to insufficient unique labels.")
            print(f"Unique labels: {unique_labels}")
            silhouette_score = 'None'

        else:
            silhouette_score = metrics.silhouette_score(
                data, cluster_labels, random_state=42
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

    def run_pca(data, comps=10):  # need to be the same as in line 93
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
            max_iter=1000,
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

    # if logger:
    #     logger.info("[Clustering] Starting clustering analysis PCA")
    # pca, pca_fit = run_pca(embedded_sequences)

    if logger:
        logger.info("[Clustering] Starting clustering analysis TSNE")
    tsne_fit = run_tsne(embedded_sequences)

    if logger:
        logger.info("[Clustering] Starting clustering analysis UMAP")
    umap_fit = run_umap(embedded_sequences)

    # print("[Clustering] Running KMeans for PCA")
    # pca_kmeans, pca_kmeans_labels = clustering(pca_fit, sequence_labels)

    print("[Clustering] Running KMeans for TSNE")
    tsne_kmeans, tnse_kmeans_labels = clustering(tsne_fit, sequence_labels)
    print("[Clustering] Running KMeans for UMAP")
    umap_kmeans, umap_kmeans_labels = clustering(umap_fit, sequence_labels)
    #
    # plot_pca(pca, pca_fit, sequence_labels, plot_path=f"{plot_path}{tag}_pca_plot")


    print("[Clustering] Plotting TSNE")
    plot_tsne(tsne_fit, sequence_labels, plot_path=f"{plot_path}{tag}_tsne_plot")

    print("[Clustering] Plotting UMAP")
    plot_umap(umap_fit, sequence_labels, plot_path=f"{plot_path}{tag}_umap_plot")



def encode_peptides(sequence_file,
                    device,
                    plot_path: str,
                    add_features: bool,
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
    # Prepare peptides
    # peptides_prepared = [' '.join(pep) for pep in df['sequence'].to_list()]

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
                            add_features: bool,
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
    :param add_features: Whether to add features to the sequence data
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
                                                          plot_path=plot_path,
                                                          add_features=add_features)


    perform_clustering(embedded_sequences=embedding,
                       sequence_labels=labels,
                       logger=logger,
                       tag=data_tag,
                       seq_lens=seq_lens,
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
                     index=False)  # TODO maybe make this optional?

    return result_df


if __name__ == "__main__":
    ...
