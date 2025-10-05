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
    #return np.array(positive), np.array(negative)
    return np.array(positive).reshape(-1, 2), np.array(negative).reshape(-1, 2)


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
    #     x=umapped[:, 0],
    #     y=umapped[:, 1],
    #     ax=ax,
    #     labels=labels,
    # )

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


def perform_clustering(embedded_sequences,
                       sequence_labels,
                       plot_path: str,
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
            silhouette_score = None
            return
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

    # # print("[Clustering] Running KMeans for PCA")
    # pca_kmeans, pca_kmeans_labels = clustering(pca_fit, sequence_labels)

    print("[Clustering] Running KMeans for TSNE")
    tsne_kmeans, tnse_kmeans_labels = clustering(tsne_fit, sequence_labels)
    print("[Clustering] Running KMeans for UMAP")
    umap_kmeans, umap_kmeans_labels = clustering(umap_fit, sequence_labels)
    #
    # plot_pca(pca, pca_fit, sequence_labels, plot_path=f"{plot_path}/{tag}_pca_plot")

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
                    scaler = None,
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
    df = df.sample(frac=1).reset_index(drop=False).rename(columns={'index':'orig_idx'})

    # ------------
    # Change number of datapoints used for clustering here.
    df = reduce_data_points_for_clustering(df,
                                           label_0_data=label_0_cluster_data,
                                           label_1_data=label_1_cluster_data,
                                           plot_path=plot_path)
    # ------------

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
    embeddings_to_return = [('seq_embedding',embeddings)]
    if add_features:
        seq_feature_embedding = np.hstack([embeddings, sequence_file.features[df['orig_idx'].values]])

        embeddings_to_return.append(('seq_feature_embedding' , seq_feature_embedding))
        embeddings_to_return.append(('feature_vector', sequence_file.features[df['orig_idx'].values]))



    return embeddings_to_return, df['label'].to_list(), scaler



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
                            scaler = None,
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

    embedding, labels, scaler = encode_peptides(sequence_file=file_path,
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

    for emb in embedding:
        perform_clustering(embedded_sequences=emb[1],
                           sequence_labels=labels,
                           logger=logger,
                           tag=data_tag + emb[0],
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