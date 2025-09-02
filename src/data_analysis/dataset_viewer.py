from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

"""
This script is thought to be used to visualize the dataset used for training the model.
It contains functions to plot the distribution of labels and the frequency of amino acids in the sequences.

Add new functions here.

TODO:
- Add more plots to visualize the dataset
"""



def look_into_datasets(file_paths: list[tuple[str, str]],
                       k_mer_inspection_file: str or None,
                       show_top_kmers: int = 20 or None,
                       out_path: str or None = None):
    """
    Function to look into the datasets and print the first 5 rows of each dataset.

    Args:
        file_paths (list[tuple[str, str]]): A list of tuples containing the file path and the separator.
        k_mer_inspection_file (str or None): The path to the k-mer inspection file. If None, no k-mer inspection will be done.
        show_top_kmers (int or None): The number of top k-mers to show. If None, all k-mers will be
        out_path (str or None): The path to save the plots. If None, the plots will be shown.
    """
    print(f"\033[31m\n----------------------------Analysing Train Files-------------------------------------\033[0m")
    for file_path in file_paths:
        df = pd.read_csv(file_path[0], sep=file_path[1])

        print(f"\033[31m-----------------Generating Info for {file_path[0]}:----------------------------------------\033[0m")
        print(f"\033[31m\n\nNumber of sequences: {df.shape[0]}\033[0m")
        print(f"\033[31mNumber of unique sequences: {df['sequence'].nunique()}\n\033[0m")

        plot_binary_label_distribution(df, plot_path=out_path)
        plot_concentration_to_percent(df, plot_path=out_path)
        sequences_0 = df[df["label"] == 0]["sequence"].tolist()
        sequences_1 = df[df["label"] == 1]["sequence"].tolist()
        amino_acid_frequency_comparison(sequences_0, sequences_1, plot_path=out_path)
        sequence_length_correlation_plot(df, plot_path=out_path)

    print(f"\033[31m\n-------------------------Analysing High Difference Sequences------------------------------\033[0m")
    if k_mer_inspection_file is not None:
        for k_mer_file in k_mer_inspection_file:
            df = pd.read_csv(k_mer_file[0], sep=k_mer_file[1])

            print(f"\033[31m-------------------Generating Info for {k_mer_file[0]}:-------------------------------\033[0m")
            print(f"\033[31m\n\nNumber of sequences: {df.shape[0]}\033[0m")
            print(f"\033[31mPlotting Ground Truth Label distribution for k_mer file\n\033[0m")
            plot_binary_label_distribution(df, plot_path=out_path)
            plot_concentration_to_percent(df, plot_path=out_path)

            print(f"\033[31m\nDifferences in hemolytic concentrations inside one sequence\033[0m")
            print(
                f"\033[31m\n[WARNING]Extrem Values are set to 0! Its expected that data feeded here only show data OVER a certrain threshold set by train_data creation!!!\033[0m")
            check_value_distance_inside_one_sequence(df=df,
                                                     plot_path=out_path)

            print(f"\033[31m\nCheck for k_mer content\033[0m")
            df = get_k_mer_overview(df=df,
                               plot_path=out_path,
                               show_top_kmers=show_top_kmers)
            df = df.drop(columns=['k_mers'])
            df.to_csv(f"../../data/data_for_data_viewer/{k_mer_file[0].split('/')[-1].strip('.csv')}_k_mer_overview.csv", sep=';', index=False)

def plot_binary_label_distribution(df, plot_path):
    """
    Plots the distribution of binary labels in the dataset.

    Args:
        df (pd.DataFrame): The dataframe containing the dataset.
        plot_path (str): The path to save the plot. If empty, the plot will be shown.
    """
    label_counts = df["label"].value_counts()
    plt.bar(label_counts.index, label_counts.values / df.shape[0] * 100)
    plt.xlabel("Label")
    plt.ylabel("Häufigkeit [%]")
    plt.title("Label Verteilung der Whitelab Daten")
    plt.xticks([0, 1], ["0", "1"])
    if plot_path is None:
        plt.show()
    else:
        plt.savefig(f"{plot_path}_label_distribution.png")
        plt.clf()


def amino_acid_frequency_comparison(
        sequences_0, sequences_1, plot_path=""
):
    """
    Function from lucas bayerle snippet codes
    """

    def _tokenize(sequences) -> dict:
        counts = Counter()
        for seq in sequences:
            counts += Counter(seq)
        return counts

    counts_0 = _tokenize(sequences_0)
    counts_1 = _tokenize(sequences_1)

    tick_labels = sorted(set(counts_0.keys()))  # | set(counts_1.keys()))
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

    if plot_path is None:
        plt.show()
    else:
        plt.savefig(f"{plot_path}_amino_acid_frequency.png")
        plt.clf()





def sequence_length_correlation_plot(df: pd.DataFrame,
                                     plot_path: str or None = None):


    # Example: df contains 'Sequence' and 'Label'
    df['SeqLength'] = df['sequence'].apply(len)

    # Optional: Basic correlation check
    correlation = df['SeqLength'].corr(df['label'])
    print(f"Correlation between sequence length and label: {correlation:.3f}")

    # Plot using seaborn (boxplot + swarm for visibility)
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='label', y='SeqLength', data=df, showfliers=False)
    sns.stripplot(x='label', y='SeqLength', data=df, color='black', alpha=0.5, jitter=True)

    plt.title('Sequence Length vs Hemolytic Label')
    plt.xlabel('Label (0 = Non-Hemolytic, 1 = Hemolytic)')
    plt.ylabel('Peptide Sequence Length')
    plt.xticks([0, 1], ['Non-Hemolytic', 'Hemolytic'])
    plt.grid(True, axis='y')
    plt.tight_layout()
    if plot_path is None:
        plt.show()
    else:
        plt.savefig(f"{plot_path}_sequence_length_correlation.png")
        plt.clf()


def check_ambiguous_labeled_sequences():
    """
    Check how many sequences are labeled ambiguously.
    """

    ...


def check_value_distance_inside_one_sequence(df: pd.DataFrame or str,
                                             plot_path: str or None,
                                             min_concentration_difference: int = 50,
                                             filter_extrem_threshold: int = 400):
    """
    Check how much the measured hemolytic values differ inside one sequence.
    """
    if df.__class__ == str:
        df = pd.read_csv(df, sep=';')
    print(df.shape)
    difference_list = []
    high_dif_sequence = []
    for sequence, sequence_df in df.groupby('sequence'):
        if len(sequence_df) != 0:
            difference = sequence_df['hemo_concentration'].max() - sequence_df['hemo_concentration'].min()
            if difference > min_concentration_difference:
                high_dif_sequence.append(sequence)
                difference_list.append(

                     difference if difference < filter_extrem_threshold else 0
                )
    sorted_differences = sorted(difference_list, reverse=True)

    plt.figure(figsize=(8, 5))
    plt.hist(sorted_differences, bins=50)
    plt.title('Distribution of Hemolytic Value Differences')
    plt.xlabel('Difference in Hemolytic Value')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y')
    plt.tight_layout()
    if plot_path is None:
        plt.show()
    else:
        plt.savefig(f"{plot_path}")


    return high_dif_sequence



    # print(difference_list)



def get_k_mer_overview(df: str or pd.DataFrame,
                       plot_path: str,
                       show_top_kmers: int = 20 or None):
    """
    Get an overview of the k-mers in the dataset.
    """

    def get_kmers(sequence, k):
        return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

    if df.__class__ == str:
        df = pd.read_csv(df, sep=';')
    df['k_mers'] = df['sequence'].apply(lambda x: get_kmers(x, 3))
    k_mer_list = df['k_mers'].tolist()
    ground_truth_label = df['label'].tolist()
    _overall_kmer_abundancy(k_mer_list=k_mer_list, plot_path=plot_path, show_top_kmers=show_top_kmers)

    cluster_label = _cluster_kmers_per_sequence(k_mer_list=k_mer_list, plot_path=plot_path, show_top_kmers=show_top_kmers, ground_truth_label=ground_truth_label)

    # add cluster label to input datraframe as new column
    df['cluster_label'] = cluster_label
    return df

def _cluster_kmers_per_sequence(k_mer_list, plot_path, show_top_kmers, ground_truth_label):
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.manifold import TSNE
    # prepare for cluster analysis
    kmer_frequency_per_sequence = [Counter(kmers) for kmers in k_mer_list]
    kmer_vector = DictVectorizer(sparse=False)
    x = kmer_vector.fit_transform(kmer_frequency_per_sequence)

    # prepare data for heatmap
    heatmap_count = _count_kmers(k_mer_list)
    top_kmers = set([k for k, _ in heatmap_count.most_common(show_top_kmers)])

    n_cluster = 5
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    cluster_labels = kmeans.fit_predict(x)

    print(f"\033[31m\nLabel Enrichment per Cluster: (pd.crosstab)\033[0m")
    ct_norm = pd.crosstab(ground_truth_label, cluster_labels, rownames=['ground_truth_label'], colnames=['cluster_label'], normalize='columns')
    print(f"{ct_norm}")
    sns.heatmap(ct_norm, cmap='viridis', cbar=True, annot=False)
    plt.title('Cluster Label Enrichment')
    plt.ylabel('Ground Truth Label')
    plt.xlabel('Cluster Label')
    plt.tight_layout()
    if plot_path is None:
        plt.show()
    else:
        plt.savefig(f"{plot_path}_k_mer_label_clustering_enrichment.png")

    generate_heatmap(k_mer_list, cluster_labels, top_kmers, plot_path)

    x_tsne = TSNE(n_components=2, random_state=42).fit_transform(x)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=cluster_labels, cmap='tab10', s=80)
    plt.title('K-mer t-SNE Clustering')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend(*scatter.legend_elements(), title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    if plot_path is None:
        plt.show()
    else:
        plt.savefig(f"{plot_path}_k_mer_clustering_cluster_label.png")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=ground_truth_label, cmap='tab10', s=80)
    plt.title('K-mer t-SNE Clustering')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend(*scatter.legend_elements(), title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    if plot_path is None:
        plt.show()
    else:
        plt.savefig(f"{plot_path}_k_mer_clustering_ground_label.png")
    return cluster_labels

def generate_heatmap(k_mer_list, labels, top_kmers, plot_path, inspect_heatmap_file: bool = False):

    cluster_kmers_heatmap = defaultdict(list)
    for label, kmer in zip(labels, k_mer_list):
        cluster_kmers_heatmap[label].extend(kmer)


    heatmap_kmer_count = {}
    for label, kmers in cluster_kmers_heatmap.items():
        filtered = [k for k in kmers if k in top_kmers]
        heatmap_kmer_count[label] = Counter(filtered)

    heatmap_df = pd.DataFrame(heatmap_kmer_count).fillna(0).astype(int)
    heatmap_df = heatmap_df.loc[heatmap_df.sum(axis=1) > 0]
    heatmap_df["total"] = heatmap_df.sum(axis=1)
    heatmap_df = heatmap_df.sort_values(by="total", ascending=False).drop(columns=["total"])

    if inspect_heatmap_file:
        heatmap_df.to_csv(f"{plot_path}_k_mer_clustering_heatmap.csv", sep=';', index=False)


    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_df, cmap='viridis', cbar=True, annot=False, fmt='d')
    plt.title('K-mer Clustering Heatmap')
    plt.ylabel('K-mer')
    plt.xlabel('Cluster')

    plt.tight_layout()
    if plot_path is None:
        plt.show()
    else:
        plt.savefig(f"{plot_path}_k_mer_clustering_heatmap.png")


def _overall_kmer_abundancy(k_mer_list, plot_path, show_top_kmers):
    # comprehension for unflatten nested lists
    k_mer_counts = _count_kmers(k_mer_list)

    k_mer_counts = k_mer_counts.most_common(show_top_kmers)

    # generate_heatmap(k_mer_list=k_mer_list,
    #                  labels=k_mer_counts,
    #                  plot_path=plot_path,
    #                  inspect_heatmap_file=True)
    # Split into labels and values
    kmers, counts = zip(*k_mer_counts)
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(kmers, counts, color='skyblue')
    plt.xlabel('k-mer')
    plt.ylabel('Frequency')
    title_value = show_top_kmers if show_top_kmers is not None else "all"
    plt.title(f"Top {title_value} Most Frequent k-mers")
    plt.xticks(rotation=90)
    plt.tight_layout()
    if plot_path is None:
        plt.show()
    else:
        plt.savefig(f"{plot_path}_k_mer_overview.png")
    plt.clf()
    print(k_mer_counts)


def _count_kmers(k_mer_list):
    flatten_k_mer_list = [kmer for k_mer_sub_list in k_mer_list for kmer in k_mer_sub_list]
    k_mer_counts = Counter(flatten_k_mer_list)
    return k_mer_counts


def plot_concentration_to_percent(df: pd.DataFrame,
                                  plot_path: str or None = None):
    """
    Plot the concentration to percent ratio.
    """
    # sort values by concentration for cleaner curve
    df = df[df['hemo_concentration'] < 500]
    df = df.sort_values(by='hemo_percent', ascending=False).reset_index(drop=True)


    data_index = df.index


    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = 'tab:blue'
    ax1.set_xlabel('Datapoint ID')
    ax1.set_ylabel('Concentration', color=color1)
    ax1.scatter(data_index, df['hemo_concentration'], color=color1, label='Concentration')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Percent', color=color2)
    ax2.scatter(data_index, df['hemo_percent'], color=color2, label='Percent')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Concentration and Percent per Datapoint')
    fig.tight_layout()

    if plot_path is None:
        plt.show()
    else:
        plt.savefig(f"{plot_path}_concentration_to_percent.png")
    plt.clf()

if __name__ == "__main__":
    # file_path = "../../data/train_data/happen_style_raw.csv"
    # check_value_distance_inside_one_sequence(df=file_path, plot_path="../../data/train_data/hemolytic_value_differences.png")
    get_k_mer_overview(df="../../data/train_data/happen_style_high_difference_sequences_100.csv",
                       plot_path="../../data/train_data/",
                       show_top_kmers=100)