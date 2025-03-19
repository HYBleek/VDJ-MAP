# -*- coding: utf-8 -*-
"""2D_reference_build.ipynb
"""

import torch
import numpy as np
import pandas as pd
import math
from collections import Counter

# Global device: use GPU if available.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_fasta(filename: str) -> dict:
    """
    Reads a FASTA file and returns a dictionary of sequences.

    Parameters:
        filename: Path to the FASTA file.

    Returns:
        Dictionary where keys are sequence IDs and values are the corresponding sequences.
    """
    sequences = {}
    with open(filename, "r") as file:
        sequence_id = None
        sequence_parts = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence_id is not None:
                    sequences[sequence_id] = "".join(sequence_parts)
                sequence_id = line[1:]  # Remove the ">" character
                sequence_parts = []
            else:
                sequence_parts.append(line)
        if sequence_id is not None:
            sequences[sequence_id] = "".join(sequence_parts)
    return sequences


def count_kmers_in_seq(seq: str, k: int) -> dict:
    """
    Count all k-mers in a given DNA sequence.

    Parameters:
        seq: DNA sequence.
        k: Length of k-mer.

    Returns:
        Dictionary with k-mer counts.
    """
    counts = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def count_kmers_in_dict(dna_dict: dict, k: int, weight: int = 1) -> dict:
    """
    Count weighted k-mers for each sequence in a dictionary.

    Parameters:
        dna_dict: Dictionary of sequences.
        k: Length of k-mer.
        weight: Weight factor for each sequence.

    Returns:
        Dictionary of weighted k-mer counts.
    """
    weighted_counts = {}
    for seq_id, seq in dna_dict.items():
        kmer_counts = count_kmers_in_seq(seq, k)
        for kmer, count in kmer_counts.items():
            weighted_counts[kmer] = weighted_counts.get(kmer, 0) + count * weight
    return weighted_counts


def merge_kmer_counts(dict1_counts: dict, dict2_counts: dict) -> dict:
    """
    Merge two dictionaries containing k-mer counts.

    Parameters:
        dict1_counts: First k-mer count dictionary.
        dict2_counts: Second k-mer count dictionary.

    Returns:
        Merged dictionary with summed counts.
    """
    merged_counts = dict1_counts.copy()
    for kmer, count in dict2_counts.items():
        merged_counts[kmer] = merged_counts.get(kmer, 0) + count
    return merged_counts


def get_kmers(seq: str, k: int) -> list:
    """
    Return a list of all k-mers in a given sequence.

    Parameters:
        seq: DNA sequence.
        k: Length of k-mer.

    Returns:
        List of k-mer strings.
    """
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def compute_tf_idf(dna_dict: dict, k: int, weight_factors: dict = None) -> dict:
    """
    Compute the TF-IDF for k-mers in a dictionary of DNA sequences.

    Parameters:
        dna_dict: Dictionary with sequence IDs as keys and DNA sequences as values.
        k: Length of the k-mer.
        weight_factors: Optional dictionary mapping sequence IDs to weights (default weight is 1).

    Returns:
        Nested dictionary mapping each sequence ID to a dictionary of k-mer TF-IDF scores.
    """
    if weight_factors is None:
        weight_factors = {key: 1 for key in dna_dict.keys()}

    # Count weighted k-mers for each document.
    doc_kmer_counts = {}
    for key, seq in dna_dict.items():
        kmers = get_kmers(seq, k)
        weight = weight_factors.get(key, 1)
        counts = Counter(kmers)
        weighted_counts = {kmer: count * weight for kmer, count in counts.items()}
        doc_kmer_counts[key] = weighted_counts

    # Compute term frequency (TF) for each document.
    doc_tf = {}
    for key, counts in doc_kmer_counts.items():
        total = sum(counts.values())
        tf = {kmer: count / total for kmer, count in counts.items()}
        doc_tf[key] = tf

    # Compute document frequency (DF) for each k-mer.
    df = Counter()
    for counts in doc_kmer_counts.values():
        for kmer in counts.keys():
            df[kmer] += 1

    # Compute inverse document frequency (IDF).
    N = len(dna_dict)
    idf = {kmer: math.log(N / df_val) for kmer, df_val in df.items()}

    # Compute TF-IDF for each document.
    doc_tf_idf = {}
    for key, tf in doc_tf.items():
        tf_idf = {kmer: tf_val * idf[kmer] for kmer, tf_val in tf.items()}
        doc_tf_idf[key] = tf_idf

    return doc_tf_idf


def compute_tf_idf_combined(dna_dict: dict, k: int, weight_factors: dict = None) -> dict:
    """
    Generate a combined k-mer TF-IDF mapping by summing the TF-IDF scores across all documents.

    Parameters:
        dna_dict: Dictionary of DNA sequences.
        k: Length of k-mer.
        weight_factors: Optional dictionary mapping sequence IDs to weights.

    Returns:
        Dictionary mapping each k-mer to its cumulative TF-IDF score.
    """
    doc_tf_idf = compute_tf_idf(dna_dict, k, weight_factors)
    combined_tf_idf = {}
    for tf_idf in doc_tf_idf.values():
        for kmer, score in tf_idf.items():
            combined_tf_idf[kmer] = combined_tf_idf.get(kmer, 0) + score
    return combined_tf_idf


def compute_hamming_distance_matrix(filtered_tf_idf: dict) -> tuple:
    """
    Compute the Hamming distance matrix for k-mers using GPU acceleration if available.

    Parameters:
        filtered_tf_idf: Dictionary with k-mer strings as keys and TF-IDF scores as values.

    Returns:
        Tuple (kmers, hamming_distance_matrix):
          - kmers: List of k-mer strings.
          - hamming_distance_matrix: Numpy array of shape (N, N) with Hamming distances.
    """
    kmers = list(filtered_tf_idf.keys())
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    kmer_ints = [[mapping[base] for base in kmer] for kmer in kmers]
    kmer_tensor = torch.tensor(kmer_ints, device=DEVICE)
    diff = (kmer_tensor.unsqueeze(1) != kmer_tensor.unsqueeze(0)).sum(dim=2)
    hamming_distance_matrix = diff.cpu().numpy()
    return kmers, hamming_distance_matrix


def compute_adjusted_hamming_distance_matrix(filtered_tf_idf: dict, scaling_factor: float = 1.0) -> tuple:
    """
    Compute a modified Hamming distance matrix for k-mers by adjusting distances based on their TF-IDF scores.

    Parameters:
        filtered_tf_idf: Dictionary with k-mer strings as keys and TF-IDF scores as values.
        scaling_factor: Factor to scale the TF-IDF adjustment.

    Returns:
        Tuple (kmers, adjusted_distance_matrix):
          - kmers: List of k-mer strings.
          - adjusted_distance_matrix: Numpy array of adjusted distances.
    """
    kmers = list(filtered_tf_idf.keys())
    tfidf_scores = torch.tensor([filtered_tf_idf[kmer] for kmer in kmers],
                                  dtype=torch.float, device=DEVICE)
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    kmer_ints = [[mapping[base] for base in kmer] for kmer in kmers]
    kmer_tensor = torch.tensor(kmer_ints, dtype=torch.int64, device=DEVICE)
    diff = (kmer_tensor.unsqueeze(1) != kmer_tensor.unsqueeze(0)).sum(dim=2)
    adjustment = (tfidf_scores.unsqueeze(1) + tfidf_scores.unsqueeze(0)) / 2.0
    adjusted = diff.float() - scaling_factor * adjustment
    adjusted = torch.clamp(adjusted, min=0.01)
    return kmers, adjusted.cpu().numpy()


def approximate_mds(distance_matrix: np.ndarray, n_components: int = 2, lr: float = 0.01,
                    n_iter: int = 5000, device: torch.device = None) -> np.ndarray:
    """
    Perform approximate multidimensional scaling (MDS) using gradient descent.

    Parameters:
        distance_matrix: Precomputed distance matrix (numpy array) of shape (N, N).
        n_components: Target dimension for the embedding.
        lr: Learning rate.
        n_iter: Number of iterations.
        device: Device to use (defaults to global DEVICE if None).

    Returns:
        Numpy array of shape (N, n_components) with the embedding coordinates.
    """
    if device is None:
        device = DEVICE

    D = torch.tensor(distance_matrix, dtype=torch.float, device=device)
    N = D.shape[0]
    X = torch.randn(N, n_components, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([X], lr=lr)

    for i in range(n_iter):
        optimizer.zero_grad()
        X_norm = torch.sum(X**2, dim=1, keepdim=True)
        squared_distances = X_norm + X_norm.t() - 2 * torch.matmul(X, X.t())
        squared_distances = torch.clamp(squared_distances, min=0)
        X_dists = torch.sqrt(squared_distances + 1e-8)
        loss = torch.sum((X_dists - D) ** 2)
        loss.backward()
        optimizer.step()
        if i % 200 == 0:
            print(f"Iteration {i}, stress: {loss.item()}")

    return X.detach().cpu().numpy()


def create_storage_table(kmers: list, embedding: np.ndarray, tfidf_dict: dict) -> pd.DataFrame:
    """
    Combine k-mers with their embedding coordinates and TF-IDF scores into a pandas DataFrame.

    Parameters:
        kmers: List of k-mer strings.
        embedding: Numpy array of shape (N, n_components) with embedding coordinates.
        tfidf_dict: Dictionary mapping k-mers to their TF-IDF scores.

    Returns:
        DataFrame with columns: 'kmer', 'tfidf', 'dim1', 'dim2', ...
    """
    n_components = embedding.shape[1]
    # Extract the TF-IDF scores in the order of kmers.
    tfidf_scores = [tfidf_dict[kmer] for kmer in kmers]
    # Create the DataFrame.
    data = np.column_stack((kmers, tfidf_scores, embedding))
    columns = ["kmer", "tfidf"] + [f"dim{i+1}" for i in range(n_components)]
    return pd.DataFrame(data, columns=columns)


if __name__ == "__main__":
    # Read FASTA files.
    region_dict = read_fasta("regions.fa")
    supp_region_dict = read_fasta("supp_regions.fa")
    combined_dna_dict = {**region_dict, **supp_region_dict}

    # Define weight factors: assign a higher weight for sequences from region_dict.
    weight_factors = {**{k: 2 for k in region_dict.keys()},
                      **{k: 1 for k in supp_region_dict.keys()}}

    # Compute combined TF-IDF for k-mers (k=8).
    combined_tf_idf = compute_tf_idf_combined(combined_dna_dict, k=8, weight_factors=weight_factors)

    # Save all k-mers (before filtering) to a CSV file.
    all_kmers_table = pd.DataFrame(list(combined_tf_idf.items()), columns=['kmer', 'tfidf'])
    all_kmers_table.to_csv("all_kmers.csv", index=False)
    print("All k-mers saved as 'all_kmers.csv'")

    # Filter TF-IDF scores.
    filtered_tf_idf = {kmer: score for kmer, score in combined_tf_idf.items() if score > 0.3}

    # Compute adjusted Hamming distance matrix.
    kmers, adjusted_matrix = compute_adjusted_hamming_distance_matrix(filtered_tf_idf, scaling_factor=1.0)

    # Perform approximate MDS to obtain 2D embedding.
    embedding = approximate_mds(adjusted_matrix, n_components=2, lr=0.01, n_iter=5000)

    # Create a storage table with k-mers, TF-IDF scores, and their embedding coordinates.
    embedding_table = create_storage_table(kmers, embedding, filtered_tf_idf)

    # Save the resulting table as a CSV file.
    embedding_table.to_csv("embedding_table.csv", index=False)
    print("Embedding table saved as 'embedding_table.csv'")
