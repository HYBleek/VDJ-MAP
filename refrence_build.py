import torch
import numpy as np
import pandas as pd
if torch.cuda.is_available():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def read_fasta(filename):
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
        # Save the last sequence
        if sequence_id is not None:
            sequences[sequence_id] = "".join(sequence_parts)
    return sequences

region_dict = read_fasta("regions.fa")
supp_region_dict = read_fasta("supp_regions.fa")

combined = {**region_dict, **supp_region_dict}

def count_kmers_in_seq(seq, k):
    """
    Count all k-mers in a single DNA sequence.
    Returns a dictionary of k-mer counts.
    """
    counts = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts

def count_kmers_in_dict(dna_dict, k, weight=1):
    """
    For each sequence in a dictionary, count its k-mers,
    and multiply the counts by the given weight.
    Returns a dictionary with weighted k-mer counts.
    """
    weighted_counts = {}
    for seq_id, seq in dna_dict.items():
        kmer_counts = count_kmers_in_seq(seq, k)
        for kmer, count in kmer_counts.items():
            weighted_counts[kmer] = weighted_counts.get(kmer, 0) + count * weight
    return weighted_counts

def merge_kmer_counts(dict1_counts, dict2_counts):
    """
    Merge two dictionaries containing k-mer counts.
    """
    merged_counts = dict1_counts.copy()
    for kmer, count in dict2_counts.items():
        merged_counts[kmer] = merged_counts.get(kmer, 0) + count
    return merged_counts

import math
from collections import Counter

def get_kmers(seq, k):
    """Return a list of all k-mers in a given sequence."""
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

def compute_tf_idf(dna_dict, k, weight_factors=None):
    """
    Computes the TF-IDF for k-mers in a dictionary of DNA sequences.

    Parameters:
      dna_dict (dict): Keys are sequence IDs, values are DNA sequences.
      k (int): The length of the k-mer.
      weight_factors (dict, optional): Mapping from sequence IDs to a weight factor.
                                       If None, every document is given a weight of 1.

    Returns:
      dict: A nested dictionary mapping sequence ID to a dict of {k-mer: tf-idf weight}.
    """
    if weight_factors is None:
        weight_factors = {key: 1 for key in dna_dict.keys()}

    # Step 1: Count weighted k-mers for each document.
    doc_kmer_counts = {}
    for key, seq in dna_dict.items():
        kmers = get_kmers(seq, k)
        weight = weight_factors.get(key, 1)
        counts = Counter(kmers)
        weighted_counts = {kmer: count * weight for kmer, count in counts.items()}
        doc_kmer_counts[key] = weighted_counts

    # Step 2: Compute Term Frequency (TF) for each document.
    doc_tf = {}
    for key, counts in doc_kmer_counts.items():
        total = sum(counts.values())
        tf = {kmer: count / total for kmer, count in counts.items()}
        doc_tf[key] = tf

    # Step 3: Compute Document Frequency (DF) for each k-mer.
    df = Counter()
    for counts in doc_kmer_counts.values():
        for kmer in counts.keys():
            df[kmer] += 1

    # Step 4: Compute Inverse Document Frequency (IDF) for each k-mer.
    N = len(dna_dict)
    idf = {kmer: math.log(N / df_val) for kmer, df_val in df.items()}

    # Step 5: Compute TF-IDF for each document.
    doc_tf_idf = {}
    for key, tf in doc_tf.items():
        tf_idf = {kmer: tf_val * idf[kmer] for kmer, tf_val in tf.items()}
        doc_tf_idf[key] = tf_idf

    return doc_tf_idf

def compute_tf_idf_combined(dna_dict, k, weight_factors=None):
    """
    Generates a combined k-mer to TF-IDF mapping by summing the TF-IDF scores
    across all documents.

    Returns:
      dict: Mapping from k-mer to the cumulative TF-IDF score.
    """
    # Calculate TF-IDF for each document.
    doc_tf_idf = compute_tf_idf(dna_dict, k, weight_factors)
    combined_tf_idf = {}
    for tf_idf in doc_tf_idf.values():
        for kmer, score in tf_idf.items():
            combined_tf_idf[kmer] = combined_tf_idf.get(kmer, 0) + score
    return combined_tf_idf


# Combine the two dictionaries.
combined_dna_dict = {**region_dict, **supp_region_dict}

# Define weight factors: assign a higher weight to sequences from dna_dict1.
weight_factors = {**{k: 2 for k in region_dict.keys()}, **{k: 1 for k in supp_region_dict.keys()}}

# Choose the k-mer length, for example k=3.
combined_tf_idf = compute_tf_idf_combined(combined_dna_dict, k=8, weight_factors=weight_factors)

len(combined_tf_idf)

filtered_tf_idf = {kmer: score for kmer, score in combined_tf_idf.items() if score > 0.3}

len(filtered_tf_idf)

import torch

def compute_hamming_distance_matrix(filtered_tf_idf):
    """
    Compute the Hamming distance matrix for the k-mers in filtered_tf_idf using GPU acceleration if available.

    Parameters:
        filtered_tf_idf (dict): A dictionary where keys are k-mer strings and values are TF-IDF scores.

    Returns:
        tuple: (kmers, hamming_distance_matrix)
            - kmers (list): List of k-mer strings.
            - hamming_distance_matrix (numpy.ndarray): A matrix of shape (N, N) with the Hamming distances.
    """
    # Extract k-mers from the dictionary keys.
    kmers = list(filtered_tf_idf.keys())

    # Map nucleotides to integers.
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # Convert each k-mer into a list of integers.
    kmer_ints = [[mapping[base] for base in kmer] for kmer in kmers]

    # Create a PyTorch tensor of shape (N, k) where N is the number of k-mers.
    kmer_tensor = torch.tensor(kmer_ints)

    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kmer_tensor = kmer_tensor.to(device)

    # Compute pairwise Hamming distances:
    # Expand dimensions for broadcasting and sum mismatches across k positions.
    diff = (kmer_tensor.unsqueeze(1) != kmer_tensor.unsqueeze(0)).sum(dim=2)

    # Move the result back to CPU and convert to numpy array.
    hamming_distance_matrix = diff.cpu().numpy()

    return kmers, hamming_distance_matrix


kmers, distance_matrix = compute_hamming_distance_matrix(filtered_tf_idf)

distance_matrix.shape

max(filtered_tf_idf.values())

def compute_adjusted_hamming_distance_matrix(filtered_tf_idf, scaling_factor=1.0):
    """
    Computes a modified Hamming distance matrix for k-mers, adjusting the distance
    based on their TF-IDF scores. The higher the TF-IDF score, the more the distance
    is reduced. The adjustment is computed as:
        adjusted_distance = max(0, original_hamming_distance - scaling_factor * average_tfidf)
    where average_tfidf = (tfidf_i + tfidf_j) / 2.

    Parameters:
        filtered_tf_idf (dict): A dictionary with k-mer strings as keys and TF-IDF scores as values.
        scaling_factor (float): A factor to scale the TF-IDF adjustment.

    Returns:
        tuple: (kmers, adjusted_distance_matrix)
            - kmers (list): List of k-mer strings.
            - adjusted_distance_matrix (numpy.ndarray): A matrix (shape: N x N) with the adjusted distances.
    """
    # Determine device for GPU acceleration.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract k-mers and their TF-IDF scores.
    kmers = list(filtered_tf_idf.keys())
    tfidf_scores = torch.tensor([filtered_tf_idf[kmer] for kmer in kmers], dtype=torch.float, device=device)

    # Mapping from nucleotide to integer.
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # Convert each k-mer into a list of integers.
    kmer_ints = [[mapping[base] for base in kmer] for kmer in kmers]

    # Create a tensor for the k-mers.
    kmer_tensor = torch.tensor(kmer_ints, dtype=torch.int64, device=device)

    # Compute the pairwise Hamming distances.
    # The expression below expands dimensions to compare every k-mer with every other.
    diff = (kmer_tensor.unsqueeze(1) != kmer_tensor.unsqueeze(0)).sum(dim=2)

    # Compute the pairwise average TF-IDF scores.
    adjustment = (tfidf_scores.unsqueeze(1) + tfidf_scores.unsqueeze(0)) / 2.0

    # Adjust the Hamming distances by reducing them based on the TF-IDF adjustment.
    adjusted = diff.float() - scaling_factor * adjustment
    # Ensure no negative distances.
    adjusted = torch.clamp(adjusted, min=0.1)

    # Return the list of k-mers and the adjusted distance matrix (moved back to CPU as a numpy array).
    return kmers, adjusted.cpu().numpy()

kmers, adjusted_matrix = compute_adjusted_hamming_distance_matrix(filtered_tf_idf, scaling_factor=1.0)

adjusted_matrix.shape

def approximate_mds(distance_matrix, n_components=2, lr=0.01, n_iter=500, device=None):
    """
    An approximate MDS algorithm using gradient descent in PyTorch with GPU acceleration.

    Parameters:
        distance_matrix (numpy.ndarray): Precomputed distance matrix of shape (N, N).
        n_components (int): Target dimension for the embedding.
        lr (float): Learning rate for the optimizer.
        n_iter (int): Number of iterations.
        device (str or torch.device, optional): Device to use; if None, GPU will be used if available.

    Returns:
        numpy.ndarray: An array of shape (N, n_components) where each row is the embedding for a k-mer.
    """
    # Set device: use GPU if available.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Convert the distance matrix to a torch tensor.
    D = torch.tensor(distance_matrix, dtype=torch.float, device=device)
    N = D.shape[0]

    # Initialize the embedding coordinates randomly.
    X = torch.randn(N, n_components, device=device, requires_grad=True)

    # Use an optimizer (Adam) to minimize the stress function.
    optimizer = torch.optim.Adam([X], lr=lr)

    for i in range(n_iter):
        optimizer.zero_grad()

        # Compute pairwise distances of the current embedding X.
        # Using the formula: ||x_i - x_j||^2 = x_i^2 + x_j^2 - 2 * (x_i dot x_j)
        X_norm = torch.sum(X**2, dim=1, keepdim=True)  # shape: (N, 1)
        squared_distances = X_norm + X_norm.t() - 2 * torch.matmul(X, X.t())
        squared_distances = torch.clamp(squared_distances, min=0)
        X_dists = torch.sqrt(squared_distances + 1e-8)

        # Stress: sum((||x_i - x_j|| - D_ij)^2)
        loss = torch.sum((X_dists - D) ** 2)
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            print(f"Iteration {i}, stress: {loss.item()}")

    # Return the final embedding coordinates (moved back to CPU as a NumPy array)
    return X.detach().cpu().numpy()

embedding = approximate_mds(adjusted_matrix, n_components=2, lr=0.01, n_iter=5000)

def create_storage_table(kmers, embedding):
    """
    Combines a list of k-mers with their corresponding embedding coordinates into a storage table.

    Parameters:
        kmers (list): A list of k-mer strings.
        embedding (numpy.ndarray): An array of shape (N, n_components) with embedding coordinates.

    Returns:
        pandas.DataFrame: A DataFrame with a 'kmer' column and one column per embedding dimension.
    """
    n_components = embedding.shape[1]
    # Create column names for the dimensions.
    dim_columns = [f"dim{i+1}" for i in range(n_components)]
    # Create the DataFrame for the embedding coordinates.
    df = pd.DataFrame(embedding, columns=dim_columns)
    # Insert the k-mer list as the first column.
    df.insert(0, "kmer", kmers)
    return df

embedding = np.array(embedding)
embedding_table = create_storage_table(kmers, embedding)
