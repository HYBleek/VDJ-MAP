# VDJ-MAP

**MDS-TFIDF Visualization Tool for V(D)J Data**

VDJ-MAP is an innovative visualization toolkit designed for the analysis of V(D)J DNA sequences. It integrates advanced algorithms—including Multi-Dimensional Scaling (MDS), Term Frequency-Inverse Document Frequency (TF-IDF), and (in future releases) Uniform Manifold Approximation and Projection (UMAP)—to map high-dimensional sequence data into a low-dimensional space. This mapping enables researchers to visually explore complex relationships in immune repertoire studies.

![Alt text](figure/vdjmap_cr_vdj.png "Plot of the Cell Ranger VDJ")


## Key Features

- **Dimensionality Reduction:**  
  - **Approximate MDS:** Currently, the tool uses an approximate MDS algorithm to project k-mer distance data onto a 2D space.
  - **Planned UMAP Integration:** Future updates may incorporate UMAP to offer an alternative perspective on the data's intrinsic structure.

- **Text-Based Analysis:**  
  - **TF-IDF Scoring:** Utilizes TF-IDF to evaluate the importance of k-mers (default length = 8) across multiple DNA sequences, thereby capturing key sequence features and relative abundances.
  
- **Biological Insights:**  
  - **Pattern Recognition:** Enables the identification of clusters and patterns in VDJ gene usage, aiding in hypothesis generation and deeper immunological analyses.
  - **Statistical Comparison:** Provides detailed k-mer statistics comparing gene-specific segments against a comprehensive reference dataset.

- **Interactive Visualization:**  
  - **Overlay Plots:** Gene segment k-mers are overlaid onto a 2D embedding space derived from reference sequences. The visualization uses color gradients to denote frequency and adjusts point sizes based on k-mer density, with a light gray background representing the full k-mer embedding.

## Workflow and Implementation Details

### 1. Reference Build (2d_reference_build.py)
This script prepares the foundational embedding from reference DNA sequences:
- **Data Input:**  
  Reads reference sequences from FASTA files (e.g., `regions.fa` and `supp_regions.fa`).
- **TF-IDF Computation:**  
  Computes weighted TF-IDF scores for each k-mer, taking into account different weight factors for primary and supplementary regions.
- **CSV Output:**  
  Generates `all_kmers.csv` containing the complete set of k-mers with their TF-IDF scores.
- **Filtering and Distance Matrix:**  
  Filters out low-scoring k-mers (TF-IDF ≤ 0.3) and computes an adjusted Hamming distance matrix, where distances are modulated by the TF-IDF values.
- **Dimensionality Reduction:**  
  Uses an approximate MDS algorithm to create a 2D embedding, then saves the result in `embedding_table.csv` (including k-mer strings, TF-IDF scores, and 2D coordinates).
  
![Alt text](figure/vdjmap_ref.png "Plot of the Cell Ranger Reference")

### 2. Visualization & Analysis (output_vdj_map.py)
This script leverages the embedding to visualize and analyze gene-specific data:
- **Data Input:**  
  Uses the `embedding_table.csv` generated from the reference build and a user-supplied FASTA file containing gene segments.
- **Overlay Plot:**  
  - Plots the entire k-mer embedding as a light gray background.
  - Overlays k-mers from the gene segments with point sizes scaled by density and colors representing their frequency within the gene.
- **K-mer Statistics:**  
  Computes and prints statistics that include:
  - The number of gene k-mers absent from the overall k-mer dataset.
  - The number of k-mers present in the reference data but missing from the embedding.
  - The ratio of common k-mers, which helps assess the representativeness of the gene segments.


## Dependencies

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- PyTorch
Note that there is mMDZ in this code, it takes one A100 about 40 seconds.
*(Ensure these packages are installed in your environment.)*

## Future Enhancements

- **UMAP Integration:**  
  To offer an alternative dimensionality reduction method.
- **Interactive Features:**  
  Enhanced interactivity in plots and additional exportable analysis reports.
- **Extended Statistical Analysis:**  
  More comprehensive metrics and comparison tools for further biological insights.

## Reference Data

The reference data utilized by this toolkit is derived from the 10x public database.
- **Human V(D)J Reference (GRCh38):** Human reference (GRCh38/Ensembl/10x) dataset required for Cell Ranger V(D)J. [Download Human Reference](https://www.10xgenomics.com/support/software/cell-ranger/downloads) | File size: 3.3 MB | md5sum: 65b5b033723b07fc1bb5375e5645761c.

## Acknowledgments

This project benefited from suggestions provided by ChatGPT, an AI language model developed by OpenAI, which helped improve code structure and documentation.

---

For additional information, contributions, or to report issues, please refer to the project documentation and issue tracker.

---
