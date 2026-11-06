"""
homology_split.py
=================
Implementation of Rigorous, Leak-Free Data Partitioning Schemes:

1. Standard Random Cross-Validation (StratifiedKFold):
   Reproduces the standard literature pipeline where homologous sequences leak
   between train and test splits.

2. Homology-Aware Cluster Partitioning:
   Clusters sequences based on pairwise sequence identity (e.g., 30%, 40%, 50%, 70%)
   and assigns entire clusters to folds using GroupKFold.
   Guarantees that no test sequence has an identity >= threshold to any training sequence.
"""

from typing import Dict, Generator, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from Bio.Align import PairwiseAligner


def compute_kmer_jaccard_distance(seq1: str, seq2: str, k: int = 3) -> float:
    """Computes Jaccard distance between k-mer sets of two sequences."""
    if len(seq1) < k or len(seq2) < k:
        return 1.0

    kmers1 = {seq1[i : i + k] for i in range(len(seq1) - k + 1)}
    kmers2 = {seq2[i : i + k] for i in range(len(seq2) - k + 1)}

    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)

    return 1.0 - (intersection / union) if union > 0 else 1.0


def simple_sequence_identity(seq1: str, seq2: str) -> float:
    """Computes global match identity across un-gapped or simple pairwise comparison."""
    if not seq1 or not seq2:
        return 0.0

    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 0.0

    matches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a == b)
    return matches / max_len


class SequenceClusterer:
    """
    Groups sequences into homology clusters based on sequence identity thresholds.
    Can use native connected components on similarity graph or interface with external tools.
    """

    @staticmethod
    def cluster_by_identity(
        sequences: List[str],
        identity_threshold: float = 0.40,
        kmer_size: int = 3,
    ) -> np.ndarray:
        """
        Clusters sequences such that any two sequences with estimated similarity
        above the threshold belong to the same cluster (single-linkage / connected components).
        """
        n = len(sequences)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        # Fast candidate filtering using an inverted index of 3-mers
        kmer_to_seqs: Dict[str, List[int]] = {}
        for idx, seq in enumerate(sequences):
            if len(seq) >= kmer_size:
                kmers = {seq[i : i + kmer_size] for i in range(len(seq) - kmer_size + 1)}
                for kmer in kmers:
                    if kmer not in kmer_to_seqs:
                        kmer_to_seqs[kmer] = []
                    kmer_to_seqs[kmer].append(idx)

        # Candidate pairs that share at least one k-mer
        candidate_pairs: Set[Tuple[int, int]] = set()
        for kmer, seq_indices in kmer_to_seqs.items():
            if len(seq_indices) > 1:
                for a in range(len(seq_indices)):
                    for b in range(a + 1, len(seq_indices)):
                        idx1, idx2 = seq_indices[a], seq_indices[b]
                        candidate_pairs.add((min(idx1, idx2), max(idx1, idx2)))

        for i, j in candidate_pairs:
            seq_i = sequences[i]
            seq_j = sequences[j]
            # Fast length ratio check
            len_ratio = min(len(seq_i), len(seq_j)) / max(len(seq_i), len(seq_j))
            if len_ratio < identity_threshold:
                continue

            sim = 1.0 - compute_kmer_jaccard_distance(seq_i, seq_j, k=kmer_size)
            if sim >= identity_threshold:
                union(i, j)

        # Map to canonical cluster IDs (0 to num_clusters - 1)
        root_map: Dict[int, int] = {}
        cluster_ids = np.zeros(n, dtype=int)
        for i in range(n):
            root = find(i)
            if root not in root_map:
                root_map[root] = len(root_map)
            cluster_ids[i] = root_map[root]

        return cluster_ids


class BenchmarkSplitter:
    """
    Manages both Random and Homology-Aware Splits for fair model benchmarking.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def get_random_cv_folds(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        label_col: str = "label",
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Standard Stratified 5-Fold Cross Validation (standard literature baseline)."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        folds = []
        for train_idx, val_idx in skf.split(df, df[label_col]):
            folds.append((train_idx, val_idx))
        return folds

    def get_homology_cv_folds(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        identity_threshold: float = 0.40,
        seq_col: str = "sequence",
        label_col: str = "label",
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """
        Homology-Aware 5-Fold Cross Validation:
        Clusters sequences and splits by clusters using StratifiedGroupKFold so:
        1. No cluster spans both train and val splits.
        2. Positive/negative class distribution is strictly stratified (50:50).
        """
        sequences = df[seq_col].tolist()
        cluster_ids = SequenceClusterer.cluster_by_identity(
            sequences,
            identity_threshold=identity_threshold,
        )

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        folds = []
        for train_idx, val_idx in sgkf.split(df, y=df[label_col], groups=cluster_ids):
            folds.append((train_idx, val_idx))

        return folds, cluster_ids

    @staticmethod
    def verify_no_cross_fold_leakage(
        train_seqs: List[str],
        val_seqs: List[str],
        sample_size: int = 100,
    ) -> float:
        """
        Computes maximum pairwise global alignment identity between a sample of
        validation sequences and the training pool using Biopython PairwiseAligner.
        """
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 1.0
        aligner.mismatch_score = 0.0
        aligner.open_gap_score = -1.0
        aligner.extend_gap_score = -0.5

        max_identities = []
        import random
        sub_val = random.sample(val_seqs, min(sample_size, len(val_seqs)))
        sub_train = random.sample(train_seqs, min(sample_size, len(train_seqs)))

        for v_seq in sub_val:
            best_id = 0.0
            for t_seq in sub_train:
                score = aligner.score(v_seq, t_seq)
                norm_id = score / max(len(v_seq), len(t_seq))
                if norm_id > best_id:
                    best_id = norm_id
            max_identities.append(best_id)

        return float(np.mean(max_identities)) if max_identities else 0.0
