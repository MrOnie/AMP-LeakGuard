"""
physicochemical.py
==================
Extraction of interpretable, biophysically-grounded descriptors for bioactive peptides:
- Net charge at physiological pH (Henderson-Hasselbalch with standard pKa values).
- Mean Hydrophobicity (Eisenberg and Kyte-Doolittle scales).
- Amphipathic Hydrophobic Moment (alpha-helix @ 100 deg, beta-sheet @ 180 deg).
- Aliphatic Index (Ikai, 1980).
- Boman Index / Protein-Binding Potential (Boman, 2003).
- Amino Acid Composition (AAC - 20 features).
- Positively Charged / Hydrophobic residue fractions.

All computations are vector-accelerated with NumPy and require zero proprietary software.
"""

import math
from typing import Dict, List, Union
import numpy as np
import pandas as pd


# Eisenberg Consensus Scale of Hydrophobicity (Eisenberg et al., 1984)
EISENBERG_SCALE: Dict[str, float] = {
    "A": 0.25,  "C": 0.04,  "D": -0.72, "E": -0.62, "F": 0.73,
    "G": 0.16,  "H": -0.40, "I": 0.73,  "K": -1.10, "L": 0.53,
    "M": 0.26,  "N": -0.64, "P": -0.07, "Q": -0.69, "R": -1.76,
    "S": -0.26, "T": -0.18, "V": 0.54,  "W": 0.37,  "Y": -0.03,
}

# Kyte-Doolittle Hydropathy Scale (Kyte & Doolittle, 1982)
KYTE_DOOLITTLE_SCALE: Dict[str, float] = {
    "A": 1.8,   "C": 2.5,   "D": -3.5,  "E": -3.5,  "F": 2.8,
    "G": -0.4,  "H": -3.2,  "I": 4.5,   "K": -3.9,  "L": 3.8,
    "M": 1.9,   "N": -3.5,  "P": -1.6,  "Q": -3.5,  "R": -4.5,
    "S": -0.8,  "T": -0.7,  "V": 4.2,   "W": -0.9,  "Y": -1.3,
}

# Boman Protein Interaction Scale (kcal/mol, Boman, 2003)
BOMAN_SCALE: Dict[str, float] = {
    "A": -0.49, "C": -0.40, "D": 2.76,  "E": 2.76,  "F": -2.98,
    "G": 0.00,  "H": 0.49,  "I": -4.92, "K": 2.76,  "L": -4.92,
    "M": -2.35, "N": 1.55,  "P": 0.49,  "Q": 1.55,  "R": 2.76,
    "S": 0.00,  "T": 0.00,  "V": -4.04, "W": -2.98, "Y": -1.31,
}

# Standard pKa values for charge estimation (Bjellqvist et al., 1993)
PKA_SIDE_CHAINS: Dict[str, float] = {
    "C": 8.55,  "D": 4.05,  "E": 4.45,  "H": 6.00,
    "K": 10.00, "R": 12.00, "Y": 10.00,
}
PKA_N_TERM: float = 9.69
PKA_C_TERM: float = 2.34

CANONICAL_20: List[str] = list("ACDEFGHIKLMNPQRSTVWY")


class PhysicochemicalExtractor:
    """
    Computes standard, biophysically meaningful features for peptide sequences.
    """

    @staticmethod
    def calculate_net_charge(sequence: str, ph: float = 7.0) -> float:
        """
        Calculates the net electrostatic charge of the peptide at a given pH
        using the Henderson-Hasselbalch equation.
        """
        seq = sequence.upper()
        # N-terminus (positive at acidic/neutral pH)
        charge = 1.0 / (1.0 + 10.0 ** (ph - PKA_N_TERM))
        # C-terminus (negative at neutral/basic pH)
        charge -= 1.0 / (1.0 + 10.0 ** (PKA_C_TERM - ph))

        # Positively charged side chains: K, R, H
        for aa in ("K", "R", "H"):
            count = seq.count(aa)
            if count > 0:
                pka = PKA_SIDE_CHAINS[aa]
                charge += count * (1.0 / (1.0 + 10.0 ** (ph - pka)))

        # Negatively charged side chains: D, E, C, Y
        for aa in ("D", "E", "C", "Y"):
            count = seq.count(aa)
            if count > 0:
                pka = PKA_SIDE_CHAINS[aa]
                charge -= count * (1.0 / (1.0 + 10.0 ** (pka - ph)))

        return float(charge)

    @staticmethod
    def calculate_hydrophobicity(sequence: str, scale: Dict[str, float]) -> float:
        """Calculates the mean hydrophobicity per residue for a given scale."""
        seq = sequence.upper()
        if not seq:
            return 0.0
        return float(sum(scale.get(aa, 0.0) for aa in seq) / len(seq))

    @staticmethod
    def calculate_hydrophobic_moment(sequence: str, angle_degrees: float = 100.0) -> float:
        """
        Calculates the amphipathic hydrophobic moment (mu_H) on the Eisenberg scale.
        Default angle is 100 degrees for alpha-helices; 180 degrees for beta-sheets.
        """
        seq = sequence.upper()
        n = len(seq)
        if n == 0:
            return 0.0

        angle_rad = math.radians(angle_degrees)
        sin_sum = 0.0
        cos_sum = 0.0

        for idx, aa in enumerate(seq, start=1):
            h_val = EISENBERG_SCALE.get(aa, 0.0)
            sin_sum += h_val * math.sin(idx * angle_rad)
            cos_sum += h_val * math.cos(idx * angle_rad)

        return float(math.sqrt(sin_sum ** 2 + cos_sum ** 2) / n)

    @staticmethod
    def calculate_aliphatic_index(sequence: str) -> float:
        """
        Calculates the aliphatic index according to Ikai (1980):
        Aliphatic Index = X_Ala + 2.9 * X_Val + 3.9 * (X_Ile + X_Leu)
        where X_aa is mole percent (0 to 100).
        """
        seq = sequence.upper()
        n = len(seq)
        if n == 0:
            return 0.0

        x_ala = (seq.count("A") / n) * 100.0
        x_val = (seq.count("V") / n) * 100.0
        x_ile = (seq.count("I") / n) * 100.0
        x_leu = (seq.count("L") / n) * 100.0

        return float(x_ala + 2.9 * x_val + 3.9 * (x_ile + x_leu))

    @staticmethod
    def calculate_boman_index(sequence: str) -> float:
        """
        Calculates the Boman protein-interaction index (Boman, 2003).
        A high positive value indicates high potential for protein-protein binding.
        """
        seq = sequence.upper()
        if not seq:
            return 0.0
        return float(sum(BOMAN_SCALE.get(aa, 0.0) for aa in seq) / len(seq))

    @classmethod
    def extract_sequence_features(cls, sequence: str) -> Dict[str, float]:
        """Extracts the complete set of scalar and compositional features for one sequence."""
        seq = sequence.upper()
        length = len(seq)

        features: Dict[str, float] = {
            "length": float(length),
            "net_charge_ph7": cls.calculate_net_charge(seq, ph=7.0),
            "net_charge_ph7_per_res": cls.calculate_net_charge(seq, ph=7.0) / max(length, 1),
            "hydrophobicity_eisenberg": cls.calculate_hydrophobicity(seq, EISENBERG_SCALE),
            "hydrophobicity_kyte_doolittle": cls.calculate_hydrophobicity(seq, KYTE_DOOLITTLE_SCALE),
            "hydrophobic_moment_alpha": cls.calculate_hydrophobic_moment(seq, angle_degrees=100.0),
            "hydrophobic_moment_beta": cls.calculate_hydrophobic_moment(seq, angle_degrees=180.0),
            "aliphatic_index": cls.calculate_aliphatic_index(seq),
            "boman_index": cls.calculate_boman_index(seq),
            # Key grouped structural categories
            "cationic_ratio": float(sum(seq.count(aa) for aa in ("K", "R")) / length),
            "anionic_ratio": float(sum(seq.count(aa) for aa in ("D", "E")) / length),
            "hydrophobic_ratio": float(sum(seq.count(aa) for aa in ("A", "I", "L", "V", "F", "W", "M")) / length),
            "aromatic_ratio": float(sum(seq.count(aa) for aa in ("F", "W", "Y")) / length),
        }

        # Canonical 20 Amino Acid Composition (AAC)
        for aa in CANONICAL_20:
            features[f"aac_{aa}"] = float(seq.count(aa) / length)

        return features

    @classmethod
    def extract_dataframe(cls, df: pd.DataFrame, seq_col: str = "sequence") -> pd.DataFrame:
        """Computes feature matrix for a DataFrame of sequences."""
        feature_rows: List[Dict[str, float]] = []
        for seq in df[seq_col]:
            feature_rows.append(cls.extract_sequence_features(str(seq)))

        feature_df = pd.DataFrame(feature_rows, index=df.index)
        return feature_df
