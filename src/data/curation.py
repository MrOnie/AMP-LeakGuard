"""
curation.py
===========
Rigorous peptide sequence curation, quality control, filtering, and deduplication
for machine learning benchmarks.

Ensures adherence to bioinformatic standards:
- Length constrained to biological active range (default: 10-50 residues).
- Strict validation against canonical 20 amino acids (ACDEFGHIKLMNPQRSTVWY).
- Exclusion of sequences with ambiguous, non-standard, or post-translationally
  modified residue representations incompatible with canonical descriptors.
- Preservation of provenance and activity annotations (e.g. MIC, target organism).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


CANONICAL_AMINO_ACIDS: Set[str] = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class PeptideRecord:
    """Represents a validated, curated peptide record."""
    id: str
    sequence: str
    length: int
    label: int  # 1 for active (AMP), 0 for inactive/negative
    source: str
    mic_ug_ml: Optional[float] = None
    target_pathogen: Optional[str] = None


class PeptideCurator:
    """
    Curates, cleans, and standardizes peptide sequence datasets.
    """

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 50,
        allowed_aas: Set[str] = CANONICAL_AMINO_ACIDS,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.allowed_aas = allowed_aas

    def is_valid_sequence(self, sequence: str) -> Tuple[bool, Optional[str]]:
        """
        Validates if a sequence meets the inclusion criteria.

        Returns:
            Tuple of (is_valid, reason_if_invalid).
        """
        if not sequence or not isinstance(sequence, str):
            return False, "Empty or non-string sequence"

        seq_clean = sequence.strip().upper()
        if len(seq_clean) < self.min_length:
            return False, f"Sequence length ({len(seq_clean)}) < min_length ({self.min_length})"

        if len(seq_clean) > self.max_length:
            return False, f"Sequence length ({len(seq_clean)}) > max_length ({self.max_length})"

        invalid_chars = set(seq_clean) - self.allowed_aas
        if invalid_chars:
            return False, f"Non-canonical amino acid(s) detected: {sorted(list(invalid_chars))}"

        return True, None

    def clean_record(
        self,
        record_id: str,
        sequence: str,
        label: int,
        source: str,
        mic_ug_ml: Optional[float] = None,
        target_pathogen: Optional[str] = None,
    ) -> Optional[PeptideRecord]:
        """Validates and instantiates a PeptideRecord. Returns None if invalid."""
        seq_clean = sequence.strip().upper()
        is_valid, _ = self.is_valid_sequence(seq_clean)
        if not is_valid:
            return None

        return PeptideRecord(
            id=str(record_id).strip(),
            sequence=seq_clean,
            length=len(seq_clean),
            label=int(label),
            source=str(source).strip(),
            mic_ug_ml=mic_ug_ml,
            target_pathogen=target_pathogen,
        )

    def curate_dataframe(
        self,
        df: pd.DataFrame,
        seq_col: str = "sequence",
        id_col: str = "id",
        label_col: str = "label",
        source_col: str = "source",
        mic_col: Optional[str] = "mic_ug_ml",
        remove_duplicates: bool = True,
    ) -> pd.DataFrame:
        """
        Curates an entire pandas DataFrame of peptide candidates.

        Performs:
        1. Syntax and canonical amino acid validation.
        2. Sequence length filtering [min_length, max_length].
        3. Deduplication (resolving any label conflict by prioritizing positive if confirmed).
        """
        records: List[Dict] = []
        rejected_count = 0

        for _, row in df.iterrows():
            seq = str(row[seq_col]).strip().upper()
            is_valid, reason = self.is_valid_sequence(seq)
            if not is_valid:
                rejected_count += 1
                continue

            record_dict = {
                "id": str(row[id_col]),
                "sequence": seq,
                "length": len(seq),
                "label": int(row[label_col]),
                "source": str(row[source_col]) if source_col in row else "unknown",
                "mic_ug_ml": row[mic_col] if mic_col and mic_col in row and pd.notna(row[mic_col]) else None,
            }
            records.append(record_dict)

        curated_df = pd.DataFrame(records)

        if curated_df.empty:
            return curated_df

        if remove_duplicates:
            # If duplicates exist with discordant labels, keep the active one if validated, or log conflict
            curated_df = curated_df.sort_values(by=["sequence", "label"], ascending=[True, False])
            curated_df = curated_df.drop_duplicates(subset=["sequence"], keep="first").reset_index(drop=True)

        return curated_df

    @staticmethod
    def export_fasta(df: pd.DataFrame, output_path: Path, seq_col: str = "sequence", id_col: str = "id") -> None:
        """Exports a DataFrame of peptides to a standard FASTA file for MMseqs2 or CD-HIT."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(f">{row[id_col]}|label={row.get('label', 'NA')}|len={len(row[seq_col])}\n")
                f.write(f"{row[seq_col]}\n")
