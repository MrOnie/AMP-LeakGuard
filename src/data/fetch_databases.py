"""
fetch_databases.py
==================
Automated retrieval and caching of large-scale antimicrobial peptide datasets
from UniProtKB (Swiss-Prot reviewed) and curated background proteomes.
"""

from pathlib import Path
from typing import Optional
import requests
import pandas as pd


RAW_DATA_DIR = Path("data/raw")


def fetch_uniprot_amps(output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Downloads curated, reviewed antimicrobial peptides (Swiss-Prot) from UniProt REST API.
    Uses UniProt Keyword identifier KW-0929 ('Antimicrobial').
    """
    if output_path is None:
        output_path = RAW_DATA_DIR / "uniprot_reviewed_amps.tsv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 5000:
        print(f"[i] Loading cached UniProt AMPs from {output_path}")
        df = pd.read_csv(output_path, sep="\t")
        return df

    print("[*] Downloading reviewed Swiss-Prot AMPs (KW-0929) via UniProt Stream API...")
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": "keyword:KW-0929 AND reviewed:true AND length:[10 TO 100]",
        "format": "tsv",
        "fields": "accession,id,sequence,length",
    }

    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    df = pd.read_csv(output_path, sep="\t")
    print(f"[+] Successfully retrieved and saved {len(df)} AMP candidate records to {output_path}")
    return df


def fetch_uniprot_negative_background(
    output_path: Optional[Path] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Downloads reviewed Swiss-Prot non-antimicrobial proteins (excluding toxins, antimicrobials)
    to serve as source for extracting negative sequence slices (Schemes A and B).
    """
    if output_path is None:
        output_path = RAW_DATA_DIR / "uniprot_background_proteins.tsv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 5000:
        print(f"[i] Loading cached background proteins from {output_path}")
        df = pd.read_csv(output_path, sep="\t")
        return df

    print(f"[*] Downloading {limit} background proteins from UniProt Search API...")
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": "reviewed:true AND organism_id:9606 AND NOT keyword:KW-0929 AND length:[50 TO 500]",
        "format": "tsv",
        "fields": "accession,id,sequence,length",
        "size": "500",
    }

    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    df = pd.read_csv(output_path, sep="\t")
    print(f"[+] Successfully retrieved and saved {len(df)} background protein records to {output_path}")
    return df


if __name__ == "__main__":
    df_amp = fetch_uniprot_amps()
    df_bg = fetch_uniprot_negative_background()
    print(f"Total AMPs: {len(df_amp)}, Total Background Proteins: {len(df_bg)}")
