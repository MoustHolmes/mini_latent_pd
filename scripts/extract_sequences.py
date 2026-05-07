"""Extract protein sequences from mdCATH HDF5 files and write a FASTA file.

Run this on the HPC before running generate_msa.py:

    python scripts/extract_sequences.py \
        --data_dir /path/to/mdcath/data \
        --out_fasta /path/to/mdcath/sequences.fasta
"""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np


_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # MD force-field variants
    "HIE": "H", "HID": "H", "HIP": "H",
    "CYX": "C", "CYM": "C",
    "ASH": "D", "GLH": "E",
    "LYN": "K",
}


def sequence_from_pdb_text(pdb_text: str) -> str:
    """Extract one-letter sequence from PDB ATOM records (CA atoms only)."""
    seq = []
    seen_resids = set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        resname = line[17:20].strip()
        resid = line[22:26].strip()
        chain = line[21]
        key = (chain, resid)
        if key in seen_resids:
            continue
        seen_resids.add(key)
        aa = _3TO1.get(resname, "X")
        seq.append(aa)
    return "".join(seq)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sequences from mdCATH HDF5 files")
    parser.add_argument("--data_dir", required=True, type=Path,
                        help="Directory containing mdcath_dataset_*.h5 files")
    parser.add_argument("--out_fasta", required=True, type=Path,
                        help="Output FASTA file path")
    parser.add_argument("--min_len", type=int, default=10,
                        help="Skip proteins shorter than this (default: 10)")
    parser.add_argument("--max_len", type=int, default=9999,
                        help="Skip proteins longer than this (default: 9999)")
    args = parser.parse_args()

    h5_files = sorted(args.data_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {args.data_dir}")

    print(f"Found {len(h5_files)} HDF5 files in {args.data_dir}")

    args.out_fasta.parent.mkdir(parents=True, exist_ok=True)
    records = []

    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                for domain_id in f.keys():
                    try:
                        pdb_text = f[domain_id]["pdbProteinAtoms"][()].decode("utf-8")
                        seq = sequence_from_pdb_text(pdb_text)
                        if not seq or len(seq) < args.min_len or len(seq) > args.max_len:
                            continue
                        records.append((domain_id, seq))
                        print(f"  {domain_id:12s}  L={len(seq):5d}  {seq[:40]}...")
                    except Exception as e:
                        print(f"  WARNING: could not extract {domain_id} from {h5_path.name}: {e}")
        except Exception as e:
            print(f"WARNING: could not open {h5_path.name}: {e}")

    with open(args.out_fasta, "w") as f:
        for domain_id, seq in records:
            f.write(f">{domain_id}\n{seq}\n")

    print(f"\nWrote {len(records)} sequences to {args.out_fasta}")


if __name__ == "__main__":
    main()
