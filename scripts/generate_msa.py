"""Generate MSA .a3m files for all proteins in a FASTA file.

Uses the ColabFold mmseqs2 web API (no local database needed).
Run with the colabfold pipx environment, not the project venv:

    ~/.local/pipx/venvs/colabfold/bin/python scripts/generate_msa.py \
        --fasta data/mdcath/sequences.fasta \
        --out_dir data/mdcath/msa

The script skips proteins that already have an .a3m file, so it's safe to
interrupt and resume.
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_fasta(fasta_path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_id = None
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:].split()[0]  # take first word as ID
            elif current_id is not None:
                sequences[current_id] = sequences.get(current_id, "") + line
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MSA .a3m files via ColabFold API")
    parser.add_argument("--fasta", required=True, type=Path, help="Input FASTA file")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output directory for .a3m files")
    parser.add_argument("--use_env", default=True, action=argparse.BooleanOptionalAction,
                        help="Include environmental sequences (default: True)")
    parser.add_argument("--use_templates", default=False, action=argparse.BooleanOptionalAction,
                        help="Include template search (default: False — MSA only)")
    parser.add_argument("--host_url", default="https://api.colabfold.com",
                        help="ColabFold API URL")
    args = parser.parse_args()

    from colabfold.colabfold import run_mmseqs2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / ".cache"
    cache_dir.mkdir(exist_ok=True)

    sequences = parse_fasta(args.fasta)
    log.info(f"Found {len(sequences)} sequences in {args.fasta}")

    n_done = 0
    n_skipped = 0
    for domain_id, seq in sequences.items():
        out_path = args.out_dir / f"{domain_id}.a3m"
        if out_path.exists():
            log.info(f"  {domain_id}: already exists, skipping")
            n_skipped += 1
            continue

        log.info(f"  {domain_id} (L={len(seq)}): querying ColabFold API...")
        try:
            result = run_mmseqs2(
                seq,
                prefix=str(cache_dir / domain_id),
                use_env=args.use_env,
                use_templates=args.use_templates,
                host_url=args.host_url,
                user_agent="mini_latent_pd/0.0.1 moust.holmes@gmail.com",
            )
            # colabfold returns (a3m_lines, template_lines) normally, but
            # some versions return just (a3m_lines,) when use_templates=False
            a3m_lines = result[0] if isinstance(result, (tuple, list)) else result
            with open(out_path, "w") as f:
                f.write(a3m_lines[0] if isinstance(a3m_lines, list) else a3m_lines)
            n_done += 1
            log.info(f"  {domain_id}: saved {out_path} ({len(a3m_lines[0])} chars)")
        except Exception as e:
            log.error(f"  {domain_id}: FAILED — {e}")

    log.info(f"Done. Generated {n_done} new MSA files, skipped {n_skipped} existing.")


if __name__ == "__main__":
    main()
