"""Generate MSA .a3m files for all proteins in a FASTA file.

Uses the ColabFold mmseqs2 web API (no local database needed).
Submits sequences in batches so the server searches them in parallel —
much faster than one-by-one submission.

Run with the colabfold pipx environment, not the project venv:

    ~/.local/pipx/venvs/colabfold/bin/python scripts/generate_msa.py \
        --fasta data/mdcath/sequences.fasta \
        --out_dir data/mdcath/msa

    # For large datasets (>100 proteins), tune batch size:
    ~/.local/pipx/venvs/colabfold/bin/python scripts/generate_msa.py \
        --fasta data/mdcath/sequences.fasta \
        --out_dir data/mdcath/msa \
        --batch_size 64

The script is resumable: proteins that already have a .a3m file are skipped.
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
                current_id = line[1:].split()[0]
            elif current_id is not None:
                sequences[current_id] = sequences.get(current_id, "") + line
    return sequences


def _extract_a3m(result, idx: int) -> str:
    """Pull the a3m string for sequence idx out of whatever run_mmseqs2 returned."""
    # run_mmseqs2 returns (a3m_lines, template_lines) or just (a3m_lines,)
    # depending on colabfold version and use_templates setting
    a3m_lines = result[0] if isinstance(result, (tuple, list)) else result
    entry = a3m_lines[idx]
    return entry if isinstance(entry, str) else entry[0]


def run_batch(ids: list[str], seqs: list[str], out_dir: Path, cache_dir: Path,
              use_env: bool, use_templates: bool, host_url: str) -> tuple[int, list[str]]:
    """Submit one batch to the ColabFold API, save .a3m files. Returns (n_saved, failed_ids)."""
    from colabfold.colabfold import run_mmseqs2

    label = f"{ids[0]}…{ids[-1]}" if len(ids) > 1 else ids[0]
    log.info(f"Batch [{label}]: submitting {len(ids)} sequences to ColabFold API...")

    try:
        result = run_mmseqs2(
            seqs,
            prefix=str(cache_dir / f"batch_{ids[0]}"),
            use_env=use_env,
            use_templates=use_templates,
            host_url=host_url,
            user_agent="mini_latent_pd/0.0.1 moust.holmes@gmail.com",
        )
    except Exception as e:
        log.error(f"Batch [{label}]: API call FAILED — {e}")
        return 0, ids

    n_saved = 0
    failed = []
    for i, domain_id in enumerate(ids):
        try:
            a3m = _extract_a3m(result, i)
            out_path = out_dir / f"{domain_id}.a3m"
            with open(out_path, "w") as f:
                f.write(a3m)
            n_saved += 1
            log.info(f"  {domain_id}: saved ({len(a3m):,} chars)")
        except Exception as e:
            log.error(f"  {domain_id}: FAILED to save — {e}")
            failed.append(domain_id)

    return n_saved, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MSA .a3m files via ColabFold API")
    parser.add_argument("--fasta", required=True, type=Path, help="Input FASTA file")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output directory for .a3m files")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Sequences per API call (default: 32). The server searches these "
                             "in parallel. Larger = faster but bigger single request.")
    parser.add_argument("--use_env", default=True, action=argparse.BooleanOptionalAction,
                        help="Include environmental sequences (default: True)")
    parser.add_argument("--use_templates", default=False, action=argparse.BooleanOptionalAction,
                        help="Include template search (default: False)")
    parser.add_argument("--host_url", default="https://api.colabfold.com",
                        help="ColabFold API URL")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / ".cache"
    cache_dir.mkdir(exist_ok=True)

    all_sequences = parse_fasta(args.fasta)
    log.info(f"Found {len(all_sequences)} sequences in {args.fasta}")

    # Filter out proteins that already have MSA files
    todo = {k: v for k, v in all_sequences.items()
            if not (args.out_dir / f"{k}.a3m").exists()}
    n_skipped = len(all_sequences) - len(todo)
    if n_skipped:
        log.info(f"Skipping {n_skipped} already-computed MSA files")
    log.info(f"Need to compute {len(todo)} MSAs (batch_size={args.batch_size})")

    ids = list(todo.keys())
    seqs = list(todo.values())
    n_total_saved = 0
    all_failed: list[str] = []

    for start in range(0, len(ids), args.batch_size):
        batch_ids = ids[start: start + args.batch_size]
        batch_seqs = seqs[start: start + args.batch_size]
        n_saved, failed = run_batch(
            batch_ids, batch_seqs, args.out_dir, cache_dir,
            args.use_env, args.use_templates, args.host_url,
        )
        n_total_saved += n_saved
        all_failed.extend(failed)
        log.info(f"Progress: {n_total_saved + n_skipped}/{len(all_sequences)} done")

    log.info(f"\nFinished. Generated {n_total_saved} new MSA files, "
             f"skipped {n_skipped} existing.")
    if all_failed:
        log.warning(f"{len(all_failed)} proteins failed: {all_failed}")
        log.warning("Re-run the script to retry failed proteins (they will be picked up automatically).")


if __name__ == "__main__":
    main()
