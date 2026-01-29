import torch
import h5py
import numpy as np
import random
from huggingface_hub import hf_hub_download, list_repo_files

# --- Helper: Robust HDF5 Reader ---
def robust_read_h5(file_path, protein_id):
    with h5py.File(file_path, "r") as f:
        # 1. Find correct group
        if protein_id not in f:
            protein_id = list(f.keys())[0] # Fallback
        grp = f[protein_id]

        # 2. Extract Sequence (Metadata)
        sequence = "UNK"
        try:
            if "resname" in grp:
                res_names = [r.decode() for r in grp["resname"][:]]
                sequence = "".join([r[0] for r in res_names]) 
        except Exception: pass

        # 3. Find valid data paths (Temp/Replica)
        valid_paths = []
        for temp in grp.keys():
            if not temp.isdigit(): continue
            for repl in grp[temp].keys():
                dset_name = "ca_coords" if "ca_coords" in grp[temp][repl] else "coords"
                if dset_name in grp[temp][repl]:
                    n_frames = grp[temp][repl][dset_name].shape[0]
                    valid_paths.append((temp, repl, dset_name, n_frames))
    
    return protein_id, sequence, valid_paths

# --- Generator ---
# Note: Arguments are passed in, no global lookups
def mdcath_generator(repo_id, sub_sample_proteins):
    # Discovery
    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    h5_files = [f for f in all_files if f.endswith(".h5") and "mdcath_dataset_" in f]
    random.shuffle(h5_files)
    
    # Laptop Test: Limit to subset
    print(f"Generator finding files... (using subset of {sub_sample_proteins})")
    for file_name in h5_files[:sub_sample_proteins]:
        # Lazy Download
        local_path = hf_hub_download(repo_id=repo_id, filename=file_name, repo_type="dataset")
        raw_id = file_name.split("_")[-1].replace(".h5", "")
        
        try:
            pid, seq, paths = robust_read_h5(local_path, raw_id)
            if not paths: continue

            # Yield multiple frames from this file
            with h5py.File(local_path, "r") as f:
                for _ in range(10): # Yield 10 random frames per file
                    temp, repl, dset, n_frames = random.choice(paths)
                    if n_frames < 2: continue
                    idx = random.randint(0, n_frames - 1)
                    
                    # Get data
                    coords = f[pid][temp][repl][dset][idx]
                    
                    yield {
                        "id": pid,
                        "sequence": seq,
                        "temp": int(temp),
                        "coords": coords.astype(np.float32)
                    }
        except Exception as e:
            print(f"Skipping {file_name}: {e}")