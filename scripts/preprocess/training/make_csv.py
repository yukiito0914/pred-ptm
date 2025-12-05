import os
import sys
import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley
from tqdm import tqdm

# ==========================================
# Configuration for Window 31
# ==========================================
LABEL_FILE = "/projectnb/cs523aw/students/yuki/Phosphorylation_human_S.txt"
RESULTS_DIR = "/projectnb/cs523aw/students/yuki/results"
# The script will check this file to skip existing proteins
OUTPUT_CSV = "/projectnb/cs523aw/students/yuki/nn/data/train_data_w31.csv"

# Window Size 31 (Radius 15)
WINDOW_SIZE = 31
RADIUS = WINDOW_SIZE // 2

# Amino Acid Mapping
AA_TO_INT = {
    'PAD': 0, 'ALA': 1, 'CYS': 2, 'ASP': 3, 'GLU': 4, 'PHE': 5,
    'GLY': 6, 'HIS': 7, 'ILE': 8, 'LYS': 9, 'LEU': 10, 'MET': 11,
    'ASN': 12, 'PRO': 13, 'GLN': 14, 'ARG': 15, 'SER': 16, 'THR': 17,
    'VAL': 18, 'TRP': 19, 'TYR': 20
}

# Hydrophobicity Scale
HYDROPHOBICITY = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
    'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
    'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
    'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}

def get_cif_path(results_dir, uniprot_id):
    path = os.path.join(
        results_dir,
        f"boltz_results_{uniprot_id}",
        "predictions",
        uniprot_id,
        f"{uniprot_id}_model_0.cif"
    )
    return path

def get_full_data_window31(cif_path, protein_id):
    """
    Extracts BOTH structural features and sequence indices for the whole protein.
    Returns: (features_array, sequence_array)
    """
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(protein_id, cif_path)
    except:
        return None, None

    model = structure[0]
    
    # SASA Calculation (Computationally Expensive)
    sr = ShrakeRupley()
    try:
        sr.compute(model, level="R")
    except:
        return None, None

    feat_list = []
    seq_list = []
    
    # Process Chain A
    try:
        chain = next(iter(model))
    except StopIteration:
        return None, None

    for residue in chain:
        resname = residue.get_resname()
        if resname not in HYDROPHOBICITY:
            continue

        # 1. Structural Features
        try:
            plddt = residue['CA'].get_bfactor() / 100.0
        except:
            plddt = 0.0
        sasa = residue.sasa
        hydro = HYDROPHOBICITY.get(resname, 0.0)
        feat_list.append([plddt, sasa, hydro])
        
        # 2. Sequence Index
        seq_idx = AA_TO_INT.get(resname, 0)
        seq_list.append(seq_idx)
            
    return np.array(feat_list, dtype=np.float32), np.array(seq_list, dtype=np.int64)

def main():
    # Make sure dir exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # --- [NEW] Check for existing data to resume ---
    processed_uids = set()
    existing_df = None
    
    if os.path.exists(OUTPUT_CSV):
        print(f"Found existing CSV: {OUTPUT_CSV}")
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            if 'uniprot_id' in existing_df.columns:
                processed_uids = set(existing_df['uniprot_id'].unique())
                print(f"--> Already processed {len(processed_uids)} proteins. Skipping them.")
        except Exception as e:
            print(f"Warning: Could not read existing CSV ({e}). Starting from scratch.")
            existing_df = None
    # -----------------------------------------------

    print(f"Reading labels from {LABEL_FILE}...")
    try:
        df = pd.read_csv(LABEL_FILE, sep='\t')
        if 'Class' not in df.columns: raise ValueError()
    except:
        df = pd.read_csv(LABEL_FILE, delim_whitespace=True)

    df = df.rename(columns={'Accession': 'uniprot_id', 'Site': 'location', 'Class': 'label'})
    grouped = df.groupby('uniprot_id')
    
    final_data = []
    skipped = 0
    newly_processed_count = 0
    
    print("Processing proteins (Calculations for Window 31)...")
    
    for uid, group in tqdm(grouped):
        # --- [NEW] Skip if already processed ---
        if uid in processed_uids:
            continue
        # ---------------------------------------

        cif_path = get_cif_path(RESULTS_DIR, uid)
        
        if not os.path.exists(cif_path):
            skipped += 1
            continue
            
        # Get full sequence data
        full_feats, full_seqs = get_full_data_window31(cif_path, uid)
        
        if full_feats is None:
            skipped += 1
            continue
            
        seq_len = len(full_feats)
        newly_processed_count += 1
        
        for _, row in group.iterrows():
            center_idx = int(row['location']) - 1
            label_char = row['label']
            
            if center_idx < 0 or center_idx >= seq_len:
                continue

            # Calculate Window
            start = center_idx - RADIUS
            end = center_idx + RADIUS + 1
            
            # --- Padding Logic ---
            window_feat = np.zeros((WINDOW_SIZE, 3), dtype=np.float32)
            window_seq = np.zeros(WINDOW_SIZE, dtype=np.int64)
            
            valid_start = max(0, start)
            valid_end = min(seq_len, end)
            
            dest_start = valid_start - start
            dest_end = dest_start + (valid_end - valid_start)
            
            if valid_end > valid_start:
                window_feat[dest_start:dest_end] = full_feats[valid_start:valid_end]
                window_seq[dest_start:dest_end] = full_seqs[valid_start:valid_end]
            
            # Flatten Features: (31, 3) -> (93,)
            flat_feat = window_feat.flatten()
            
            # Combine all data into one row
            # [ID, Loc, Label] + [93 features] + [31 sequence indices]
            sample = [uid, row['location'], label_char] + flat_feat.tolist() + window_seq.tolist()
            final_data.append(sample)

    # --- [NEW] Merge and Save ---
    if not final_data and existing_df is not None:
        print("No new data to add.")
        return

    print("Saving to CSV...")
    # Generate column names
    feat_cols = [f"feat_{i}" for i in range(WINDOW_SIZE * 3)]
    seq_cols = [f"seq_{i}" for i in range(WINDOW_SIZE)]
    columns = ['uniprot_id', 'location', 'label'] + feat_cols + seq_cols
    
    new_df = pd.DataFrame(final_data, columns=columns)
    
    if existing_df is not None:
        # Append new data to existing data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Merged {len(new_df)} new rows with {len(existing_df)} existing rows.")
    else:
        new_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Created new file with {len(new_df)} rows.")
        
    print(f"Done! Saved to {OUTPUT_CSV}.")
    print(f"Skipped {skipped} proteins (missing files).")
    print(f"Newly processed proteins: {newly_processed_count}")

if __name__ == "__main__":
    main()