import os
import sys
import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser
from tqdm import tqdm

# ==========================================
# Path Configuration
# ==========================================
RESULTS_DIR = "/projectnb/cs523aw/students/yuki/results"
INPUT_CSV = "/projectnb/cs523aw/students/yuki/nn/data/train_data.csv"
OUTPUT_CSV = "/projectnb/cs523aw/students/yuki/nn/data/train_data_with_seq_31.csv"

# ==========================================
# Amino Acid Encoding (Mapping to Integers)
# ==========================================
# 20 standard amino acids + 1 for padding/unknown (0)
AA_TO_INT = {
    'PAD': 0, 'ALA': 1, 'CYS': 2, 'ASP': 3, 'GLU': 4, 'PHE': 5,
    'GLY': 6, 'HIS': 7, 'ILE': 8, 'LYS': 9, 'LEU': 10, 'MET': 11,
    'ASN': 12, 'PRO': 13, 'GLN': 14, 'ARG': 15, 'SER': 16, 'THR': 17,
    'VAL': 18, 'TRP': 19, 'TYR': 20
}
WINDOW_SIZE = 31
RADIUS = WINDOW_SIZE // 2

def get_cif_path(results_dir, uniprot_id):
    path = os.path.join(
        results_dir,
        f"boltz_results_{uniprot_id}",
        "predictions",
        uniprot_id,
        f"{uniprot_id}_model_0.cif"
    )
    return path

def get_sequence_indices(cif_path, protein_id):
    """
    Parses CIF and returns the sequence as a list of integers.
    NO SASA CALCULATION -> Very Fast.
    """
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(protein_id, cif_path)
    except:
        return None

    model = structure[0]
    chain = next(iter(model))
    
    seq_indices = []
    
    for residue in chain:
        resname = residue.get_resname()
        # Map residue name to integer (Default to 0 if unknown)
        idx = AA_TO_INT.get(resname, 0)
        seq_indices.append(idx)
            
    return np.array(seq_indices, dtype=np.int64)

def main():
    print(f"Reading input CSV: {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Store new columns here
    new_cols = []
    
    # Group by UniProt ID to minimize file I/O
    grouped = df.groupby('uniprot_id')
    
    print("Adding sequence info (skipping SASA calc)...")
    
    # We need to reconstruct the order later, so we'll store results in a dict
    # Key: original_index, Value: list of sequence integers
    results_map = {}
    
    for uid, group in tqdm(grouped):
        cif_path = get_cif_path(RESULTS_DIR, uid)
        
        if not os.path.exists(cif_path):
            # Should not happen if CSV was created correctly, but just in case
            seq_indices = np.zeros(10000, dtype=int) # Dummy
        else:
            seq_indices = get_sequence_indices(cif_path, uid)
            if seq_indices is None:
                seq_indices = np.zeros(10000, dtype=int)
        
        seq_len = len(seq_indices)
        
        for idx, row in group.iterrows():
            center_idx = int(row['location']) - 1
            
            # Window Calculation
            start = center_idx - RADIUS
            end = center_idx + RADIUS + 1
            
            # Create window with padding (0)
            window_seq = np.zeros(WINDOW_SIZE, dtype=int)
            
            valid_start = max(0, start)
            valid_end = min(seq_len, end)
            
            dest_start = valid_start - start
            dest_end = dest_start + (valid_end - valid_start)
            
            if valid_end > valid_start:
                window_seq[dest_start:dest_end] = seq_indices[valid_start:valid_end]
            
            results_map[idx] = window_seq

    # Convert results map to DataFrame columns
    print("Merging data...")
    sorted_indices = sorted(results_map.keys())
    seq_matrix = np.array([results_map[i] for i in sorted_indices])
    
    # Create column names: seq_0, seq_1, ... seq_14
    seq_col_names = [f"seq_{i}" for i in range(WINDOW_SIZE)]
    seq_df = pd.DataFrame(seq_matrix, columns=seq_col_names, index=sorted_indices)
    
    # Combine original features with new sequence features
    final_df = pd.concat([df, seq_df], axis=1)
    
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()