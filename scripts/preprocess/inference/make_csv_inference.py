import os
import sys
import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
# Path to the list of UniProt IDs you want to predict
TARGET_LIST_FILE = "/projectnb/cs523aw/students/yuki/target_ids.txt"

# Directory containing Boltz structure files (.cif)
RESULTS_DIR = "/projectnb/cs523aw/students/yuki/results"

# Output CSV path for inference
OUTPUT_CSV = "/projectnb/cs523aw/students/yuki/nn/data/inference_data_w31.csv"

# Window Size (Must match training configuration)
WINDOW_SIZE = 31
RADIUS = WINDOW_SIZE // 2  # 15

# Amino Acid Mapping (Must match training)
AA_TO_INT = {
    'PAD': 0, 'ALA': 1, 'CYS': 2, 'ASP': 3, 'GLU': 4, 'PHE': 5,
    'GLY': 6, 'HIS': 7, 'ILE': 8, 'LYS': 9, 'LEU': 10, 'MET': 11,
    'ASN': 12, 'PRO': 13, 'GLN': 14, 'ARG': 15, 'SER': 16, 'THR': 17,
    'VAL': 18, 'TRP': 19, 'TYR': 20
}

# Hydrophobicity Scale (Kyte-Doolittle)
HYDROPHOBICITY = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
    'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
    'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
    'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}

def get_cif_path(results_dir, uniprot_id):
    """
    Constructs the file path for Boltz CIF files.
    """
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
    Extracts structural features (pLDDT, SASA, Hydrophobicity) and 
    sequence indices for the entire protein chain.
    
    Returns:
        tuple: (features_array, sequence_array, residue_names_list)
               Returns (None, None, None) if parsing/calculation fails.
    """
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(protein_id, cif_path)
    except:
        return None, None, None

    model = structure[0]
    
    # Compute SASA (Solvent Accessible Surface Area)
    # This step is computationally expensive.
    sr = ShrakeRupley()
    try:
        sr.compute(model, level="R")
    except:
        return None, None, None

    feat_list = []
    seq_list = []
    res_names = [] 

    try:
        # Assume single chain or take the first chain
        chain = next(iter(model))
    except StopIteration:
        return None, None, None

    for residue in chain:
        resname = residue.get_resname()
        
        # Skip non-standard residues (water, ligands, etc.)
        if resname not in HYDROPHOBICITY:
            continue

        # 1. Structural Features
        try:
            # Normalize pLDDT to 0.0 - 1.0 range
            plddt = residue['CA'].get_bfactor() / 100.0
        except:
            plddt = 0.0
        
        sasa = residue.sasa
        hydro = HYDROPHOBICITY.get(resname, 0.0)
        
        feat_list.append([plddt, sasa, hydro])
        
        # 2. Sequence Index
        seq_idx = AA_TO_INT.get(resname, 0)
        seq_list.append(seq_idx)
        
        # 3. Residue Name (for filtering target sites later)
        res_names.append(resname)
            
    return np.array(feat_list, dtype=np.float32), np.array(seq_list, dtype=np.int64), res_names

def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Load target ID list
    if not os.path.exists(TARGET_LIST_FILE):
        print(f"Error: Target list file not found at {TARGET_LIST_FILE}")
        sys.exit(1)
        
    with open(TARGET_LIST_FILE, 'r') as f:
        target_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(target_ids)} target proteins from {TARGET_LIST_FILE}")
    
    final_data = []
    skipped = 0
    
    print("Generating inference data (Processing ALL Serines)...")
    
    for uid in tqdm(target_ids):
        cif_path = get_cif_path(RESULTS_DIR, uid)
        
        if not os.path.exists(cif_path):
            skipped += 1
            continue
            
        # Extract features for the whole sequence
        full_feats, full_seqs, res_names = get_full_data_window31(cif_path, uid)
        
        if full_feats is None:
            skipped += 1
            continue
            
        seq_len = len(full_feats)
        
        # Scan the sequence and extract windows for every Serine (S)
        for i in range(seq_len):
            # Target only Serine (SER) residues
            if res_names[i] != 'SER':
                continue
            
            # Location is 1-based for the output CSV
            loc_1based = i + 1
            center_idx = i # 0-based index for slicing

            # Calculate Window Indices
            start = center_idx - RADIUS
            end = center_idx + RADIUS + 1
            
            # --- Padding Logic ---
            window_feat = np.zeros((WINDOW_SIZE, 3), dtype=np.float32)
            window_seq = np.zeros(WINDOW_SIZE, dtype=np.int64)
            
            valid_start = max(0, start)
            valid_end = min(seq_len, end)
            
            # Indices within the window array
            dest_start = valid_start - start
            dest_end = dest_start + (valid_end - valid_start)
            
            if valid_end > valid_start:
                window_feat[dest_start:dest_end] = full_feats[valid_start:valid_end]
                window_seq[dest_start:dest_end] = full_seqs[valid_start:valid_end]
            
            # Flatten feature matrix: (31, 3) -> (93,)
            flat_feat = window_feat.flatten()
            
            # Dummy label for compatibility with existing Dataset classes (if needed)
            dummy_label = '?' 
            
            # Construct row: [ID, Location, Label, Features..., Sequence...]
            sample = [uid, loc_1based, dummy_label] + flat_feat.tolist() + window_seq.tolist()
            final_data.append(sample)

    print("Saving to CSV...")
    
    # Generate column names
    feat_cols = [f"feat_{i}" for i in range(WINDOW_SIZE * 3)]
    seq_cols = [f"seq_{i}" for i in range(WINDOW_SIZE)]
    columns = ['uniprot_id', 'location', 'label'] + feat_cols + seq_cols
    
    # Create DataFrame and save
    result_df = pd.DataFrame(final_data, columns=columns)
    result_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Done! Saved to {OUTPUT_CSV}")
    print(f"Total sites extracted: {len(result_df)}")
    print(f"Skipped proteins (missing file/error): {skipped}")

if __name__ == "__main__":
    main()