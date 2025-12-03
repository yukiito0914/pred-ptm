import os
import sys
import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley
from tqdm import tqdm
import traceback # For printing detailed error logs

# ==========================================
# Path Configuration
# ==========================================
LABEL_FILE = "/projectnb/cs523aw/students/yuki/Phosphorylation_human_S.txt"
RESULTS_DIR = "/projectnb/cs523aw/students/yuki/results"
OUTPUT_CSV = "/projectnb/cs523aw/students/yuki/nn/data/train_data.csv"

# ==========================================
# Hyperparameters
# ==========================================
WINDOW_SIZE = 15
RADIUS = WINDOW_SIZE // 2

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

def get_structure_features(cif_path, protein_id):
    """
    Parses CIF and extracts features + residue names.
    Returns: 
        (features_array, residue_names_list)
        - features_array: np.array of shape (SeqLen, 3)
        - residue_names_list: list of strings ['MET', 'ALA', 'SER', ...]
    """
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(protein_id, cif_path)
    except Exception as e:
        # print(f"Error parsing {cif_path}: {e}")
        return None, None

    model = structure[0]
    
    sr = ShrakeRupley()
    try:
        sr.compute(model, level="R")
    except Exception as e:
        return None, None

    features = []
    res_names = [] # Store residue names for validation
    
    chain = next(iter(model))

    for residue in chain:
        resname = residue.get_resname()
        
        if resname not in HYDROPHOBICITY:
            continue

        try:
            plddt = residue['CA'].get_bfactor() / 100.0
        except:
            plddt = 0.0

        sasa = residue.sasa
        hydro = HYDROPHOBICITY.get(resname, 0.0)

        features.append([plddt, sasa, hydro])
        res_names.append(resname) # Add name to list
            
    return np.array(features, dtype=np.float32), res_names

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    print(f"Reading labels from {LABEL_FILE}...")
    
    try:
        df = pd.read_csv(LABEL_FILE, sep='\t')
        if 'Class' not in df.columns:
            raise ValueError("Tab separation failed")
    except:
        df = pd.read_csv(LABEL_FILE, delim_whitespace=True)

    df = df.rename(columns={'Accession': 'uniprot_id', 'Site': 'location', 'Class': 'label'})
    print(f"Total labels found: {len(df)}")
    
    grouped = df.groupby('uniprot_id')
    
    final_data = []
    skipped_proteins = 0
    skipped_sites_mismatch = 0 # Counter for mismatch errors
    processed_count = 0
    
    print("Processing proteins...")
    
    for uid, group in tqdm(grouped):
        # --- Error Handling Block for Protein Level ---
        try:
            cif_path = get_cif_path(RESULTS_DIR, uid)
            
            if not os.path.exists(cif_path):
                skipped_proteins += 1
                continue
                
            # Get features AND residue names
            seq_features, seq_resnames = get_structure_features(cif_path, uid)
            
            if seq_features is None:
                skipped_proteins += 1
                continue
                
            seq_len = len(seq_features)
            
            # Iterate through sites
            for _, row in group.iterrows():
                # --- Error Handling Block for Site Level ---
                try:
                    center_idx = int(row['location']) - 1
                    label_char = row['label']
                    
                    if center_idx < 0 or center_idx >= seq_len:
                        continue

                    # === VALIDATION STEP ===
                    # Check if the amino acid at center_idx is actually SER
                    actual_res = seq_resnames[center_idx]
                    if actual_res != 'SER':
                        # If you want to see the error log, uncomment the print below
                        # print(f"Warning: ID {uid} Site {row['location']} is {actual_res}, not SER. Skipping.")
                        skipped_sites_mismatch += 1
                        continue
                    # =======================

                    start = center_idx - RADIUS
                    end = center_idx + RADIUS + 1
                    
                    window_feat = np.zeros((WINDOW_SIZE, 3), dtype=np.float32)
                    valid_start = max(0, start)
                    valid_end = min(seq_len, end)
                    dest_start = valid_start - start
                    dest_end = dest_start + (valid_end - valid_start)
                    
                    if valid_end > valid_start:
                        window_feat[dest_start:dest_end] = seq_features[valid_start:valid_end]
                    
                    flat_feat = window_feat.flatten()
                    
                    sample = [uid, row['location'], label_char] + flat_feat.tolist()
                    final_data.append(sample)
                    processed_count += 1
                    
                except Exception as e:
                    # Catch individual site processing errors
                    print(f"Error processing site {row['location']} for protein {uid}: {e}")
                    traceback.print_exc() # Print full error trace
                    continue

        except Exception as e:
            # Catch protein file processing errors
            print(f"Error processing protein {uid}: {e}")
            skipped_proteins += 1
            continue

    print("Saving to CSV...")
    feat_cols = [f"feat_{i}" for i in range(WINDOW_SIZE * 3)]
    columns = ['uniprot_id', 'location', 'label'] + feat_cols
    
    result_df = pd.DataFrame(final_data, columns=columns)
    result_df.to_csv(OUTPUT_CSV, index=False)
    
    print("="*30)
    print(f"Processing Complete!")
    print(f"Output saved to: {OUTPUT_CSV}")
    print(f"Total samples generated: {len(result_df)}")
    print(f"Skipped proteins (missing file/parse error): {skipped_proteins}")
    print(f"Skipped sites (Not SER mismatch): {skipped_sites_mismatch}")
    print("="*30)

if __name__ == "__main__":
    main()