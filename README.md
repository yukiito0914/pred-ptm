# PTM Site Prediction

## Overview
This project implements a hybrid deep learning model combining **Deep 1D-CNN**, **Self-Attention**, and **Residual Connections** to predict phosphorylation sites (specifically Serine) from protein sequence and structural features.

The model integrates:
* **Structural Features:** pLDDT, SASA (Solvent Accessible Surface Area), and Hydrophobicity extracted from Boltz2 structure predictions.
* **Sequence Features:** Amino acid embeddings processed by CNN and Attention layers to capture local motifs and global context.

**Window Size:** 31 residues (Target +/- 15).

<img width="431" height="658" alt="image" src="https://github.com/user-attachments/assets/75e99601-5491-4e2a-8e70-619e144ed78d" />


## Environment Setup

### 1. Prerequisites
* Python 3.9+
* Conda (Anaconda or Miniconda)
* CUDA-enabled GPU (Recommended)

### 2. Installation
Create the conda environment using the provided YAML file.

```bash
conda env create -f environment.yml
conda activate ptm_pred
```


## Inference 
To predict phosphorylation sites on new proteins (without labels).

### 1: Prepare Target List
Create a text file containing the UniProt IDs you want to predict.

**File:** `target_ids.txt`
```text
P04637
Q9XYZ1
A0A024R1R8
```

### 2: Generate Inference Data
Extracts features for ALL Serine (S) residues found in the target proteins.

* **Input:** `target_ids.txt` + CIF files
* **Output:** `data/inference_data_w31.csv`

```bash
python scripts/preprocess/inference/make_csv_inference.py
```

### 3: Run Prediction
Loads the trained model and outputs probabilities.

* **Input:** `data/inference_data_w31.csv` + `experiments_final/best_model.pt`
* **Output:** `results/predictions.csv`

```bash
python scripts/predict.py \
  --input_csv data/inference_data_w31.csv \
  --model_path experiments_final/best_model.pt \
  --output_csv results/predictions.csv \
  --threshold 0.52  # Use the optimal threshold found during training
```





## Training 

To train the model from scratch using labeled data.

### 1: Prepare Input Data
You need two raw inputs:
1.  **Label File:** A tab-separated file containing ground truth labels.
    * Format: `Accession` (UniProt ID), `Site` (1-based index), `Class` (P/U).
    * Example: `Phosphorylation_human_S.txt`
2.  **Structure Files:** Directory containing `.cif` files (e.g., from Boltz or AlphaFold).

### 2: Preprocessing (Heavy Calculation)
Extracts features (SASA, pLDDT, Hydrophobicity) and generates a CSV file with **Window Size 31**.
* **Input:** Label File + CIF files
* **Output:** `data/train_data_w31.csv`

```bash
python scripts/preprocess/training/make_csv.py
```

### 3: Run Training
Trains the model using the generated CSV.
* **Input:** `data/train_data_w31.csv`
* **Output:** `experiments_final/best_model.pt`, `loss_curve.png`

```bash
python -u scripts/train.py \
  --csv_path data/train_data_w31.csv \
  --batch_size 1024 \
  --epochs 50 \
  --patience 10 \
  --output_dir experiments_final
```
