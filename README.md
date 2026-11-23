# DTAF-DGRP: Drug–Transcriptome Attention Fusion and Dual-Graph Response Propagation for Cancer Drug Response Prediction

DTAF-DGRP predicts how cancer cell lines respond to drugs by integrating molecular structure, gene expression, miRNA–mRNA regulatory relationships and measured drug responses. Its Drug–Transcriptome Attention Fusion (DTAF) branch uses hybrid-score cross-attention to weight drug atoms according to a cell line's gene expression profile, then models cell–drug interactions with a dual-gated bilinear multilayer perceptron. Its Dual-Graph Response Propagation (DGRP) branch builds cell-line and drug similarity graphs from measured responses, refines cell-line similarities with regulatory information and reconstructs missing responses by dual-graph Laplacian completion. Ridge regression combines the two predictions. Across five runs on the GDSC benchmark, DTAF-DGRP achieved the highest mean Pearson correlation (0.9401) and the lowest mean root mean squared error (0.04120) among all eight methods. Compared with DIPK, the strongest baseline, it increased mean correlation by 0.0104 and reduced mean RMSE by 7.8%. Atom-level attention and structural interaction analyses highlighted drug substructures associated with known target-binding features.

## Framework

![DTAF-DGRP model framework](framework.png)

## File list

```text
- data
  - inputs.zip.001 to inputs.zip.009: Packaged inputs for training and evaluation.
  - parts.json: Checksums for the nine archive parts.
  - unpack.py: Verifies and extracts the data files.
- code
  - Data.py: Loads expression profiles, drug features and response data.
  - Model.py: Defines the drug-transcriptome prediction model.
  - Train.py: Trains the neural prediction branch.
  - regulatory_similarity_caan.py: Builds the regulation-enhanced cell-line graph.
  - dual_graph_laplacian.py: Reconstructs drug responses using the cell-line and drug graphs.
  - fusion.py: Provides utilities for combining and evaluating branch predictions.
  - run_gdsc.py: Runs training, response reconstruction and prediction integration.
```

## Dataset

Before training, run `python data/unpack.py` to extract the required data.

## Run Code

### Step 1: Install dependencies

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

### Step 2: Unpack the data

```bash
python data/unpack.py
```

### Step 3: Run the model (fold 0)

```bash
python code/run_gdsc.py train --fold 0 --output outputs/fold0
python code/run_gdsc.py dgrp --fold 0 --output outputs/fold0
python code/run_gdsc.py fuse --fold 0 --output outputs/fold0
```

## Requirements

- numpy==1.26.4
- pandas==2.3.3
- scipy==1.13.1
- scikit-learn==1.6.1
- joblib==1.5.2
- torch==2.8.0+cu126
- torch-geometric==2.6.1
- optuna==4.5.0
- threadpoolctl==3.6.0
