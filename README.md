# DTAF-DGRP: Drug–Transcriptome Attention Fusion and Dual-Graph Response Propagation for Cancer Drug Response Prediction

Cancer cell lines can respond differently to the same drug, and screening data cover only a subset of drug–cell line pairs. DTAF-DGRP predicts responses for unmeasured pairs by combining molecular features with patterns in measured responses. The DTAF branch uses a cell line's gene expression profile to weight individual drug atoms through cross-attention that combines dot-product and additive scores. A dual-gated bilinear network then predicts the response from the cell and drug representations. The DGRP branch uses miRNA–mRNA regulatory information to refine cell-line similarities and jointly reconstructs missing responses on cell-line and drug graphs through dual-graph Laplacian completion. Ridge regression fitted on validation predictions combines the two estimates. On the GDSC benchmark, DTAF-DGRP achieved a mean Pearson correlation of 0.9401 and a mean RMSE of 0.04120, outperforming all seven baselines on both metrics. Atom-level analysis also highlighted structures involved in target binding: for JQ1, the triazole ring accounted for 16.1% of the atoms and received 45.6% of the model's attention on average.

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
