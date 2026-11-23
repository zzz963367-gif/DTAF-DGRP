# DTAF-DGRP: Drug–Transcriptome Attention Fusion and Dual-Graph Response Propagation for Cancer Drug Response Prediction

Accurate cancer drug response prediction can support treatment selection and improve the efficiency of cancer drug screening. However, effective drug response prediction requires integrating diverse information from molecular features, regulatory relationships and observed drug responses. Therefore, a heterogeneous information fusion framework combining Drug–Transcriptome Attention Fusion (DTAF) and Dual-Graph Response Propagation (DGRP) is proposed. In DTAF, drug features are weighted according to cell line gene expression through hybrid-score cross-attention. A Dual-Gated Bilinear Multilayer Perceptron (DGBMLP) is then used to selectively weight relevant features and explicitly model cell–drug feature interactions through feature gating and bilinear fusion. In DGRP, unmeasured responses are reconstructed using cell and drug similarity networks derived from observed responses, with cell similarity enhanced by miRNA–mRNA regulatory information. The predictions of DTAF and DGRP are integrated through Ridge regression. On the GDSC benchmark, higher prediction accuracy was achieved by DTAF-DGRP than by seven baseline methods. Ablation and network-replacement experiments supported the contributions of the interaction module, the integration of the two branches and the graph construction strategy. Atomic attention and structural interaction analyses highlighted drug substructures associated with known target-binding features. DTAF-DGRP provides an effective computational tool for prioritizing cancer drug candidates and supporting personalized medicine.

## Framework

![DTAF-DGRP model framework](framework.png)

## File list

```text
- data
  - inputs.zip.001 to inputs.zip.009: Experiment inputs
  - parts.json: Archive checksums
  - unpack.py: Data extraction
- code
  - Data.py: Data loading
  - Model.py: DTAF model
  - Train.py: DTAF training
  - regulatory_similarity_caan.py: Regulation-enhanced cell graph
  - dual_graph_laplacian.py: Dual-graph response completion
  - fusion.py: Prediction fusion and evaluation
  - run_gdsc.py: GDSC experiment entry point
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
