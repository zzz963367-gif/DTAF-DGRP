from paths import DATA
from abc import ABC
import os

import joblib
import pandas as pd
import torch
from torch_geometric.data import Batch, Data, Dataset

from DataConfig import *


print("Loading Feature Dictionaries...")
PTREG_miRNA_dict = joblib.load(str(DATA / "Dataset" / 'PTReg_miRNA_expression_dict.pkl'))
PTREG_graph = joblib.load(str(DATA / "Dataset" / 'PTReg_miRNA_mRNA_graph.pkl'))
GRAPH_dict = joblib.load(str(DATA / "Dataset" / 'GRAPH_dict.pkl'))
RMA_dict = joblib.load(str(DATA / "Dataset" / 'RMA_dict.pkl'))
MolGNet_dict = joblib.load(str(DATA / "Dataset" / 'MolGNet_dict.pkl'))
USE_SCGPT_FEATURE = os.environ.get("USE_SCGPT_LATENT", "0") == "1"
if USE_SCGPT_FEATURE:
    SCGPT_payload = joblib.load(
        str(DATA / "unsupported_scgpt.pkl")
    )
else:
    SCGPT_payload = {"cell_feature_by_name": {}, "feature_dim": 512}
print("Dictionaries Loaded.")

PTREG_miRNA_dim = len(PTREG_graph["mirna_names"])
PTREG_missing_miRNA = torch.zeros(PTREG_miRNA_dim, dtype=torch.float32)
SCGPT_feature_dict = SCGPT_payload["cell_feature_by_name"]
SCGPT_feature_dim = int(SCGPT_payload["feature_dim"])
SCGPT_missing_feature = torch.zeros(SCGPT_feature_dim, dtype=torch.float32)

val_set_path = os.environ.get(
    "MODELADV_VAL_CSV",
    str(DATA / "Dataset" / f"lenient_split_fold_{fold_num}_validation.csv"),
)
test_set_path = os.environ.get(
    "MODELADV_TEST_CSV",
    str(DATA / "Dataset" / 'lenient_split_test.csv'),
)
total_set_path = os.environ.get(
    "MODELADV_TOTAL_CSV",
    str(DATA / "Dataset" / 'lenient_split_fold_0_dataset.csv'),
)
train_set_path = os.environ.get("MODELADV_TRAIN_CSV", "").strip()


def _make_graph(cell, drug, ic50):
    graph = GRAPH_dict[drug]
    mirna_available = cell in PTREG_miRNA_dict
    scgpt_available = cell in SCGPT_feature_dict
    mirna_feature = torch.as_tensor(
        PTREG_miRNA_dict[cell] if mirna_available else PTREG_missing_miRNA,
        dtype=torch.float32,
    )
    scgpt_feature = torch.as_tensor(
        SCGPT_feature_dict[cell] if scgpt_available else SCGPT_missing_feature,
        dtype=torch.float32,
    )
    return Data(
        x=MolGNet_dict[drug],
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        GEF=torch.tensor(RMA_dict[cell], dtype=torch.float32),
        MIRNA=mirna_feature.clone(),
        MIRNA_AVAILABLE=torch.tensor([1.0 if mirna_available else 0.0], dtype=torch.float32),
        SCGPT=scgpt_feature.clone(),
        ic50=torch.tensor([float(ic50)], dtype=torch.float32),
    )


def GetData(csv_path):
    dataset = pd.read_csv(csv_path, header=0)
    cells = dataset.iloc[:, 0].astype(str).tolist()
    drugs = dataset.iloc[:, 1].astype(str).tolist()
    ic50s = dataset.iloc[:, -1].astype(float).tolist()
    return [_make_graph(cell, drug, ic50) for cell, drug, ic50 in zip(cells, drugs, ic50s)]


def LenientRemove(cells, drugs, ic50s, val_csv_path, test_csv_path):
    held_out = pd.concat((pd.read_csv(val_csv_path, header=0), pd.read_csv(test_csv_path, header=0)))
    held_out_pairs = set(zip(held_out.iloc[:, 0].astype(str), held_out.iloc[:, 1].astype(str)))
    kept = [
        (cell, drug, ic50)
        for cell, drug, ic50 in zip(cells, drugs, ic50s)
        if (cell, drug) not in held_out_pairs
    ]
    return zip(*kept)


def GetTrainData(total_csv_path, val_csv_path, test_csv_path):
    dataset = pd.read_csv(total_csv_path, header=0)
    cells = dataset.iloc[:, 0].astype(str).tolist()
    drugs = dataset.iloc[:, 1].astype(str).tolist()
    ic50s = dataset.iloc[:, -1].astype(float).tolist()
    cells, drugs, ic50s = LenientRemove(cells, drugs, ic50s, val_csv_path, test_csv_path)
    return [_make_graph(cell, drug, ic50) for cell, drug, ic50 in zip(cells, drugs, ic50s)]


if debug == 0:
    print(f"Mode: Training (Fold {fold_num})")
    Gtrain = GetData(train_set_path) if train_set_path else GetTrainData(total_set_path, val_set_path, test_set_path)
    Gval = GetData(val_set_path)
else:
    print("Mode: Debugging")
    Gtrain = GetData(val_set_path)
    Gval = GetData(val_set_path)
Gtest = GetData(test_set_path)


class CollateFn:
    def __init__(self, follow_batch=None, exclude_keys=None):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        pyg_list = [
            Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, ic50=graph.ic50)
            for graph in batch
        ]
        pyg_batch = Batch.from_data_list(pyg_list, self.follow_batch, self.exclude_keys)
        gene_features = torch.stack([graph.GEF for graph in batch])
        mirna_features = torch.stack([graph.MIRNA for graph in batch])
        mirna_available = torch.stack([graph.MIRNA_AVAILABLE for graph in batch])
        scgpt_features = torch.stack([graph.SCGPT for graph in batch])
        return pyg_batch, gene_features, mirna_features, mirna_available, scgpt_features


class MyDataSet(Dataset, ABC):
    def __init__(self, graphs):
        self._graphs = graphs

    def __getitem__(self, index):
        return self._graphs[index]

    def __len__(self):
        return len(self._graphs)
