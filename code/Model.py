import random
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from optuna.samplers import TPESampler
from torch_geometric.utils import to_dense_batch

from Model_DAE import ldim
from finalmlp_transfer.hybrid_attention import HybridScoreAttention as MultiHeadAttentionLayer


features_dim_gene = ldim
features_dim_raw_gene = 5757
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISABLE_SCGPT_LATENT_ADAPTER = (
    os.environ.get("USE_SCGPT_LATENT", "0") != "1"
    or os.environ.get("SCGPT_LATENT_ABLATION", "0") == "1"
    or os.environ.get("MRNA_DENOISING_INPUT", "1") == "1"
)
ATOM_AGGREGATION_MODE = os.environ.get("DTAF_ATOM_AGGREGATION", "transcriptome_attention").strip().lower()
VALID_ATOM_AGGREGATION_MODES = {"mean_pooling", "learned_query", "transcriptome_attention"}
if ATOM_AGGREGATION_MODE not in VALID_ATOM_AGGREGATION_MODES:
    raise ValueError(
        "DTAF_ATOM_AGGREGATION must be one of {}, got {!r}".format(
            sorted(VALID_ATOM_AGGREGATION_MODES), ATOM_AGGREGATION_MODE
        )
    )


class SCGPTLatentResidualAdapter(nn.Module):

    def __init__(self):
        super(SCGPTLatentResidualAdapter, self).__init__()
        self.scgpt_proj = nn.Sequential(
            nn.LayerNorm(features_dim_gene),
            nn.Linear(features_dim_gene, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim_gene),
        )
        self.latent_norm = nn.LayerNorm(features_dim_gene)
        self.residual_scale = nn.Parameter(torch.tensor(-4.5))

    def forward(self, gene_latent, scgpt_cell_feature):
        if DISABLE_SCGPT_LATENT_ADAPTER:
            return gene_latent
        delta = self.scgpt_proj(scgpt_cell_feature)
        gate = torch.sigmoid(self.residual_scale)
        return self.latent_norm(gene_latent + gate * delta)


class CellEncoder(nn.Module):
    def __init__(self):
        super(CellEncoder, self).__init__()
        self.enc_gene = nn.Linear(features_dim_gene, 512)

    def forward(self, gene):
        z_gene = F.relu(self.enc_gene(gene))
        return z_gene


class AtomOnlyCrossAttentionLayer(nn.Module):

    def __init__(self, heads):
        super(AtomOnlyCrossAttentionLayer, self).__init__()
        self.fc_map_gene = nn.Linear(512, 768)
        self.attention_gene = MultiHeadAttentionLayer(hid_dim=768, n_heads=heads, dropout=0.3, device=DEVICE)
        self.learned_query = nn.Parameter(torch.zeros(1, 1, 768))
        self.aggregation_mode = ATOM_AGGREGATION_MODE

    def forward(self, x, g, z_gene):
        drug_atoms, atom_mask = to_dense_batch(x, g.batch)
        mode = getattr(self, "aggregation_mode", "transcriptome_attention")
        if mode == "mean_pooling":
            weights = atom_mask.unsqueeze(-1).to(drug_atoms.dtype)
            return (drug_atoms * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        mask = torch.unsqueeze(torch.unsqueeze(atom_mask, 1), 1)
        if mode == "learned_query":
            query = self.learned_query.expand(drug_atoms.size(0), -1, -1)
        else:
            query = F.relu(self.fc_map_gene(z_gene)).unsqueeze(1)
        x_att = self.attention_gene(query, drug_atoms, drug_atoms, mask)[0]
        return torch.squeeze(x_att, dim=1)


from finalmlp_transfer.strict_head import StrictFinalMLPDense

class DenseLayers(StrictFinalMLPDense):
    pass


class Predictor(nn.Module):
    def __init__(self, embedding_dim, heads, fc_layer_num, fc_layer_dim, dropout_rate):
        super(Predictor, self).__init__()
        del embedding_dim
        self.scgpt_latent_adapter = SCGPTLatentResidualAdapter()
        self.cell_encoder = CellEncoder()
        self.atom_attention_fusion = AtomOnlyCrossAttentionLayer(heads=heads)
        self.dense_layers = DenseLayers(heads, fc_layer_num, fc_layer_dim, dropout_rate)

    def adapt_gene_latent(self, gene_latent, scgpt_cell_feature):
        return self.scgpt_latent_adapter(gene_latent, scgpt_cell_feature)

    def forward(self, x, g, gene_latent, gene_raw, scgpt_cell_feature):
        del gene_raw
        gene_latent = self.adapt_gene_latent(gene_latent, scgpt_cell_feature)
        z_gene = self.cell_encoder(gene_latent)
        x_fused = self.atom_attention_fusion(x, g, z_gene)
        prediction = self.dense_layers(x_fused, z_gene)
        return prediction


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    sampler = TPESampler(seed=seed)
    return sampler
