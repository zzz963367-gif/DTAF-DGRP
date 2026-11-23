"""DenseLayers replacement only; no input or target normalization.

FinalMLP: Mao et al., AAAI 2023. Uses the attributed bilinear module in model.py.
All legacy TrainConfig values are accepted unchanged. Per-stream widths follow
the active original DenseLayers width sequence; dropout stays at the output of
each stream, matching the original MLP's dropout position and probability.
"""
import torch
from torch import nn
from .model import FeatureSelection,InteractionAggregation

class StrictFinalMLPDense(nn.Module):
    def __init__(self,heads,fc_layer_num,fc_layer_dim,dropout_rate):
        super().__init__()
        self.selection=FeatureSelection(cell_dim=512,drug_dim=768)
        original_widths=[512,*fc_layer_dim]
        hidden=[1280,*original_widths[:fc_layer_num]]
        def stream():
            layers=[];last=1280
            for width in hidden:
                layers.extend([nn.Linear(last,width),nn.ReLU()]);last=width
            layers.append(nn.Dropout(dropout_rate))
            return nn.Sequential(*layers)
        self.stream1=stream();self.stream2=stream()
        self.fusion=InteractionAggregation(hidden[-1],hidden[-1],output_dim=1,num_heads=heads)
    def forward(self,drug,cell):
        a,b=self.selection(torch.cat([cell,drug],dim=1))
        return self.fusion(self.stream1(a),self.stream2(b))
