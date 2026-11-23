import torch
from torch import nn

class FeatureSelection(nn.Module):
    def __init__(self,cell_dim=512,drug_dim=768):
        super().__init__();width=cell_dim+drug_dim
        self.gate_cell=nn.Sequential(nn.Linear(cell_dim,64),nn.ReLU(),nn.Linear(64,width),nn.Sigmoid())
        self.gate_drug=nn.Sequential(nn.Linear(drug_dim,64),nn.ReLU(),nn.Linear(64,width),nn.Sigmoid())
        self.cell_dim=cell_dim
    def forward(self,x):
        cell,drug=x[:,:self.cell_dim],x[:,self.cell_dim:]
        return x*(2*self.gate_cell(cell)),x*(2*self.gate_drug(drug))

class InteractionAggregation(nn.Module):
    def __init__(self,x_dim,y_dim,output_dim=1,num_heads=4):
        super().__init__();assert x_dim%num_heads==0 and y_dim%num_heads==0
        self.num_heads=num_heads;self.output_dim=output_dim
        self.head_x_dim=x_dim//num_heads;self.head_y_dim=y_dim//num_heads
        self.w_x=nn.Linear(x_dim,output_dim);self.w_y=nn.Linear(y_dim,output_dim)
        self.w_xy=nn.Parameter(torch.empty(num_heads*self.head_x_dim*self.head_y_dim,output_dim))
        nn.init.xavier_normal_(self.w_xy)
    def forward(self,x,y):
        output=self.w_x(x)+self.w_y(y)
        hx=x.reshape(-1,self.num_heads,self.head_x_dim)
        hy=y.reshape(-1,self.num_heads,self.head_y_dim)
        weights=self.w_xy.reshape(self.num_heads,self.head_x_dim,self.output_dim,self.head_y_dim)
        return output+torch.einsum('bhi,hioj,bhj->bo',hx,weights,hy)
