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

class FinalMLPHead(nn.Module):
    def __init__(self,mean,std,hidden=(512,256,128),heads=4,dropout=.1):
        super().__init__();self.register_buffer('mean',mean.clone());self.register_buffer('std',std.clone())
        self.selection=FeatureSelection()
        def stream():
            modules=[];last=1280
            for width in hidden:modules.extend([nn.Linear(last,width),nn.ReLU(),nn.Dropout(dropout)]);last=width
            return nn.Sequential(*modules)
        self.stream1=stream();self.stream2=stream()
        self.fusion=InteractionAggregation(hidden[-1],hidden[-1],1,heads)
    def forward(self,x):
        a,b=self.selection((x-self.mean)/self.std)
        return self.fusion(self.stream1(a),self.stream2(b)).squeeze(-1)

class FinalResponseModel(nn.Module):
    def __init__(self,encoder,cell_encoder,attention,head):
        super().__init__();self.encoder=encoder;self.cell_encoder=cell_encoder
        self.attention=attention;self.head=head
    def extract(self,rr,dd,data):
        self.encoder.eval();unique,inverse=torch.unique(rr,return_inverse=True)
        cell=self.cell_encoder(self.encoder(data['raw'][unique]))[inverse]
        query=torch.relu(self.attention.fc_map_gene(cell)).unsqueeze(1)
        drug=self.attention.attention_gene(query,data['atoms'][dd],data['atoms'][dd],data['mask'][dd,None,None,:])[0].squeeze(1)
        return torch.cat([cell,drug],dim=1)
    def forward(self,rr,dd,data):return self.head(self.extract(rr,dd,data))

def self_check():
    torch.manual_seed(73);m=InteractionAggregation(12,16,2,4).double()
    x=torch.randn(5,12,dtype=torch.double,requires_grad=True);y=torch.randn(5,16,dtype=torch.double,requires_grad=True)
    out=m(x,y);hx=x.view(-1,4,3);hy=y.view(-1,4,4)
    xy=torch.matmul(torch.matmul(hx.unsqueeze(2),m.w_xy.view(4,3,-1)).view(-1,4,2,4),hy.unsqueeze(-1)).squeeze(-1)
    reference=m.w_x(x)+m.w_y(y)+xy.sum(1)
    assert torch.allclose(out,reference,atol=1e-10)
    scalar=InteractionAggregation(12,16,1,4).double()
    ours=scalar(x,y)
    original=scalar.w_x(x)+scalar.w_y(y)+torch.matmul(torch.matmul(hx.unsqueeze(2),scalar.w_xy.view(4,3,-1)).view(-1,4,1,4),hy.unsqueeze(-1)).squeeze(-1).sum(1)
    error=float((ours-original).detach().abs().max());assert error<1e-10
    params=(x,y,*scalar.parameters())
    a=torch.autograd.grad(ours.sum(),params,retain_graph=True);b=torch.autograd.grad(original.sum(),params)
    grad=max(float((u-v).abs().max()) for u,v in zip(a,b));assert grad<1e-10
    head=FinalMLPHead(torch.zeros(1280),torch.ones(1280));z=torch.randn(4,1280)
    pred=head(z);pred.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    return dict(bilinear_value_error=error,bilinear_gradient_error=grad,parameters=sum(p.numel() for p in head.parameters()))
