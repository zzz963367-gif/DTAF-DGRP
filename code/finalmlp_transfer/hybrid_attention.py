import torch
from torch import nn
from Model_MHA import MultiHeadAttentionLayer as OriginalAttention


class HybridScoreAttention(OriginalAttention):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__(hid_dim, n_heads, dropout, device)
        with torch.random.fork_rng(devices=[]):
            self.add_q = nn.Parameter(torch.empty(n_heads, self.head_dim, self.head_dim))
            self.add_k = nn.Parameter(torch.empty(n_heads, self.head_dim, self.head_dim))
            for head in range(n_heads):
                nn.init.xavier_uniform_(self.add_q[head])
                nn.init.xavier_uniform_(self.add_k[head])
            self.add_bias = nn.Parameter(torch.zeros(n_heads, self.head_dim))
            self.add_out = nn.Parameter(torch.zeros(n_heads, self.head_dim))

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        def split(layer, x):
            return layer(x).view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        q, k, v = split(self.fc_q, query), split(self.fc_k, key), split(self.fc_v, value)
        energy = torch.matmul(q, k.transpose(-1,-2)) / self.scale
        aq = torch.einsum('bhqd,hdr->bhqr', q, self.add_q)
        ak = torch.einsum('bhkd,hdr->bhkr', k, self.add_k)
        joint = torch.tanh(aq.unsqueeze(3) + ak.unsqueeze(2) + self.add_bias[None,:,None,None,:])
        energy = energy + torch.einsum('bhqkr,hr->bhqk', joint, self.add_out)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), v)
        x = x.permute(0,2,1,3).contiguous().view(batch_size,-1,self.hid_dim)
        return self.fc_o(x), attention
