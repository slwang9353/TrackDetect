import torch
import torch.nn as nn
from torch import einsum
from einops import layers, rearrange, repeat
from einops.layers.torch import Rearrange


class MLP(nn.Module):
    '''project=True then use ReLU between layers, bn or ln after the last layer'''
    def __init__(self, demensions, bn=False, ln=False, dropout=0., bias=False, project=True):
        super(MLP, self).__init__()
        self.demensions = demensions
        self.bn, self.ln, self.dropout, self.bias = bn, ln, dropout, bias
        self.project = project
        self.layers = []
        self.ac = nn.ReLU(inplace=True) if project else nn.Identity()
        for i in range(len(self.demensions) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(self.demensions[i], self.demensions[i + 1], bias=self.bias),
                    nn.Dropout(p=self.dropout),
                    self.ac
                )
            )
        if self.bn:
            self.layers.append(
                nn.BatchNorm1d(self.demensions[-1])
            )
        if self.ln:
            self.layers.append(
                nn.LayerNorm(self.demensions[-1])
            )
        self.mlp = nn.Sequential(*self.layers)
    def forward(self, x):
        return self.mlp(x)


class MSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_tokens=90):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(self.heads, num_tokens, num_tokens)
        )
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots + self.attention_biases)
        out = einsum('b h i j , b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class EncoderBlock(nn.Module):
    def __init__(self, d_model,heads=8, dim_head=64, dropout=0., num_tokens=90, exp_ratio=4):
        super(EncoderBlock, self).__init__()
        self.d_model, self.heads, self.dim_head= d_model, heads, dim_head
        self.dropout, self.num_tokens, self.exp_ratio= dropout, num_tokens, exp_ratio
        self.attention = MSA(
            self.d_model, heads=self.heads, dim_head=self.dim_head, 
            dropout=self.dropout, num_tokens=self.num_tokens
        )
        self.ln_1 = nn.LayerNorm(self.d_model)
        self.shortcut_1 = nn.Sequential()
        self.mlp = MLP(
            [self.d_model, self.exp_ratio * self.d_model, self.d_model], dropout=self.dropout
        )
        self.shortcut_2 = nn.Sequential()
        self.ln_2 = nn.LayerNorm(self.d_model)

    def forward(self, x):
        short_cut = self.shortcut_1(x)
        attention = self.attention(x)
        att_out = self.ln_1(short_cut + attention)
        short_cut = self.shortcut_2(att_out)
        mlp_out = self.mlp(att_out)
        return self.ln_2(short_cut + mlp_out)


class SeqPool(nn.Module):
    '''From tokens (n, m, d) to a class token (n, d)'''
    def __init__(self, dim, dropout=0., num_tokens=90):
        super().__init__()
        self.project = MLP([dim, 2 * dim, 1], dropout=dropout, ln=True)
        self.attend = nn.Softmax(dim=-1)
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_tokens)
        )
    
    def forward(self, x):
        score = self.project(x).squeeze()
        attn = self.attend(score + self.attention_biases)   # (n, m, 1)
        out = einsum('n m, n m d -> n d', attn, x)
        return out

class DetecModel(nn.Module):
    def __init__(self, 
            in_channel, d_model, heads=8, dim_head=64, dropout=0., 
            num_tokens=90, exp_ratio=4, num_blocks=4, class_exp=2, num_class=100
    ):
        super(DetecModel, self).__init__()
        self.in_channel = in_channel
        self.d_model, self.heads, self.dim_dead, self.dropout = d_model, heads, dim_head, dropout
        self.num_tokens, self.exp_ratio, self.num_blocks = num_tokens, exp_ratio, num_blocks
        self.class_exp, self.num_class = class_exp, num_class

        self.embedding = MLP([in_channel, d_model], dropout=dropout, bias=True, project=True)
        self.blocks = []
        for _ in range(self.num_blocks):
            self.blocks.append(
                EncoderBlock(
                    self.d_model, heads=self.heads, dim_head=self.dim_dead, 
                    dropout=self.dropout, num_tokens=self.num_tokens, exp_ratio=self.exp_ratio
                )
            )
        self.att_blocks = nn.Sequential(*self.blocks)
        self.seqpool = SeqPool(self.d_model, dropout=self.dropout, num_tokens=self.num_tokens)
        self.mlp = nn.Sequential(
            MLP([self.d_model, self.class_exp * self.d_model], dropout=self.dropout),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(self.d_model * self.class_exp, self.num_class, bias=True)
    
    def forward(self, x):
        '''Input (n, m, c)'''
        x = self.embedding(x)
        x = self.att_blocks(x)
        x = self.seqpool(x)
        x = self.mlp(x)
        scores = self.fc(x)
        return scores         


if __name__ == '__main__':
    print()
    print('########################## Inference Test ##########################')
    print()
    model = DetecModel(
        600, 512, heads=8, dim_head=64, dropout=0.5, 
        num_tokens=90, exp_ratio=4, num_blocks=4, class_exp=2, num_class=10
    )
    input_tensor = torch.randn(32, 90, 600)
    scores = model(input_tensor)
    print('Input Size: ', input_tensor.shape)
    print('Output Size: ', scores.shape)
    print()
    print('##########################                ##########################')
    print()
        
            

