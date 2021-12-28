# adapted from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import checkpoint

from einops import rearrange, repeat


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Local3dAttention(nn.Module):
    def __init__(self, extents, dim, heads=8, dim_head=64, dropout=.0, use_checkpointing=True):
        super().__init__()

        self.extents = extents
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.use_checkpointing = use_checkpointing

    def pad(self, x, pad_value=0, mask=False):
        padding = ()
        if not mask:
            padding += (0, 0)   # 'skip' embedding dim
        for i in reversed(range(3)):
            padding += (self.extents[i], self.extents[i])       
        return F.pad(x, pad=padding, value=pad_value)

    def unfold(self, x):
        for i in range(3):
            kernel_size = self.extents[i] * 2 + 1
            x = x.unfold(dimension=i+1, size=kernel_size, step=1)
        return x

    def get_mask(self, batch_shape):
        _,s,h,w,_ = batch_shape
        m = torch.zeros(1, s, h, w, dtype=torch.bool)
        m = self.pad(m, pad_value=True, mask=True)
        m = self.unfold(m)
        return m

    def attention(self, k, v, q):
        batch_size = v.shape[0]
        mask = self.get_mask(k.shape).to(k.device)

        k = self.unfold(self.pad(k))    # pad border cases to get equal sizes
        v = self.unfold(self.pad(v))

        q = rearrange(q, 'b s h w d -> (b s h w) 1 d')
        v = rearrange(v, 'b s h w d i j k -> (b s h w) (i j k) d')
        k = rearrange(k, 'b s h w d i j k -> (b s h w) (i j k) d')

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # masking
        mask_value = -1e9
        mask = repeat(mask, '1 s h w i j k -> (b s h w) heads 1 (i j k)', b=batch_size, heads=self.heads)
        dots.masked_fill_(mask, mask_value)

        attn = self.attend(dots)
        out = torch.matmul(attn, v)

        return out

    # todo: add causal masking
    def forward(self, x, q):
        q_shape = q.shape
        
        # key & value projections
        k = self.to_k(x)
        v = self.to_v(x)
        q = self.to_q(q)

        if self.use_checkpointing:
            out = checkpoint.checkpoint(self.attention, k, v, q)
        else:
            out = self.attention(k, v, q)
       
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out.reshape(q_shape)


class Local3dAttentionTransformer(nn.Module):
    def __init__(self, *, data_shape, dim, num_classes, extents, depth, heads, dim_head, mlp_dim, dropout=.0):
        super().__init__()

        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, dim)

        # position embeddings
        self.pos_emb_s = nn.Embedding(data_shape[0], dim)
        self.pos_emb_h = nn.Embedding(data_shape[1], dim)
        self.pos_emb_w = nn.Embedding(data_shape[2], dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Local3dAttention(extents, dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def get_pos_embedding(self, batch_shape):
        _,s,h,w, = batch_shape

        device = self.pos_emb_s.weight.device
        indices = torch.arange(s*h*w, device=device).view(1, s, h, w)
        w_pos = indices % w
        h_pos = indices.div(w, rounding_mode='trunc') % h
        s_pos = indices.div(h * w, rounding_mode='trunc')

        return (self.pos_emb_s(s_pos.expand(batch_shape))
                + self.pos_emb_h(h_pos.expand(batch_shape))
                + self.pos_emb_w(w_pos.expand(batch_shape)))

    def forward(self, img_z):
        batch_shape = img_z.shape

        x = self.embedding(img_z)
        x = x + self.get_pos_embedding(batch_shape)

        for attn, ff in self.layers:
            x = attn(x, q=x) + x
            x = ff(x) + x

        return x


def test():
    device = torch.device('cuda', 0)
    n = Local3dAttentionTransformer(data_shape=(10,16,16), dim=128, num_classes=1000, extents=(2,2,2), depth=4, mlp_dim=256, heads=3, dim_head=64, dropout=.0)
    n = n.to(device)

    x = torch.randint(0, 99, (2,4,16,16), device=device)
    y = n.forward(x)
    y.mean().backward()
    print(y.size())
    

if __name__ == '__main__':
    test()
