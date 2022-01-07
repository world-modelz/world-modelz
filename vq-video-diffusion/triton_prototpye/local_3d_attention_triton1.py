# adapted from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import checkpoint

from einops import rearrange, repeat

import triton
import triton.language as tl

import time


@triton.jit
def blub_kernel(
    q_ptr, k_ptr, c_ptr, 
    M, D,           # M: num_queries, D: d_model
    B, S, H, W,     # 3D key dimensions: time, height, width
    kS, kH, kW,     # 3D kernel sizes
    stride_qm, stride_qd,   # query strides
    stride_kB, stride_kS, stride_kH, stride_kW, stride_k_ks, stride_k_kh, stride_k_kw, stride_kd,
    stride_cm, stride_cw,   # output strides
    **meta
):
    """
     C = Q x K
     Q: (M, D)
     K: ... it's complicated
     C: (M, W)
    """
    BLOCK_SIZE_M = meta['BLOCK_SIZE_M']     # num queries we process at once
    BLOCK_SIZE_D = meta['BLOCK_SIZE_D']     # num elements of embedding we process at once

    pid = tl.program_id(axis=0)
    wnd = kS * kH * kW
    base_m = (pid // wnd) * BLOCK_SIZE_M
    base_w = pid % wnd

    # current programs key input coordinate
    base_ws = tl.arange(0, BLOCK_SIZE_M) + base_m
    b = base_ws // (W * H * S)
    z = base_ws // (W * H) % S
    y = (base_ws // W) % H
    x = base_ws % W

    s = base_w // (kH * kW)
    h = (base_w // kW) % kH
    w = base_w % kW

    # compute source key pointers
    offs_k = b * stride_kB + z * stride_kS + y * stride_kH + x * stride_kW + w * stride_kW + h * stride_kH + s * stride_kS
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    offs_q = base_m + tl.arange(0, BLOCK_SIZE_M)
    q_ptrs = q_ptr + offs_q[:, None] * stride_qm + offs_d[None, :] * stride_qd     # (M, D)
    k_ptrs = k_ptr + offs_k[:, None] + offs_d[None, :] * stride_kd                 # (M, D)

    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for d in range(0, D, BLOCK_SIZE_D):
        q = tl.load(q_ptrs)     # (BLOCK_SIZE_M, BLOCK_SIZE_D)
        k = tl.load(k_ptrs)     # (BLOCK_SIZE_M, BLOCK_SIZE_D)
        accumulator += tl.sum(q * k, axis=1)
        q_ptrs += BLOCK_SIZE_D * stride_qd
        k_ptrs += BLOCK_SIZE_D * stride_kd
    
    # write result
    offs_cm =  base_m + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + offs_cm * stride_cm + base_w * stride_cw
    c_mask = offs_cm < M
    tl.store(c_ptrs, accumulator, mask=c_mask)


def blub(q, k):
    B, S, H, W, D, kS, kH, kW = k.shape    
    M,D = q.shape

    # allocate output tensor
    window_size = kS * kH * kW
    c = torch.zeros(M, window_size, device=q.device, dtype=q.dtype)

    stride_qm,stride_qd = q.stride()
    stride_kB,stride_kS,stride_kH,stride_kW,stride_kd,stride_k_ks,stride_k_kh,stride_k_kw = k.stride()
    stride_cm,stride_cw = c.stride()
    #print('c.stride()', c.stride())

    # grid based on output elements (number of queries times local windows size)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * window_size,     # cdiv = ceil_div
    )

    blub_kernel[grid](
        q, k, c, 
        M, D,
        B, S, H, W,     # 3D key dimensions: frame, width, height
        kS, kH, kW,     # 3D kernel sizes
        stride_qm, stride_qd,   # query strides
        stride_kB, stride_kS, stride_kH, stride_kW, stride_k_ks, stride_k_kh, stride_k_kw, stride_kd,
        stride_cm, stride_cw,   # output strides
        BLOCK_SIZE_M=64,    # TODO: tuning
        BLOCK_SIZE_D=64,
    )

    return c


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
    def __init__(self, extents, dim, heads=8, dim_head=64, dropout=.0, use_checkpointing=True, use_triton=True):
        super().__init__()

        self.extents = extents
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.use_checkpointing = use_checkpointing
        self.use_triton = use_triton

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

    def local_attention(self, k, v, q):
        batch_size = v.shape[0]
        mask = self.get_mask(k.shape).to(k.device)

        k = self.unfold(self.pad(k))    # pad border cases to get equal sizes
        v = self.unfold(self.pad(v))

        # print('k', k.size(), k.stride(), k.numel(), k.storage().size())
        # print('v', v.size(), v.stride(), v.numel(), v.storage().size())
        # print('q', q.size(), q.stride(), q.numel(), q.storage().size())
 
        if self.heads == 1 and self.use_triton:
            #print('triton')
            dots = blub(q.view(-1, q.size(-1)), k) * self.scale
            dots = dots.unsqueeze(-2).unsqueeze(-2)
        else:
            q = rearrange(q, 'b s h w (H d) -> (b s h w) H 1 d', H = self.heads)
            k = rearrange(k, 'b s h w (H d) i j k -> (b s h w) H (i j k) d', H = self.heads)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        v = rearrange(v, 'b s h w (H d) i j k -> (b s h w) H (i j k) d', H = self.heads)
        # q = rearrange(q, 'b s h w (H d) -> (b s h w) H 1 d', H = self.heads)
        # v = rearrange(v, 'b s h w (H d) i j k -> (b s h w) H (i j k) d', H = self.heads)
        # k = rearrange(k, 'b s h w (H d) i j k -> (b s h w) H (i j k) d', H = self.heads)
        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

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
            out = checkpoint.checkpoint(self.local_attention, k, v, q)
        else:
            out = self.local_attention(k, v, q)
       
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out.reshape(q_shape)


class Local3dAttentionTransformer(nn.Module):
    def __init__(self, *, data_shape, dim, num_classes, extents, depth, heads, dim_head, mlp_dim, dropout=.0, use_triton=True):
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
                PreNorm(dim, Local3dAttention(extents, dim, heads=heads, dim_head=dim_head, dropout=dropout, use_triton=use_triton)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
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


@torch.no_grad()
def run_test(device, w, x, n, use_triton=True):
    net = Local3dAttentionTransformer(data_shape=(10,16,16), dim=128, num_classes=1000, extents=(2,2,2), depth=10, mlp_dim=256, heads=1, dim_head=128, dropout=.0, use_triton=use_triton)
    net.load_state_dict(w)
    net = net.to(device)

    for i in range(n):
        y = net.forward(x)
        print(i, y.size(), torch.cuda.max_memory_allocated(device))
    return y


def get_weights():
    net = Local3dAttentionTransformer(data_shape=(10,16,16), dim=128, num_classes=1000, extents=(2,2,2), depth=10, mlp_dim=256, heads=1, dim_head=128, dropout=.0)
    return net.state_dict() 


@torch.no_grad()
def test():
    device = torch.device('cuda', 0)
    
    w = get_weights()   # generate state dict to use for tests
    
    x = torch.randint(0, 99, (10,6,16,16), device=device)
    
    torch.cuda.empty_cache()
    tic = time.time_ns()
    run_test(device, w, x, 10, use_triton=True)
    toc = time.time_ns()
    time_with = toc-tic
    print('with triton: {}ms'.format(time_with/(1e6)))

    torch.cuda.empty_cache()
    tic = time.time_ns()
    run_test(device, w, x, 10, use_triton=False)
    toc = time.time_ns()
    time_without = toc-tic
    print('without triton: {}ms'.format(time_without/(1e6)))
    print('ratio:', time_without/time_with)
    
    # simple output comparison
    a = run_test(device, w, x, 1, use_triton=True)
    b = run_test(device, w, x, 1, use_triton=False)
    print('diff:', torch.abs(a-b).sum(), a.numel(), a.std())


if __name__ == '__main__':
    test()
