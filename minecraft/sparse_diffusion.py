import math
import argparse
from json import decoder
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from train_vqae import VqAutoEncoder, wandb_init, count_parameters, show_batch
from buffered_traj_sampler import BufferedTrajSampler
from transformer import Transformer
from importance_sampling import LossAwareSamplerEma
from warmup_scheduler import GradualWarmupScheduler


def transform_batch(b):
    x = torch.from_numpy(b)
    x = x.permute(0, 1, 4, 2, 3).contiguous()  # HWC -> CHW
    if isinstance(x, torch.ByteTensor):
        x = x.to(dtype=torch.float32).div(255)
    return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str, help="device to use")
    parser.add_argument("--device_index", default=0, type=int, help="device index")
    parser.add_argument("--manual_seed", default=42, type=int, help="initialization of pseudo-RNG")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=1e-7, type=float)
    parser.add_argument("--decoder_model", default="mcvq8_1k_checkpoint_0010000.pth", type=str)

    opt = parser.parse_args()
    return opt


def sample_flat_positions(batch_size, context_length, s, h, w, device):
    max_index = s * h * w
    n = batch_size * context_length
    p = torch.empty(n, device=device, dtype=torch.long)
    j = 0
    while j < n:
        r = torch.randperm(max_index, device=device)
        take = min(n - j, max_index)
        p[j : j + take] = r[:take]
        j += take
    return p.view(batch_size, context_length)


def sample_time_dependent(batch_size, context_length, s, h, w, t, device, o=None):
    """
    Sample positions in a neighborhood range along the time axis s dependent
    on the diffusion time t. For smaller values of t the range of frames from
    which positions are sampled is smaller than for larger values. As t approaches
    1 positions are sampled more and more from the whole video segment.
    Positions of individual frames are drawn uniformly (without replacement).
    """
    t = t.squeeze().clamp(0, 1).to(device)
    assert context_length > 0

    min_sample_window = math.ceil(context_length / (h*w))
    assert min_sample_window < s

    sample_window = torch.floor(min_sample_window + (t * (s - min_sample_window + 1)))
    sample_window = sample_window.clamp(max=s - min_sample_window)
    if o is None:
        o = torch.rand_like(t, device=device)
    else:
        o = o.clamp(0, 1-1e-5).to(device)
    offset = torch.floor(o * (s - sample_window + 1)).long()
    sample_window = sample_window.long() * h * w
    offset = offset * h * w

    # fill batch
    p = torch.empty(batch_size, context_length, device=device, dtype=torch.long)
    for i in range(batch_size):
        p[i] = torch.randperm(sample_window[i], device=device)[:context_length] + offset[i]
    return p


class VqSparseDiffusionModel(nn.Module):
    def __init__(self, *, shape, dim, num_classes, depth, dim_head, mlp_dim, heads=1, dropout=0.0):
        super().__init__()

        self.shape = shape

        # position embeddings
        S, H, W = shape
        self.pos_emb_s = nn.Embedding(S, dim)
        self.pos_emb_h = nn.Embedding(H, dim)
        self.pos_emb_w = nn.Embedding(W, dim)

        num_classes_in = num_classes + 1  # add mask embedding
        self.embedding = nn.Embedding(num_classes_in, dim)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.logit_proj = nn.Linear(dim, num_classes)

    def pos_embedding_3d(self, indices):
        S, H, W = self.shape
        w_pos = indices % W
        h_pos = indices.div(W, rounding_mode="trunc") % H
        s_pos = indices.div(H * W, rounding_mode="trunc")
        return self.pos_emb_s(s_pos) + self.pos_emb_h(h_pos) + self.pos_emb_w(w_pos)

    def forward(self, x, indices):
        x = self.embedding(x)
        x = x + self.pos_embedding_3d(indices)
        x = self.transformer(x)
        return self.logit_proj(x)


def clamp(x, lo, hi):
    return min(max(x, lo), hi)


@torch.no_grad()
def decode(decoder_model, batch, decode_N=16):
    batch = batch.clone()
    batch[batch >= decoder_model.vq.num_embeddings] = 0

    frames = []
    batch_shape = batch.shape
    batch_flat = batch.view(-1, batch_shape[2], batch_shape[3])

    n = batch_flat.shape[0]
    for i in range(0, n, decode_N):
        sub_batch = batch_flat[i : i + decode_N]
        frame = decoder_model.decode(sub_batch)
        frames.append(frame)

    frames = torch.cat(frames)
    frames = frames.view(batch_shape[0], -1, *frames.shape[1:])

    return frames


@torch.no_grad()
def evaluate_model(
    device,
    batch_size,
    model,
    decoder_model,
    shape,
    num_context=512,
    num_eval_iterations=100,
):
    num_embeddings = decoder_model.vq.num_embeddings
    mask_token_index = num_embeddings

    S, H, W = shape

    full_z = torch.empty(batch_size, S, H, W, device=device, dtype=torch.long)
    full_z.fill_(mask_token_index)
    full_z_flat = full_z.view(batch_size, -1)

    model.eval()
    for i in range(num_eval_iterations):
        max_index = S * H * W

        #all_indices_perm = torch.randperm(max_index, device=device).repeat(batch_size, 1)

        frac = i / (num_eval_iterations - 1)

        # sample offsets
        offset_count = max_index // num_context + 1
        offset_order = torch.randperm(offset_count)

        for k in range(offset_count):
            # sample input
            #j = k * max_index
            #indices = all_indices_perm[:, j : j + num_context]

            o = (offset_order[k].float() / (offset_count-1)).repeat(batch_size)
            indices = sample_time_dependent(batch_size, num_context, S, H, W, torch.ones(batch_size) * (1.0-frac), device, o=o)

            input = torch.gather(full_z_flat, dim=1, index=indices)

            alpha = frac
            alpha = clamp(alpha, 0, 1)
            mask = torch.rand_like(input, dtype=torch.float) > alpha
            input[mask] = mask_token_index

            # denoise
            logits = model.forward(input, indices).view(-1, num_embeddings)

            p = F.softmax(logits, dim=-1)
            denoised_samples = torch.multinomial(p, 1, True)  # B*num_context, 1
            denoised_samples = denoised_samples.view(input.shape)

            # write back to full_z_flat
            full_z_flat.scatter_(dim=1, index=indices, src=denoised_samples)

    decoded_frames = decode(decoder_model, full_z)
    return decoded_frames


def main():
    print("Using pytorch version {}".format(torch.__version__))

    opt = parse_args()
    print("Options:", opt)

    device = torch.device(opt.device, opt.device_index)
    print("Device:", device)

    torch.manual_seed(opt.manual_seed)

    # load vq model (cpu)
    print("Loading decoder_model: {}".format(opt.decoder_model))
    decoder_data = torch.load(opt.decoder_model, map_location=torch.device("cpu"))
    chkpt_opt = decoder_data["opt"]
    decoder_model = VqAutoEncoder(
        chkpt_opt.embedding_dim,
        chkpt_opt.num_embeddings,
        chkpt_opt.downscale_steps,
        hidden_planes=chkpt_opt.hidden_planes,
        in_channels=3,
    )
    decoder_model.load_state_dict(decoder_data["model_state_dict"])
    print("ok")

    mlr_data_dir = "/data/datasets/minerl"  # "/mnt/minerl"
    environment_names = ["MineRLTreechop-v0"]

    batch_size = 16 # 48
    max_steps = 100000
    warmup = 0 # 500
    # total size of a video segment to generate, e.g. 32 frames of 16x16 latent codes
    S, H, W = 32, 16, 16

    single_batch = True # False
    eval_interval = 1000

    num_embeddings = decoder_model.vq.num_embeddings
    mask_token_index = num_embeddings
    p_max_uniform = 0.1
    dim = 512
    mlp_dim = dim * 2
    heads = 4
    depth = 8
    num_context = 512  # number of tokens to pass through the transformer at once
    buffer_size = 5000 # 75000
    change_batch_interval = 4 # 32

    traj_sampler = BufferedTrajSampler(
        environment_names,
        mlr_data_dir,
        buffer_size=buffer_size,
        max_segment_length=1000,
        traj_len=S,
        skip_frames=2,
    )
    noise_sampler = LossAwareSamplerEma(num_histogram_buckets=100, uniform_p=0.01, alpha=0.9, warmup=10)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    model = VqSparseDiffusionModel(
        shape=(S, H, W),
        num_classes=num_embeddings,
        dim=dim,
        depth=depth,
        dim_head=dim // heads,
        mlp_dim=mlp_dim,
        heads=heads,
    )
    count_parameters(model)
    model.to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        betas=(0.9, 0.999),
        weight_decay=opt.weight_decay,
        amsgrad=False,
    )

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps)
    if warmup > 0:
        lr_scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1.0, total_epoch=warmup, after_scheduler=scheduler_cosine
        )
    else:
        lr_scheduler = scheduler_cosine

    decoder_model.to(device)

    for step in range(1, max_steps):
        model.train()
        optimizer.zero_grad()

        r = noise_sampler.sample(batch_size).view(batch_size, 1).to(device)

        indices = sample_time_dependent(batch_size, num_context, S, H, W, r, device)
        #indices = sample_flat_positions(batch_size, num_context, S, H, W, device)
        # print('indices', indices)

        if (not single_batch and step % change_batch_interval == 1) or step == 1:
            # quantize data
            with torch.no_grad():
                # load frames from mine rl-dataset
                batch = traj_sampler.sample_batch(batch_size)
                batch = transform_batch(batch)
                image_width = batch.shape[-1]

                batch_z = torch.zeros(batch_size * S, H, W, dtype=torch.long, device=device)
                batch_flat = batch.view(-1, 3, image_width, image_width)
                n = batch_flat.shape[0]
                encode_N = 16
                for i in range(0, n, encode_N):
                    batch_z[i : i + encode_N] = decoder_model.encode(batch_flat[i : i + encode_N].to(device))

                if single_batch:
                    gt = decode(decoder_model, batch_z.view(batch_size, S, H, W))
                    img_grid = torchvision.utils.make_grid(gt.view(-1, *gt.shape[2:]), nrow=S, pad_value=0.2)
                    torchvision.utils.save_image(img_grid, "gt.png")

        batch_z_flat = batch_z.view(batch_size, -1)
        input = torch.gather(batch_z_flat, dim=1, index=indices)

        # copy to GPU
        input = input.to(device)
        indices = indices.to(device)
        target = input.clone()

        # perturbation & masking
        threshold = r
        mask = torch.rand(input.shape, device=device) < threshold       # higher r -> more masking, r == 0 no masking

        du = torch.ones(batch_size, num_context, num_embeddings, device=device) / num_embeddings
        dt = F.one_hot(input, num_classes=num_embeddings).float().to(device)
        d = torch.lerp(dt, du, r.unsqueeze(-1) * p_max_uniform)

        input = torch.multinomial(d.view(-1, num_embeddings), num_samples=1).view(batch_size, -1)
        input[mask] = mask_token_index

        # denoise
        logits = model.forward(input, indices)

        # loss
        loss = loss_fn(logits.reshape(-1, num_embeddings), target.reshape(-1))

        # compute per batch item loss
        per_sample_loss = loss.view(batch_size, -1).mean(dim=1)
        noise_sampler.update_with_losses(r, per_sample_loss)

        loss = loss.mean()

        # backward
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if step % eval_interval == 0:
            decoded_frames = evaluate_model(
                device,
                8,
                model,
                decoder_model,
                shape=(S, H, W),
                num_context=num_context,
            )
            img_grid = torchvision.utils.make_grid(
                decoded_frames.view(-1, *decoded_frames.shape[2:]),
                nrow=S,
                pad_value=0.2,
            )
            combined_fn = f"combo_{step:08d}.png"
            torchvision.utils.save_image(img_grid, combined_fn)

        if step % 10 == 0:
            print(f"{step}: loss: {loss.item()}")


if __name__ == "__main__":
    # device = torch.device('cpu')
    # p = sample_time_dependent(4, 5, 10, 5, 5, torch.ones(4), device)
    # print(p)
    main()
