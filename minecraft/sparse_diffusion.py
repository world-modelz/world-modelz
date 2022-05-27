import argparse
from json import decoder
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from train_vqae import VqAutoEncoder, wandb_init, count_parameters, show_batch
from buffered_traj_sampler import BufferedTrajSampler
from transformer import Transformer
from importance_sampling import LossAwareSamplerEma


def transform_batch(b):
    x = torch.from_numpy(b)
    x = x.permute(0, 1, 4, 2, 3).contiguous()   # HWC -> CHW
    if isinstance(x, torch.ByteTensor):
        x = x.to(dtype=torch.float32).div(255)
    return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda',
                        type=str, help='device to use')
    parser.add_argument('--device_index', default=0,
                        type=int, help='device index')
    parser.add_argument('--manual_seed', default=42, type=int,
                        help='initialization of pseudo-RNG')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-7, type=float)
    parser.add_argument('--decoder_model',
                        default='mcvq8_1k_checkpoint_0010000.pth', type=str)

    opt = parser.parse_args()
    return opt


def sample_flat_positions(batch_size, context_length, s, h, w, device):
    max_index = s * h * w
    return torch.randint(max_index-1, (batch_size, context_length), device=device)


class VqSparseDiffusionModel(nn.Module):
    def __init__(self, *, shape, dim, num_classes, depth, dim_head, mlp_dim, heads=1, dropout=.0):
        super().__init__()

        self.shape = shape

        # position embeddings
        S, H, W = shape
        self.pos_emb_s = nn.Embedding(S, dim)
        self.pos_emb_h = nn.Embedding(H, dim)
        self.pos_emb_w = nn.Embedding(W, dim)

        num_classes_in = num_classes + 1    # add mask embedding
        self.embedding = nn.Embedding(num_classes_in, dim)

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
        self.logit_proj = nn.Linear(dim, num_classes)

    def pos_embedding_3d(self, indices):
        S, H, W = self.shape
        w_pos = indices % W
        h_pos = indices.div(W, rounding_mode='trunc') % H
        s_pos = indices.div(H * W, rounding_mode='trunc')
        return self.pos_emb_s(s_pos) + self.pos_emb_h(h_pos) + self.pos_emb_w(w_pos)

    def forward(self, x, indices):
        x = self.embedding(x)
        x = x + self.pos_embedding_3d(indices)
        x = self.transformer(x)
        return self.logit_proj(x)


def main():
    print('Using pytorch version {}'.format(torch.__version__))

    opt = parse_args()
    print('Options:', opt)

    device = torch.device(opt.device, opt.device_index)
    print('Device:', device)

    torch.manual_seed(opt.manual_seed)

    # load vq model (cpu)
    print('Loading decoder_model: {}'.format(opt.decoder_model))
    decoder_data = torch.load(
        opt.decoder_model, map_location=torch.device('cpu'))
    chkpt_opt = decoder_data['opt']
    decoder_model = VqAutoEncoder(chkpt_opt.embedding_dim, chkpt_opt.num_embeddings,
                                  chkpt_opt.downscale_steps, hidden_planes=chkpt_opt.hidden_planes, in_channels=3)
    decoder_model.load_state_dict(decoder_data['model_state_dict'])
    print('ok')

    mlr_data_dir = '/mnt/minerl'
    environment_names = ['MineRLTreechop-v0']

    batch_size = 32
    max_steps = 10000
    S, H, W = 32, 16, 16

    num_embeddings = decoder_model.vq.num_embeddings
    mask_token_index = num_embeddings
    p_max_uniform = 0.1
    dim = 512
    mlp_dim = dim*2
    heads = 4
    depth = 8
    num_context = 512

    traj_sampler = BufferedTrajSampler(
        environment_names, mlr_data_dir, buffer_size=5000, max_segment_length=2000, traj_len=S, skip_frames=2)
    noise_sampler = LossAwareSamplerEma(
        num_histogram_buckets=100, uniform_p=0.01, alpha=0.9, warmup=10)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    model = VqSparseDiffusionModel(shape=(S, H, W), num_classes=num_embeddings,
                                   dim=dim, depth=depth, dim_head=dim//heads, mlp_dim=mlp_dim, heads=heads)
    count_parameters(model)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, betas=(
        0.9, 0.999), weight_decay=opt.weight_decay, amsgrad=False)

    # batch_z = batch_z.view(-1, batch.size(1), batch_z.size(1), batch_z.size(2)) # NxSxHxW
    #latent_shape = batch_z.shape
    #print('latent_shape', latent_shape)

    decoder_model.to(device)

    for step in range(1, max_steps):
        model.train()
        optimizer.zero_grad()

        indices = sample_flat_positions(
            batch_size, num_context, S, H, W, device)
        #print('indices', indices)

        if step % 32 == 1:
            # quantize data
            with torch.no_grad():
                # load frames from mine rl-dataset
                batch = traj_sampler.sample_batch(batch_size)
                batch = transform_batch(batch)
                image_width = batch.shape[-1]

                batch_z = torch.zeros(
                    batch_size * S, H, W, dtype=torch.long, device=device)
                batch_flat = batch.view(-1, 3, image_width, image_width)
                n = batch_flat.shape[0]
                encode_N = 16
                for i in range(0, n, encode_N):
                    batch_z[i:i+encode_N] = decoder_model.encode(
                        batch_flat[i:i+encode_N].to(device))

        batch_z_flat = batch_z.view(batch_size, -1)
        input = torch.gather(batch_z_flat, dim=1, index=indices)

        # copy to GPU
        input = input.to(device)
        indices = indices.to(device)
        target = input.clone()

        # add noise and masking
        r = noise_sampler.sample(batch_size).view(batch_size, 1).to(device)

        # perturbation & masking
        threshold = r
        mask = torch.rand(input.shape, device=device) < threshold

        du = torch.ones(batch_size, num_context, num_embeddings,
                        device=device) / num_embeddings
        dt = F.one_hot(input, num_classes=num_embeddings).float().to(device)
        d = torch.lerp(dt, du, r.unsqueeze(-1) * p_max_uniform)

        input = torch.multinomial(
            d.view(-1, num_embeddings), num_samples=1).view(batch_size, -1)
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

        print(f'{step}: loss: {loss.item()}')


if __name__ == '__main__':
    main()
