from concurrent.futures import thread
import math
import random
import argparse
import uuid
import threading

#import matplotlib.pyplot as plt
import wandb
import numpy as np
import minerl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from train_vqae import VqAutoEncoder, wandb_init, count_parameters, show_batch

from local_3d_attention import Local3dAttentionTransformer
from warmup_scheduler import GradualWarmupScheduler
from model_ema_v2 import ModelEmaV2
from importance_sampling import LossAwareSamplerEma


class VqVideoDiffusionModel(nn.Module):
    def __init__(self, *, data_shape, dim, num_classes, extents, depth, dim_head, mlp_dim, heads=1, dropout=.0):
        super().__init__()

        num_classes_in = num_classes + 1    # add mask
        self.transformer = Local3dAttentionTransformer(data_shape=data_shape, dim=dim, num_classes=num_classes_in, extents=extents, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
        self.logit_proj = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        last_frames = x[:, -1]
        return self.logit_proj(last_frames)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k, largest=True, sorted=True)
    out = logits.clone()
    out[out < v[:,[-1]]] = -float('Inf')
    return out


def clamp(x, lo, hi):
    return min(max(x, lo), hi)


def transform_batch(b):
    x = torch.from_numpy(b)
    x = x.permute(0,1,4,2,3).contiguous()   # HWC -> CHW
    if isinstance(x, torch.ByteTensor):
        x = x.to(dtype=torch.float32).div(255)
    return x


@torch.no_grad()
def evaluate_model(*, device, model, decoder_model, num_embeddings, mask_token_index, batch_size, num_steps, batch_sampler, sample_topk=-1):

    # get context frames from dataset as during training
    batch = batch_sampler.sample_batch(batch_size)
    batch = transform_batch(batch)
    image_width = batch.shape[-1]

    batch = batch.to(device)
    print('batch', batch.size())
    batch_z = decoder_model.encode(batch.view(-1, 3, image_width, image_width))

    # check batch
    #reconstructed = decoder_model.decode(batch_z)
    #show_batch(reconstructed)

    batch_z = batch_z.view(-1, batch.size(1), batch_z.size(1), batch_z.size(2)) # NxSxHxW
    batch_z[:,-1] = mask_token_index # destroy all information in last frames
    w = batch_z.size(-1)

    generated_frames = [batch[:,-1].view(-1, 3, image_width, image_width).cpu().clone()]

    num_eval_iterations = 30
    noise_schedule = None
    consistent_masking = False

    for i in range(num_steps):
        print('Sampling frame {}/{}'.format(i+1, num_steps))

        # sample next frame
        logits = torch.zeros(batch_size, w*w, num_embeddings, device=device)    # start with equal weighted probabilities (flat)
        logits_shape = logits.shape
        last_mask = torch.zeros(batch_size, w, w) == 0

        for i in range(num_eval_iterations):
            logits = logits.view(-1, num_embeddings)

             # sample
            if sample_topk > 0 and i > 0:
                logits = top_k_logits(logits, sample_topk)

            p = F.softmax(logits, dim=-1)
            denoised_samples = torch.multinomial(p, 1, True).view(batch_size, w, w)

            frac = (i+1)/num_eval_iterations
            if noise_schedule is not None:
                alpha = noise_schedule(frac)
            else:
                alpha = frac

            alpha = clamp(alpha, 0, 1)
            if consistent_masking:
                mask = last_mask & (torch.rand(batch_size, w, w) > alpha)
                last_mask = mask
            else:
                mask = (torch.rand(batch_size, w, w) > alpha)

            #print('denoised_samples', denoised_samples.size())
            #print('batch_z', batch_z.size())
            #print('alpha', alpha)
            #print('mask_count', mask.count_nonzero())

            denoised_samples[mask] = mask_token_index
            batch_z[:,-1,:,:] = denoised_samples
        
            logits = model.forward(batch_z)

        dec_denoised = decoder_model.decode(denoised_samples)
        generated_frames.append(dec_denoised.cpu())
        batch_z[:,:-1] = batch_z[:,1:] # shift frames
        batch_z[:,-1] = mask_token_index

    return torch.cat(generated_frames, dim=0), generated_frames


@torch.no_grad()
def eval_model_and_save(*, save_combined=True, combined_fn='eval_result.png', save_frames=False, frames_fn='frame_{:04d}.png', batch_size=8, **kwargs):
    eval_result, eval_frames = evaluate_model(batch_size=batch_size, **kwargs)
    img_grid = torchvision.utils.make_grid(eval_result, nrow=batch_size, pad_value=0.2)
    if save_combined:
        torchvision.utils.save_image(img_grid, combined_fn)

    if save_frames:
        for i,frame in enumerate(eval_frames):
            fn = frames_fn.format(i)
            torchvision.utils.save_image(frame, fn)

    return img_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='device to use')
    parser.add_argument('--device-index', default=0, type=int, help='device index')
    parser.add_argument('--manual_seed', default=42, type=int, help='initialization of pseudo-RNG')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--optimizer', default='AdamW', type=str, help='Optimizer to use (Adam, AdamW)')
    parser.add_argument('--weight_decay', default=1e-7, type=float)
    parser.add_argument('--ema_decay', default=0, type=float, help='ema decay of shadow model, e.g. 0.999 or 0.9999')

    parser.add_argument('--data_dir', default='/mnt/minerl/', type=str, help='root directory for data')
    parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
    parser.add_argument('--decoder_model', default='mcvq7_1k_checkpoint_0010000.pth', type=str)

    parser.add_argument('--max_steps', default=200 * 1000, type=int)
    parser.add_argument('--warmup', default=500, type=int)
    parser.add_argument('--accumulation_steps', default=1, type=int)
    parser.add_argument('--checkpoint_interval', default=25000, type=int)
    parser.add_argument('--eval_interval', default=2000, type=int)
    parser.add_argument('--eval_timesteps', default=4, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)

    # wandb
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--entity', default='andreaskoepf', type=str)
    parser.add_argument('--tags', default=None, type=str)
    parser.add_argument('--project', default='mcvq-video-diffusion', type=str, help='project name for wandb')
    parser.add_argument('--name', default='mcvq_diffusion_' + uuid.uuid4().hex, type=str, help='wandb experiment name')

    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--topk', default=-1, type=int)

    # model params
    parser.add_argument('--dim', default=256, type=int)
    parser.add_argument('--extents', default='3,3,3', type=str, help='S,H,W, e.g. "3,3,3"')
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--mlp_dim', default=256, type=int)
    parser.add_argument('--dim_head', default=128, type=int)
    parser.add_argument('--heads', default=1, type=int)
    parser.add_argument('--dropout', default=0, type=float)

    # batch sampler
    parser.add_argument('--buffer_size', default=100000, type=int, help='num frames in batch sampler buffer')
    parser.add_argument('--skip_frames', default=2, type=int)

    opt = parser.parse_args()
    return opt


@torch.no_grad()
def grad_norm(model_params):
    sqsum = 0.0
    for p in model_params:
        sqsum += (p.grad ** 2).sum().item()
    return math.sqrt(sqsum)


def train(opt, model, loss_fn, device, batch_sampler, optimizer, lr_scheduler, decoder_model, image_width=64):

    batch_size = opt.batch_size
    experiment_name = opt.name
    checkpoint_interval = opt.checkpoint_interval
    eval_interval = opt.eval_interval

    max_steps = opt.max_steps
    acc_steps = opt.accumulation_steps
    epoch = 0

    num_embeddings = decoder_model.vq.num_embeddings

    p_max_uniform = 0.1
    mask_token_index = num_embeddings

    sampler = LossAwareSamplerEma(num_histogram_buckets=100, uniform_p=0.01, alpha=0.9, warmup=10)

    model_ema = ModelEmaV2(model, decay=opt.ema_decay) if opt.ema_decay > 0 else None
    for step in range(1, max_steps+1):
        model.train()

        optimizer.zero_grad()
        loss_sum = 0
        for acc_step in range(acc_steps):

            # fill batch
            batch = batch_sampler.sample_batch(batch_size)
            batch = transform_batch(batch)
            image_width = batch.shape[-1]

            with torch.no_grad():
                batch = batch.to(device)
                batch_z = decoder_model.encode(batch.view(-1, 3, image_width, image_width))
                batch_z = batch_z.view(-1, batch.size(1), batch_z.size(1), batch_z.size(2))

            batch_z = batch_z.to(device)
            last_frame = batch_z[:, -1]
            target = last_frame.clone()

            encoding = last_frame.reshape(batch_size, -1)

            # add noise and masking
            r = sampler.sample(batch_size).view(batch_size, 1).to(device)

            threshold = r.to(device)
            mask = torch.rand(batch_size, encoding.size(1), device=device) < threshold

            du = torch.ones(batch_size, encoding.size(1), num_embeddings, device=device) / num_embeddings
            dt = F.one_hot(encoding, num_classes=num_embeddings).float().to(device)
            d = torch.lerp(dt, du, r.unsqueeze(-1) * p_max_uniform)

            draw = torch.multinomial(d.view(-1, num_embeddings), num_samples=1)
            draw = draw.view(batch_size, -1)

            draw[mask] = mask_token_index
            batch_z[:, -1] = draw.view(last_frame.shape)

            # inspect batch
            #batch_z[batch_z >= num_embeddings] = 0
            #dec = decoder_model.decode(batch_z.view(-1, batch_z.shape[-2], batch_z.shape[-1]))
            #show_batch(dec, nrow=opt.n_past+1)

            y = model.forward(batch_z)

            loss = loss_fn(y.reshape(-1, num_embeddings), target.reshape(-1))

            # compute per sample loss
            per_sample_loss = loss.view(batch_size, -1).mean(dim=1)
            sampler.update_with_losses(r, per_sample_loss)

            loss = loss.mean()
            if acc_steps > 1:
                loss = loss / acc_steps

            loss.backward()

            loss_sum += loss.item()

        gn = grad_norm(model.parameters())
        optimizer.step()
        lr_scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        wandb.log({'loss': loss_sum, 'lr': lr_scheduler.get_last_lr()[0], 'grad_norm': gn})

        print('{}: Loss: {:.3e}; lr: {:.3e}; grad_norm: {:.3e}; epoch: {}; warmed_up: {}'.format(step, loss_sum, lr_scheduler.get_last_lr()[0], gn, epoch, sampler.warmed_up()))

        if step % 50 == 0:
            print('sampler.weights(): ', sampler.weights())
            wandb.log({ 'sampler_weights': wandb.Histogram(np_histogram=sampler.weights_as_numpy_histogram()) })

        if step % checkpoint_interval == 0:
            # write model_checkpoint
            fn = '{}_checkpoint_{:07d}.pth'.format(experiment_name, step)
            print('writing file: ' + fn)
            ema_state_dict = model_ema.module.state_dict() if model_ema is not None else None
            torch.save({
                'step': step,
                'lr': lr_scheduler.get_last_lr(),
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'opt': opt,
            }, fn)

        if step % eval_interval == 0:
            eval_models = [('base', model)]
            if model_ema is not None:
                eval_models.append(('ema', model_ema.module))

            for model_name, model_ in eval_models:
                print('eval ', step, model_name)
                fn = '{}_eval_{:07d}_{}.png'.format(experiment_name, step, model_name)
                img_grid = eval_model_and_save(
                    save_combined=True,
                    combined_fn=fn,
                    save_frames=False,
                    device=device,
                    model=model_,
                    decoder_model=decoder_model,
                    num_embeddings=num_embeddings,
                    mask_token_index=mask_token_index,
                    batch_size=opt.eval_batch_size,
                    num_steps=opt.eval_timesteps,
                    batch_sampler=batch_sampler
                )
                images = wandb.Image(img_grid, caption='Reconstruction ' + model_name)
                wandb.log({'reconstruction_' + model_name: images})


environment_names = [
    # 'MineRLBasaltBuildVillageHouse-v0',
    # 'MineRLBasaltCreatePlainsAnimalPen-v0',
    # 'MineRLBasaltCreateVillageAnimalPen-v0',
    # 'MineRLBasaltFindCave-v0',
    # 'MineRLBasaltMakeWaterfall-v0',
    # 'MineRLNavigateDense-v0',
    # 'MineRLNavigateExtremeDense-v0',
    # 'MineRLNavigateExtreme-v0',
    # 'MineRLNavigate-v0',
    # 'MineRLObtainDiamondDense-v0',
    # 'MineRLObtainDiamond-v0',
    # 'MineRLObtainIronPickaxeDense-v0',
    # 'MineRLObtainIronPickaxe-v0',
    'MineRLTreechop-v0',
]


class BufferedTrajSampler:
    def __init__(self, environment_names, data_dir, buffer_size=100000, max_segment_length=1000, traj_len=16, skip_frames=2, frame_shape=(64,64,3)):
        self.environment_names = environment_names
        self.data_dir = data_dir
        self.frame_shape = frame_shape

        # build list of all trajectories
        traj_names = []
        for environment_name in environment_names:
            data = minerl.data.make(environment_name, data_dir=data_dir)
            traj_names.extend((environment_name, t) for t in data.get_trajectory_names())
        self.traj_names = traj_names

        self.buffer_size = buffer_size
        self.traj_len = traj_len
        self.skip_frames = skip_frames
        self.max_segment_length = max_segment_length
        self.example_offsets = []
        self.example_index = 0
        self.fill_thread = None

        self.start_fill_buffer()

    def fill_buffer_thread(self):
        
        total_frames = 0
        segments = []
        example_offsets = []

        order = [] 
        i = 0
        
        while total_frames < self.buffer_size:
            if i >= len(order):
                order = torch.randperm(len(self.traj_names)).tolist()
                i = 0

            environment_name, trajectory_name = self.traj_names[i]
            i += 1
            data = minerl.data.make(environment_name, data_dir=self.data_dir)
            traj_data = data.load_data(trajectory_name)

            # read whole trajectory into buffer
            frames = []
            skip = 0
            for data_tuple in traj_data:
                if skip > 0:
                    skip -= 1
                else:
                    obs = data_tuple[0]
                    pov = obs['pov']
                    frames.append(pov)
                    skip = self.skip_frames

            if len(frames) <= self.traj_len:
                continue

            #print('#frames:', len(frames))

            # select random segment of trajectory to keep
            max_offset = len(frames) - self.max_segment_length
            if max_offset > 0:
                begin = random.randint(0, max_offset)
                frames = frames[begin:begin+self.max_segment_length]

            segment_index = len(segments)
            segments.append(frames)
            total_frames += len(frames)

            # generate random offsets into segment as basis for training examples
            sample_divisor = 8
            for j in range((len(frames)-self.traj_len) // sample_divisor):   # sample depending on traj len
                offset = random.randint(0, len(frames)-self.traj_len)
                example_offsets.append((segment_index, offset))

            #print(f'total_frames: {total_frames}; examples: {len(example_offsets)}')

        self.next_segments = segments
        p = np.random.permutation(len(example_offsets))
        self.next_example_offsets = [example_offsets[k] for k in p]

    def start_fill_buffer(self):
        if self.fill_thread is None:
            self.fill_thread = threading.Thread(target=self.fill_buffer_thread, daemon=True)
            self.fill_thread.start()

    def wait_for_next_buffer(self):
        self.fill_thread.join()
        self.segments = self.next_segments
        self.example_offsets = self.next_example_offsets
        self.example_index = 0

        self.fill_thread = None
        self.start_fill_buffer()

    def sample_batch(self, batch_size):
        l = self.traj_len

        batch_shape = (batch_size, l) + self.frame_shape
        batch = np.ndarray(batch_shape, dtype=np.uint8)
        
        for i in range(batch_size):
            if self.example_index >= len(self.example_offsets):
                self.wait_for_next_buffer()
            
            segment, offset = self.example_offsets[self.example_index]
            self.example_index += 1
            batch[i] = self.segments[segment][offset:offset+l]
        
        return batch


def main():
    print('Using pytorch version {}'.format(torch.__version__))

    opt = parse_args()
    print('Options:', opt)

    device = torch.device(opt.device, opt.device_index)
    print('Device:', device)

    torch.manual_seed(opt.manual_seed)

    wandb_init(opt)

    # create data set
    batch_sampler = BufferedTrajSampler(environment_names, data_dir=opt.data_dir, buffer_size=opt.buffer_size, traj_len=opt.n_past+1, skip_frames=opt.skip_frames)

    # fill buffer, test sampler
    x = batch_sampler.sample_batch(1)
    #show_batch(x.permute(0,3,1,2))

    current_opt = opt

    # load model checkpoint
    model_checkpoint = opt.checkpoint
    checkpoint_data = None
    if model_checkpoint is not None:
        print('loading model checkpoint: ', model_checkpoint)
        checkpoint_data = torch.load(model_checkpoint, map_location=device)
        opt = checkpoint_data['opt']

    # load vq model (cpu)
    print('Loading decoder_model: {}'.format(opt.decoder_model))
    decoder_data = torch.load(opt.decoder_model, map_location=torch.device('cpu'))
    chkpt_opt = decoder_data['opt']
    decoder_model = VqAutoEncoder(chkpt_opt.embedding_dim, chkpt_opt.num_embeddings, chkpt_opt.downscale_steps, hidden_planes=chkpt_opt.hidden_planes, in_channels=3)
    decoder_model.load_state_dict(decoder_data['model_state_dict'])
    print('ok')

    #dummy_batch = torch.zeros(1, 1, opt.image_width, opt.image_width)
    x = transform_batch(x)[0]
    print('x', x.size())
    z = decoder_model.encode(x)

    print('z', z.size())
    x = decoder_model.decode(z)
    #show_batch(x)
    #quit()

    extents = [int(s) for s in opt.extents.split(',')]
    assert len(extents) == 3

    model = VqVideoDiffusionModel(
        data_shape=z.shape, # (6,8,8)
        dim=opt.dim,
        num_classes=chkpt_opt.num_embeddings,
        extents=extents,
        depth=opt.depth,
        mlp_dim=opt.mlp_dim,
        dim_head=opt.dim_head,
        heads=opt.heads,
        dropout=opt.dropout)

    count_parameters(model)

    model = model.to(device)
    decoder_model = decoder_model.to(device)

    if checkpoint_data is not None:
        model.load_state_dict(checkpoint_data['model_state_dict'])

    if current_opt.eval:
        eval_model_and_save(
            save_combined=True,
            combined_fn='eval_result.png',
            save_frames=True,
            frames_fn='frame_{:04d}.png',
            device=device,
            model=model,
            decoder_model=decoder_model,
            num_embeddings=chkpt_opt.num_embeddings,
            mask_token_index=chkpt_opt.num_embeddings,
            batch_size=current_opt.eval_batch_size,
            num_steps=current_opt.eval_timesteps,
            batch_sampler=batch_sampler,
            sample_topk=current_opt.topk
        )
        return

    if opt.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay, amsgrad=False)
    elif opt.optimizer == 'Adam':
        if opt.weight_decay > 0:
            print('WARN: Adam with weight_decay > 0')
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay, amsgrad=False)
    else:
        raise RuntimeError('Unsupported optimizer specified.')

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_steps)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=opt.warmup, after_scheduler=scheduler_cosine)

    loss_fn = nn.CrossEntropyLoss(reduction='none')

    train(opt, model, loss_fn, device, batch_sampler, optimizer, lr_scheduler, decoder_model)


if __name__ == '__main__':
    main()
