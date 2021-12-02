import math
import argparse
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import wandb

from warmup_scheduler import GradualWarmupScheduler
from ema_training import ModelEmaV2
from train_ae import show_and_save, wandb_init, count_parameters, show_batch
from autoencoder import SomAutoEncoder
#from diffusion_model import SimpleDiffusionModel
from diffusion_model_unet import UNetDiffusionModel


s = 0.008  # see 3.2 in https://arxiv.org/abs/2102.09672
def alpha_from_t(t):
    return torch.cos((t+s)/(1+s) * math.pi*0.5) ** 2


@torch.no_grad()
def eval_model(opt, model, device, timesteps=1000, batch_size=32, trace_steps=20):
    model.eval()

    # start with Gaussian noise
    x0 = torch.zeros(batch_size, 2, 16, 16, device=device)
    x = torch.randn_like(x0)
    i = 0
    trace = []
    for step in range(timesteps):
        f = step / (timesteps-1)    # linear 0-1

        t = torch.ones(batch_size, 1, device=device) * (1-f)
        t_ = t.view(-1, 1, 1, 1) # Bx1x1x1
        alpha_ = alpha_from_t(t_)

        # prepare input
        noise = torch.randn_like(x0) * (1-alpha_).sqrt()
        if f > 0.25:
            x = x0 * alpha_.sqrt() + noise
        else:
            x = x0 + noise

        # perdict noise
        noise_estimate = model(x, t)

        # denoise batch
        x0 = x + noise_estimate

        # undo alpha scaling
        if f > 0.25:
            x0 = x0 / alpha_.sqrt()

        # clip denoised version
        #x0 = x0.clamp(-2.5, 2.5)

        if f >= i / (trace_steps-1):
            while f >= i / (trace_steps-1):
                i += 1
            trace.append(x0)
            print('{:.1f} (x0 norm: {:.2f}; noise norm: {:.2f})'.format(f * 100, x0.view(x0.size(0), -1).norm(dim=1).mean().item(), noise_estimate.view(x0.size(0), -1).norm(dim=1).mean().item() ) )

    return trace


@torch.no_grad()
def grad_norm(model_params):
    sqsum = 0.0
    for p in model_params:
        sqsum += (p.grad ** 2).sum().item()
    return math.sqrt(sqsum)


def  train(opt, model, loss_fn, device, dataset, optimizer, lr_scheduler, decoder_model):
   
    batch_size = opt.batch_size
    experiment_name = opt.name
    checkpoint_interval = opt.checkpoint_interval
    eval_interval = opt.eval_interval

    train_offset = dataset.size(0)
    max_steps = opt.max_steps
    epoch = 0
    data_scaling = 2.5

    model_ema =  ModelEmaV2(model, decay=opt.ema_decay) if opt.ema_decay > 0 else None
    for step in range(1, max_steps+1):
        model.train()

        # batch_indices
        if train_offset + batch_size > dataset.size(0):
            print("Epoch: {}".format(epoch))
            train_indices = torch.randperm(dataset.size(0))
            train_offset = 0
            epoch = epoch + 1
        
        # load batch
        batch_indices = train_indices[train_offset:train_offset+batch_size]
        if not opt.single_batch:
            train_offset += batch_size

        batch = dataset[batch_indices]
        #show_batch(decoder_model.decode_2d(batch))
        batch = batch * data_scaling
        batch = batch.to(device)

        t = torch.rand(batch_size, 1, device=device)
        #t = (1 - t).sqrt() # train lower t with higher probability

        t_ = t.view(-1, 1, 1, 1)

        alpha_ = alpha_from_t(t_)

        noise = torch.randn_like(batch)
        noise = noise * (1 - alpha_).sqrt()
        batch = batch * alpha_.sqrt() + noise

        y = model(batch, t)
        loss = loss_fn(y, -noise) # / batch.size(0)

        optimizer.zero_grad()

        loss.backward()
        gn = grad_norm(model.parameters())
        optimizer.step()
        lr_scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        wandb.log({'loss': loss.item(), 'lr': lr_scheduler.get_last_lr()[0], 'grand_norm': gn})

        print('{}: Loss: {:.3e}; lr: {:.3e}; grad_norm: {:.3e}; epoch: {}'.format(step, loss.item(), lr_scheduler.get_last_lr()[0], gn, epoch))

        if step % checkpoint_interval == 0:
            # write model_checkpoint
            fn = '{}_checkpoint_{:07d}.pth'.format(experiment_name, step)
            print('writing file: ' + fn)
            torch.save({
                'step': step,
                'lr': lr_scheduler.get_last_lr(),
                'model_state_dict': model_ema.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'opt': opt,
            }, fn)

        if step % eval_interval == 0:
            model_ = model_ema.module if model_ema is not None else model
            trace = eval_model(opt, model_, device, opt.eval_timesteps, opt.eval_batch_size)
            eval_latent = torch.cat(trace, dim=0)
            eval_latent = eval_latent / data_scaling
            eval_decode = decoder_model.decode_2d(eval_latent.cpu())
            # log result to wandb
            img_grid = torchvision.utils.make_grid(eval_decode.detach().cpu(), nrow=opt.eval_batch_size)
            images = wandb.Image(img_grid, caption="Sampling Result")
            wandb.log({"sampling": images})
            fn = '{}_sampling_{:07d}.png'.format(opt.name, step)
            show_and_save(fn, img_grid, show=False, save=True)
            #show_batch(eval_decode, nrow=opt.eval_batch_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='device to use')
    parser.add_argument('--device-index', default=0, type=int, help='device index')
    parser.add_argument('--manual_seed', default=42, type=int, help='initialization of pseudo-RNG')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--optimizer', default='AdamW', type=str, help='Optimizer to use (Adam, AdamW)')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--loss_fn', default='MSE', type=str)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--eval_interval', default=1000, type=int)
    parser.add_argument('--eval_timesteps', default=500, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)

    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--d_model', default=256, type=int)
    parser.add_argument('--d_pos', default=32, type=int, help='size of timestep encoding')
    parser.add_argument('--num_layers', default=8, type=int)

    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--entity', default='andreaskoepf', type=str)
    parser.add_argument('--tags', default=None, type=str)
    parser.add_argument('--project', default='som-diffusion-diffusion', type=str, help='project name for wandb')
    parser.add_argument('--name', default='diffusion_' + uuid.uuid4().hex, type=str, help='wandb experiment name')

    parser.add_argument('--input_dataset', default='experiments/ds2/diffusion_input_1k.pth', type=str)
    parser.add_argument('--decoder_model', default='experiments/ds2/som_ds2_8k_1_checkpoint_0040000.pth', type=str)
    parser.add_argument('--warmup', default=500, type=int)
    parser.add_argument('--max_steps', default=200 * 1000, type=int)
    parser.add_argument('--single_batch', default=False, action='store_true')
    parser.add_argument('--ema_decay', default=0.9999, type=float, help='ema decay of shadow model')

    opt = parser.parse_args()
    return opt


def main():
    print('Using pytorch version {}'.format(torch.__version__))

    opt = parse_args()
    print('Options:', opt)

    device = torch.device(opt.device, opt.device_index)
    print('Device:', device)

    torch.manual_seed(opt.manual_seed)

    wandb_init(opt)

    learning_rate = opt.lr

    # load latent embeding dataset (generated by create_diffusion_dataset.py)
    data_file = torch.load(opt.input_dataset, map_location=torch.device('cpu'))
    dataset = data_file['data']
    print('Loaded dataset {}, with {} examples, latent dim: {}.'.format(opt.input_dataset, dataset.size(0), dataset[0].size()))

    # load decoder model (cpu)
    print('Loading decoder_model: {}'.format(opt.decoder_model))
    decoder_data = torch.load(opt.decoder_model, map_location=torch.device('cpu'))
    chkpt_opt = decoder_data['opt']
    decoder_model = SomAutoEncoder(embedding_dim=chkpt_opt.embedding_dim, downscale_steps=chkpt_opt.downscale_steps, pass_through_som=True)
    decoder_model.load_state_dict(decoder_data['model_state_dict'])
    print('ok')

    """
    # verify decoding
    y = decoder_model.decode_2d(dataset[0:64])
    print('y', y.size())
    show_and_save('test.png', y)
    """

    # create gaussian diffusion model
    #model = SimpleDiffusionModel(d_model=opt.d_model, dropout=opt.dropout, num_layers=opt.num_layers, d_pos=32)
    model = UNetDiffusionModel(in_channels=2, out_channels=2, model_channels=128, num_res_blocks=3, channel_mult=(1,2,3), dropout=opt.dropout)
    model.to(device)

    count_parameters(model)

    if opt.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=opt.weight_decay, amsgrad=False)
    elif opt.optimizer == 'Adam':
        if opt.weight_decay > 0:
            print('WARN: AAdam with weight_decay > 0')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=opt.weight_decay, amsgrad=False)
    else:
        raise RuntimeError('Unsupported optimizer specified.')


    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_steps)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=opt.warmup, after_scheduler=scheduler_cosine)


    if opt.loss_fn == 'SmoothL1':
        loss_fn = torch.nn.SmoothL1Loss()
    elif opt.loss_fn == 'MSE':
        loss_fn = torch.nn.MSELoss()
    elif opt.loss_fn == 'MAE' or opt.loss_fn == 'L1':
        loss_fn = torch.nn.L1Loss()
    else:
        raise RuntimeError('Unsupported loss function type specified.')

    train(opt, model, loss_fn, device, dataset, optimizer, lr_scheduler, decoder_model)


if __name__ == '__main__':
    main()
