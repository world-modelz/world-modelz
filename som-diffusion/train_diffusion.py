import argparse
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from warmup_scheduler import GradualWarmupScheduler
from train_ae import wandb_init, count_parameters
from diffusion_model import SimpleDiffusionModel

def  train(opt, model, loss_fn, device, dataset, optimizer, lr_scheduler):
   
    batch_size = opt.batch_size
    experiment_name = opt.name
    checkpoint_interval = opt.checkpoint_interval

    train_offset = dataset.size(0)
    max_steps = opt.max_steps
    epoch = 0
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
        train_offset += batch_size

        batch = dataset[batch_indices]  

        batch = batch.to(device)

        t = torch.rand(batch_size, 1, device=device)
        noise = torch.randn_like(batch)

        t_ = t.view(-1, 1, 1, 1)
        batch = batch * (1.0 - t_).sqrt() + noise * t_

        y = model(batch, t)

        loss = loss_fn(y, -noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        wandb.log({'loss': loss.item(), 'lr': lr_scheduler.get_last_lr()[0]})

        print('{}: Loss: {:.3e}; lr: {:.3e}; epoch: {}'.format(step, loss.item(), lr_scheduler.get_last_lr()[0], epoch))

        if step % checkpoint_interval == 0:
            # write model_checkpoint
            fn = '{}_checkpoint_{:07d}.pth'.format(experiment_name, step)
            print('writing file: ' + fn)
            torch.save({
                'step': step,
                'lr': lr_scheduler.get_last_lr(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'opt': opt,
            }, fn)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='device to use')
    parser.add_argument('--device-index', default=1, type=int, help='device index')
    parser.add_argument('--manual_seed', default=42, type=int, help='initialization of pseudo-RNG')
    parser.add_argument('--batch_size', default=96, type=int, help='batch size')
    parser.add_argument('--optimizer', default='AdamW', type=str, help='Optimizer to use (Adam, AdamW)')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--loss_fn', default='SmoothL1', type=str)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)

    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--entity', default='andreaskoepf', type=str)
    parser.add_argument('--tags', default=None, type=str)
    parser.add_argument('--project', default='som-diffusion-diffusion', type=str, help='project name for wandb')
    parser.add_argument('--name', default='diffusion_' + uuid.uuid4().hex, type=str, help='wandb experiment name')

    parser.add_argument('--input_dataset', default='experiments/ds2/diffusion_input_1k.pth', type=str)
    parser.add_argument('--warmup', default=500, type=int)
    parser.add_argument('--max_steps', default=200 * 1000, type=int)
    

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

    data_file = torch.load(opt.input_dataset, map_location=torch.device('cpu'))
    dataset = data_file['data']

    print('Loadadd dataset {}, with {} examples, latent dim: {}.'.format(opt.input_dataset, dataset.size(0), dataset[0].size()))


    model = SimpleDiffusionModel(d_model=256, dropout=.1, num_layers=8, d_pos=32)
    model.to(device)

    count_parameters(model)

    if opt.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=False)
    elif opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    else:
        raise RuntimeError('Unsupported optimizer specified.')


    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_steps)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=opt.warmup, after_scheduler=scheduler_cosine)


    if opt.loss_fn == 'SmoothL1':
        loss_fn = torch.nn.SmoothL1Loss(reduction='sum')
    elif opt.loss_fn == 'MSE':
        loss_fn = torch.nn.MSELoss(reduction='sum')
    elif opt.loss_fn == 'MAE' or opt.loss_fn == 'L1':
        loss_fn = torch.nn.L1Loss(reduction='sum')
    else:
        raise RuntimeError('Unsupported loss function type specified.')

    train(opt, model, loss_fn, device, dataset, optimizer, lr_scheduler)


if __name__ == '__main__':
    main()
