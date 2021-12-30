import argparse
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import torchvision

import matplotlib.pyplot as plt
from data.moving_mnist import MovingMNIST

import wandb

from autoencoder import SimpleResidualEncoder, SimpleResidualDecoder
from vq import VectorQuantizerEMA


class VqAutoEncoder(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, downscale_steps=2, hidden_planes=128, in_channels=3):
        super(VqAutoEncoder, self).__init__()

        self.encoder = SimpleResidualEncoder(in_channels, embedding_dim, downscale_steps, hidden_planes)

        decoder_cfg = [hidden_planes for _ in range(downscale_steps)]
        self.decoder = SimpleResidualDecoder(decoder_cfg, in_channels=embedding_dim, out_channels=in_channels)
        
        self.vq = VectorQuantizerEMA(embedding_dim, num_embeddings)

    def forward(self, x):
        h = self.encoder(x)
    
        # convert inputs from BCHW -> BHWC
        h = h.permute(0, 2, 3, 1)
        quantized, encodings, latent_loss, perplexity = self.vq.forward(h)

        # convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return self.decoder(quantized), latent_loss, perplexity

    def encode(self, x):
        h = self.encoder(x)
        h = h.permute(0, 2, 3, 1)   # BCHW -> BHWC
        indices = self.vq.encode(h).view(h.shape[:-1])
        return indices

    def decode(self, z):
        h = self.vq.decode(z)
        h = h.permute(0, 3, 1, 2)   # BHWC -> BCHW
        x = self.decoder(h)
        return x


# parse bool args correctly, see https://stackoverflow.com/a/43357954
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='device to use')
    parser.add_argument('--device-index', default=0, type=int, help='device index')
    parser.add_argument('--manual_seed', default=0, type=int, help='initialization of pseudo-RNG')
    parser.add_argument('--batch_size', default=96, type=int, help='batch size')
    parser.add_argument('--optimizer', default='AdamW', type=str, help='Optimizer to use (Adam, AdamW)')
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--loss_fn', default='SmoothL1', type=str)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--wandb_log_interval', default=1000, type=int, help='wandb logging interval of reconstruction image')

    parser.add_argument('--downscale_steps', default=3, type=int)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--hidden_planes', default=128, type=int)
    parser.add_argument('--num_embeddings', default=512, type=int, help='VQ codebook size')

    parser.add_argument('--checkpoint_interval', default=2500, type=int)
    parser.add_argument('--latent_loss_weight', default=0.01, type=float)
    parser.add_argument('--vq_reuse_interval', default=500, type=int)

    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--entity', default='andreaskoepf', type=str)
    parser.add_argument('--tags', default=None, type=str)
    parser.add_argument('--project', default='vqvd_vq', type=str, help='project name for wandb')
    parser.add_argument('--name', default='vqvdvq_' + uuid.uuid4().hex, type=str, help='wandb experiment name')

    # moving MNIST dataset parameters
    parser.add_argument('--data_root', default='data', help='root directory for data')
    parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--num_digits', type=int, default=5, help='number of digits for moving mnist')
    parser.add_argument('--digit_size', type=int, default=24, help='size of single moving digit')

    opt = parser.parse_args()
    return opt


def show_batch(x, nrow=8):
    x = x.detach().cpu()
    grid = torchvision.utils.make_grid(x, nrow=nrow)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.show()
    plt.pause(0.1)


def show_and_save(fn, reconstructions, show=True, save=True):
    reconstructions = reconstructions.cpu()
    if show:
        show_batch(reconstructions)
    if fn and save:
        torchvision.utils.save_image(reconstructions, fn)


def train(opt, model, loss_fn, device, data_loader, optimizer, lr_scheduler):
    plt.ion()

    experiment_name = opt.name
    checkpoint_interval = opt.checkpoint_interval
    max_epochs = opt.max_epochs
    wandb_log_interval = opt.wandb_log_interval
    vq_reuse_interval = opt.vq_reuse_interval

    train_recon_error = []
    step = 1
    for epoch in range(max_epochs):
        print('Epoch: {}, lr: {}'.format(epoch, lr_scheduler.get_last_lr()))

        for t, batch in enumerate(data_loader, 1):
            model.train()

            assert batch.shape[1] == batch.shape[4] == 1
            batch = batch.squeeze(-1).to(device)

            reconstruction, latent_loss, perplexity = model(batch)
            r_loss = loss_fn(reconstruction, batch)

            loss = r_loss + opt.latent_loss_weight * latent_loss

            train_recon_error.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_data = {'loss': loss, 'r_loss': r_loss, 'latent_loss': latent_loss, 'perplexity': perplexity, 'lr': lr_scheduler.get_last_lr()[0]}
            
            print('step: {}; loss: {}; perplexity: {}; lr: {}; epoch: {};'.format(step, loss.item(), perplexity.item(), lr_scheduler.get_last_lr()[0], epoch))

            if step % vq_reuse_interval == 0:
                c = model.vq.reuse_inactive()
                log_data['reused'] = c
                print('resued: ', c)
                model.vq.reset_stats()

            wandb.log(log_data)

            if step % checkpoint_interval == 0:
                # write model_checkpoint
                fn = '{}_checkpoint_{:07d}.pth'.format(experiment_name, step)
                print('writing file: ' + fn)
                torch.save({
                    'step': step,
                    'lr': lr_scheduler.get_last_lr(),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': { 'train_recon_error': train_recon_error },
                    'opt': opt
                }, fn)

                fn = '{}_reconst_{:07d}.png'.format(experiment_name, step)
                show_and_save(fn, reconstruction, show=False, save=True)

            if step % wandb_log_interval == 0:
                # log reconstruction to wandb
                img_grid = torchvision.utils.make_grid(reconstruction.detach().cpu())
                images = wandb.Image(img_grid, caption="Reconstruction")
                wandb.log({"reconstruction": images})
            
            step = step + 1

        lr_scheduler.step()


# print the number of parameters in a module
def count_parameters(module):
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in module.parameters()])))


def wandb_init(opt):
    if opt.wandb:
        wandb_mode = 'online'
        wandb.login()
    else:
        wandb_mode = 'disabled'

    tags = []
    if opt.tags is not None:
        tags.extend(opt.tags.split(','))

    wandb.init(project=opt.project, entity=opt.entity, config=vars(opt), tags=tags, mode=wandb_mode, name=opt.name)


def main():
    print('Using pytorch version {}'.format(torch.__version__))

    opt = parse_args()
    print('Options:', opt)

    device = torch.device(opt.device, opt.device_index)
    print('Device:', device)

    torch.manual_seed(opt.manual_seed)

    # weights and biases
    wandb_init(opt)

    # create data set
    train_data = MovingMNIST(
                train=True,
                data_root=opt.data_root,
                seq_len=1,
                image_size=opt.image_width,
                deterministic=False,
                num_digits=opt.num_digits,
                digit_size=opt.digit_size)

    data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=4)

     # data loader
    model = VqAutoEncoder(opt.embedding_dim, opt.num_embeddings, opt.downscale_steps, hidden_planes=opt.hidden_planes, in_channels=1)
    
    test_batch = torch.rand(1, 1, 64, 64)
    test_latent = model.encoder(test_batch)
    print('latent size: ', test_latent.shape)

    count_parameters(model)
    
    model.to(device)

    learning_rate = opt.lr
    if opt.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=opt.weight_decay, amsgrad=False)
    elif opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=opt.weight_decay, amsgrad=False)
    else:
        raise RuntimeError('Unsupported optimizer specified.')

    # simple lr-schedule: per epoch lr halfing
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    
    if opt.loss_fn == 'SmoothL1':
        loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
    elif opt.loss_fn == 'MSE':
        loss_fn = torch.nn.MSELoss(reduction='mean')
    elif opt.loss_fn == 'MAE' or opt.loss_fn == 'L1':
        loss_fn = torch.nn.L1Loss(reduction='mean')
    else:
        raise RuntimeError('Unsupported loss function type specified.')

    train(opt, model, loss_fn, device, data_loader, optimizer, lr_scheduler)


if __name__ == '__main__':
    main()
