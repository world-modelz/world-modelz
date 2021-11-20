import os, glob, re
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
from PIL import Image
import wandb

from autoencoder import SimpleResidualEncoder, SimpleResidualDecoder
from som import SomLayer


class SomAutoEncoder(nn.Module):
    def __init__(self, embedding_dim, downscale_steps=2, hidden_planes=128, in_channels=3, pass_through_som=False):
        super(SomAutoEncoder, self).__init__()

        self.encoder = SimpleResidualEncoder(in_channels, embedding_dim, downscale_steps, hidden_planes)

        decoder_cfg = [hidden_planes for _ in range(downscale_steps)]
        self.decoder = SimpleResidualDecoder(decoder_cfg, in_channels=embedding_dim, out_channels=in_channels)
        
        self.pass_through_som = pass_through_som
        self.som = SomLayer(width=128, height=128, embedding_dim=embedding_dim)

    def forward(self, x):
        h = self.encoder(x)

        if self.pass_through_som:
            
            # convert inputs from BCHW -> BHWC
            h_in = h.permute(0, 2, 3, 1)

            h, h_diff = self.som.forward(h_in)

            # convert quantized from BHWC -> BCHW
            h = h.permute(0, 3, 1, 2).contiguous()

        return self.decoder(h), h_in, h_diff


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


def load_file_list(file_list_fn, directory_path, pattern):
    if os.path.isfile(file_list_fn):
        list = torch.load(file_list_fn)
        if len(list) > 0:
            print('File list loaded with {} file names.'.format(len(list)))
            return list

    list = []

    fn_match = re.compile(pattern, flags=re.IGNORECASE)
    file_list = glob.iglob(directory_path, recursive=True)
    print('Scanning directory: ', directory_path)

    for fn in file_list:
        if os.path.isfile(fn):
            if fn_match.match(fn):
                fn = os.path.abspath(fn)
                list.append(fn)

    if len(list) == 0:
        raise RuntimeError('No matching files found.')

    print('Matching files found: {}'.format(len(list)))
    torch.save(list, file_list_fn)
    print('File \'{}\' written.'.format(file_list_fn))
    return list


class FileListImageDataset(Dataset):
    def __init__(self, file_names, transform):
        super(FileListImageDataset, self).__init__()
        self.file_names = file_names
        self.transform = transform

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        fn = self.file_names[index]
        try: 
            img = Image.open(fn).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img
        except Exception as e:
            print(e)


def show_batch(x):
    x = x.detach().cpu()
    grid = torchvision.utils.make_grid(x)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='device to use')
    parser.add_argument('--device-index', default=1, type=int, help='device index')
    parser.add_argument('--manual_seed', default=0, type=int, help='initialization of pseudo-RNG')
    parser.add_argument('--batch_size', default=96, type=int, help='batch size')
    parser.add_argument('--optimizer', default='AdamW', type=str, help='Optimizer to use (Adam, AdamW)')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--loss_fn', default='SmoothL1', type=str)

    parser.add_argument('--downscale_steps', default=3, type=int)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--hidden_planes', default=128, type=int)

    parser.add_argument('--file_list_fn', default='imgnet_sm64_files.pth', type=str)
    parser.add_argument('--image_dir_path', default='/media/koepf/data_cache/imagenet_small/64x64/**/*', type=str)
    parser.add_argument('--image_fn_regex', default='.*\.png$', type=str)
    parser.add_argument('--checkpoint_interval', default=2000, type=int)

    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--entity', default='andreaskoepf', type=str)
    parser.add_argument('--tags', default=None, type=str)
    parser.add_argument('--project', default='finetune_ae', type=str, help='project name for wandb')
    parser.add_argument('--name', default='finetune_ae_' + uuid.uuid4().hex, type=str, help='wandb experiment name')

    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--wandb_log_interval', default=500, type=int)
    parser.add_argument('--som_checkpoint', default='experiments/som_2b_som_checkpoint_0010000.pth', type=str)
    parser.add_argument('--som_adapt_rate', default=0.02, type=float)
    parser.add_argument('--som_adapt_radius', default=0.25, type=float)
    parser.add_argument('--som_adapt_batch', default=8, type=int)
    parser.add_argument('--som_adapt_skip', default=0, type=int)
    parser.add_argument('--latent_loss_weight', default=0.25, type=float)

    opt = parser.parse_args()
    return opt


def train(opt, model, loss_fn, device, data_loader, optimizer, lr_scheduler):
    plt.ion()

    experiment_name = opt.name
    checkpoint_interval = opt.checkpoint_interval
    max_epochs = opt.max_epochs
    wandb_log_interval = opt.wandb_log_interval

    som_adapt_rate = opt.som_adapt_rate
    som_adapt_radius = opt.som_adapt_radius
    som_adapt_batch = opt.som_adapt_batch
    som_adapt_interval = opt.som_adapt_skip + 1

    latent_loss_weight = opt.latent_loss_weight

    train_recon_error = []
    step = 1
    for epoch in range(max_epochs):
        print('Epoch: {}, lr: {}'.format(epoch, lr_scheduler.get_last_lr()))

        for t, batch in enumerate(data_loader, 1):
            model.train()

            batch = batch.to(device)

            reconstruction, h_in, h_diff = model(batch)
            r_loss = loss_fn(reconstruction, batch)
            h_loss = h_diff
            loss = r_loss + latent_loss_weight * h_loss 

            train_recon_error.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if som_adapt_rate > 0 and step % som_adapt_interval == 0:
                som_loss = model.som.adapt(h_in, som_adapt_rate, som_adapt_radius, som_adapt_batch)
            else:
                som_loss = torch.tensor(0)

            
            wand_log = {'loss': loss.item(), 'r_loss': r_loss, 'h_loss': h_loss, 'lr': lr_scheduler.get_last_lr()[0]}
            if som_loss > 0:
                wand_log['som_loss'] = som_loss
            wandb.log(wand_log)
            print('step: {}; loss: {} (h: {}); lr: {}; epoch: {};'.format(step, loss.item(), h_loss.item(), lr_scheduler.get_last_lr()[0], epoch))
            
            if step % checkpoint_interval == 0:
                # write model_checkpoint
                fn = '{}_checkpoint_{:07d}.pth'.format(experiment_name, step)
                print('writing file: ' + fn)
                torch.save({
                    'step': step,
                    'lr': lr_scheduler.get_last_lr(),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': { 'train_recon_error': train_recon_error }
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

    file_names = load_file_list(opt.file_list_fn, opt.image_dir_path, opt.image_fn_regex)

    device = torch.device(opt.device, opt.device_index)
    print('Device:', device)

    torch.manual_seed(opt.manual_seed)

    # weights and biases
    wandb_init(opt)

    batch_size = opt.batch_size
    learning_rate = opt.lr
    optimizer_name = opt.optimizer
    loss_fn_name = opt.loss_fn

    print('loading SOM checkpoint: ', opt.som_checkpoint)
    checkpoint_data = torch.load(opt.som_checkpoint, map_location=device)
    
    # use checkpoint options
    chkpt_opt = checkpoint_data['opt']

    embedding_dim = chkpt_opt.embedding_dim
    downscale_steps = chkpt_opt.downscale_steps
    
    # data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = FileListImageDataset(file_names, transform)
    def remove_none_collate(batch):
        batch = list(filter(lambda x : x is not None, batch))
        return default_collate(batch)
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=remove_none_collate, num_workers=4)

    model = SomAutoEncoder(embedding_dim=embedding_dim, downscale_steps=downscale_steps, pass_through_som=True)
    model.load_state_dict(checkpoint_data['model_state_dict'])

    test_batch = torch.rand(1, 3, 64, 64)
    test_latent = model.encoder(test_batch)
    print('latent size: ', test_latent.shape)

    count_parameters(model)
    
    model.to(device)

    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=False)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    else:
        raise RuntimeError('Unsupported optimizer specified.')

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    if loss_fn_name == 'SmoothL1':
        loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
    elif loss_fn_name == 'MSE':
        loss_fn = torch.nn.MSELoss(reduction='mean')
    elif loss_fn_name== 'MAE' or loss_fn_name == 'L1':
        loss_fn = torch.nn.L1Loss(reduction='mean')
    else:
        raise RuntimeError('Unsupported loss function type specified.')

    train(opt, model, loss_fn, device, data_loader, optimizer, lr_scheduler)


if __name__ == '__main__':
    main()
