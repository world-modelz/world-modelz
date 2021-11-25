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

from autoencoder import SomAutoEncoder
from som import SomLayer


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
    parser.add_argument('--checkpoint_interval', default=2500, type=int)

    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--entity', default='andreaskoepf', type=str)
    parser.add_argument('--tags', default=None, type=str)
    parser.add_argument('--project', default='som-diffusion', type=str, help='project name for wandb')
    parser.add_argument('--name', default='ae_' + uuid.uuid4().hex, type=str, help='wandb experiment name')

    opt = parser.parse_args()
    return opt


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


def train(experiment_name, model, losss_fn, device, data_loader, optimizer, lr_scheduler, checkpoint_interval=2500, max_epochs=10, wandb_log_interval=1000):
    plt.ion()

    train_recon_error = []
    step = 1
    for epoch in range(max_epochs):
        print('Epoch: {}, lr: {}'.format(epoch, lr_scheduler.get_last_lr()))

        for t, batch in enumerate(data_loader, 1):
            model.train()

            batch = batch.to(device)

            reconstruction, _ = model(batch)
            loss = losss_fn(reconstruction, batch)

            wandb.log({'loss': loss.item(), 'lr': lr_scheduler.get_last_lr()[0]})
            print('step: {}; loss: {}; lr: {}; epoch: {};'.format(step, loss.item(), lr_scheduler.get_last_lr()[0], epoch))

            train_recon_error.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
    
    embedding_dim = opt.embedding_dim
    downscale_steps = opt.downscale_steps
    batch_size = opt.batch_size
    learning_rate = opt.lr


    # data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = FileListImageDataset(file_names, transform)
    def remove_none_collate(batch):
        batch = list(filter(lambda x : x is not None, batch))
        return default_collate(batch)
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=remove_none_collate, num_workers=4)

    model = SomAutoEncoder(embedding_dim=embedding_dim, downscale_steps=downscale_steps, pass_through_som=False)
    
    test_batch = torch.rand(1, 3, 64, 64)
    test_latent = model.encoder(test_batch)
    print('latent size: ', test_latent.shape)

    count_parameters(model)
    
    model.to(device)

    if opt.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=False)
    elif opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    else:
        raise RuntimeError('Unsupported optimizer specified.')

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

            
    if opt.loss_fn == 'SmoothL1':
        loss_fn = torch.nn.SmoothL1Loss(reduction='sum')
    elif opt.loss_fn == 'MSE':
        loss_fn = torch.nn.MSELoss(reduction='sum')
    elif opt.loss_fn == 'MAE' or opt.loss_fn == 'L1':
        loss_fn = torch.nn.L1Loss(reduction='sum')
    else:
        raise RuntimeError('Unsupported loss function type specified.')

    train(opt.name, model, loss_fn, device, data_loader, optimizer, lr_scheduler, opt.checkpoint_interval)


if __name__ == '__main__':
    main()
