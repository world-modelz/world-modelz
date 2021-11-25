import math
import argparse
import uuid

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import torchvision

import matplotlib.pyplot as plt
from PIL import Image
import wandb

from som import SomLayer
from autoencoder import SomAutoEncoder
from train_ae import load_file_list, FileListImageDataset, show_and_save, count_parameters, wandb_init


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='device to use')
    parser.add_argument('--device-index', default=1, type=int, help='device index')
    parser.add_argument('--manual_seed', default=42, type=int, help='initialization of pseudo-RNG')
    parser.add_argument('--batch_size', default=96, type=int, help='batch size')

    parser.add_argument('--downscale_steps', default=3, type=int)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--hidden_planes', default=128, type=int)

    parser.add_argument('--file_list_fn', default='imgnet_sm64_files.pth', type=str)
    parser.add_argument('--image_dir_path', default='/media/koepf/data_cache/imagenet_small/64x64/**/*', type=str)
    parser.add_argument('--image_fn_regex', default='.*\.png$', type=str)
    parser.add_argument('--checkpoint_interval', default=2000, type=int)
    
    parser.add_argument('--som_width', default=None, type=int, help='rebuild SOM layer with width')
    parser.add_argument('--som_height', default=None, type=int, help='rebuild SOM layer with height')
    parser.add_argument('--adapt_batch_size', default=32, type=int, help='batch size for parallel SOM adaption')
    parser.add_argument('--sigma_begin', default=64, type=float, help='initial adaption kernel std-width')
    parser.add_argument('--sigma_end', default=0.1, type=float, help='final adaption kernel std-width')
    parser.add_argument('--eta_begin', default=0.5, type=float, help='initial learning rate')
    parser.add_argument('--eta_end', default=0.05, type=float, help='final learning rate')
    parser.add_argument('--max_steps', default=10000, type=int)

    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--entity', default='andreaskoepf', type=str)
    parser.add_argument('--tags', default=None, type=str)
    parser.add_argument('--project', default='som-diffusion', type=str, help='project name for wandb')
    parser.add_argument('--name', default='som_' + uuid.uuid4().hex, type=str, help='wandb experiment name')

    parser.add_argument('--ae_checkpoint', default='som-diffusion_checkpoint_0045000.pth', type=str)

    opt = parser.parse_args()
    return opt


@torch.no_grad()
def train_som(opt, model, device, data_loader, wandb_log_interval=500):

    experiment_name = opt.name
    checkpoint_interval = opt.checkpoint_interval
   
    plt.ion()

    T = opt.max_steps
    sigma_begin = opt.sigma_begin
    sigma_end = opt.sigma_end

    eta_begin = opt.eta_begin
    eta_end = opt.eta_end
    
    exp_decay_scale = math.log(sigma_end/sigma_begin)

    step = 1
    epoch = 1
    while True and step <= T:
        print('Epoch: {}'.format(epoch))

        for t, batch in enumerate(data_loader, 1):
            batch = batch.to(device)

            h = model.encoder(batch)

            # convert inputs from BCHW -> BHWC
            h = h.permute(0, 2, 3, 1).contiguous()

            progress = (step-1)/T   # increases linearly from 0 to 1
            
            # learning rate: linear decay
            eta = eta_begin if eta_begin == eta_end else eta_begin * (1.0 - progress) + progress * eta_end               
            eta = max(eta, 0)

            # radius: exponential decay
            sigma = sigma_begin * math.exp(progress * exp_decay_scale)
            sigma = max(sigma, 0)

            som_error = model.som.adapt(h, eta, sigma, adapt_batch_size=opt.adapt_batch_size, stats=True)

            print('{}: som_error: {}; eta: {}; sigma: {};'.format(step, som_error, eta, sigma))

            wandb.log({'som_error': som_error, 'progress': progress, 'eta': eta, 'sigma': sigma })

            if step % checkpoint_interval == 0 or step == T:
                # write model_checkpoint
                fn = '{}_som_checkpoint_{:07d}.pth'.format(experiment_name, step)
                print('writing file: ' + fn)
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'opt': opt
                }, fn)

                # save image with checkpoint
                reconstruction,_ = model(batch)
                fn = '{}_som_{:07d}.png'.format(experiment_name, step)
                show_and_save(fn, reconstruction, show=False, save=True)

            if step % wandb_log_interval == 0:
                # log reconstruction to wandb
                reconstruction,_ = model(batch)
                img_grid = torchvision.utils.make_grid(reconstruction.detach().cpu())
                images = wandb.Image(img_grid, caption="Reconstruction")
                wandb.log({"reconstruction": images})

            step = step + 1
            if step > T:
                break

        epoch = epoch + 1


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

    # data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = FileListImageDataset(file_names, transform)
    def remove_none_collate(batch):
        batch = list(filter(lambda x : x is not None, batch))
        return default_collate(batch)
    data_loader = DataLoader(ds, batch_size=opt.batch_size, shuffle=True, collate_fn=remove_none_collate, num_workers=4)

    model = SomAutoEncoder(embedding_dim=opt.embedding_dim, downscale_steps=opt.downscale_steps, pass_through_som=True)
    
    test_batch = torch.rand(1, 3, 64, 64)
    test_latent = model.encoder(test_batch)
    print('latent size: ', test_latent.shape)

    print('loading auto-encoder checkpoint: ', opt.ae_checkpoint)
    checkpoint_data = torch.load(opt.ae_checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'])

    if opt.som_width is not None and opt.som_width > 0 or opt.som_height is not None and opt.som_height > 0:
        # reinitialze SOM layer to new size
        w = model.som.width if opt.som_width is None else opt.som_width
        h = model.som.height if opt.som_height is None else opt.som_height
        print('reinitializing SOM layer to new size {}x{} (embedding dim: {})'.format(w, h, model.som.embedding_dim))
        model.som = SomLayer(width=w, height=h, embedding_dim=model.som.embedding_dim)

    count_parameters(model)
    model.to(device)
   
    train_som(opt, model, device, data_loader)


if __name__ == '__main__':
    main()
