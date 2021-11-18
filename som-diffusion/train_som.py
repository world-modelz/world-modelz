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

from train_ae import SomAutoEncoder, load_file_list, FileListImageDataset, show_and_save, count_parameters, wandb_init


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='device to use')
    parser.add_argument('--device-index', default=0, type=int, help='device index')
    parser.add_argument('--manual_seed', default=0, type=int, help='initialization of pseudo-RNG')
    parser.add_argument('--batch_size', default=96, type=int, help='batch size')

    parser.add_argument('--downscale_steps', default=3, type=int)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--hidden_planes', default=128, type=int)

    parser.add_argument('--file_list_fn', default='imgnet_sm64_files.pth', type=str)
    parser.add_argument('--image_dir_path', default='/media/koepf/data_cache/imagenet_small/64x64/**/*', type=str)
    parser.add_argument('--image_fn_regex', default='.*\.png$', type=str)
    parser.add_argument('--checkpoint_interval', default=2000, type=int)
    parser.add_argument('--max_steps', default=10000, type=int)

    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--entity', default='andreaskoepf', type=str)
    parser.add_argument('--tags', default=None, type=str)
    parser.add_argument('--project', default='som-diffusion', type=str, help='project name for wandb')
    parser.add_argument('--name', default='som_' + uuid.uuid4().hex, type=str, help='wandb experiment name')

    parser.add_argument('--ae_checkpoint', default='som-diffusion_checkpoint_0032500.pth', type=str)

    opt = parser.parse_args()
    return opt


@torch.no_grad()
def train_som(opt, model, device, data_loader, max_epochs=10, wandb_log_interval=500):

    experiment_name = opt.name
    checkpoint_interval = opt.checkpoint_interval
   
    plt.ion()

    T = opt.max_steps
    initial_sigma = model.som.width / 2
    initial_alpha = 0.85

    final_sigma = 1.0
    exp_decay_scale = math.log(final_sigma/initial_sigma)

    step = 1
    for epoch in range(max_epochs):
        print('Epoch: {}'.format(epoch))

        for t, batch in enumerate(data_loader, 1):
            batch = batch.to(device)

            h = model.encoder(batch)

            # convert inputs from BCHW -> BHWC
            h = h.permute(0, 2, 3, 1).contiguous()

            progress = (step-1)/T   # increases linearly from 0 to 1
            alpha = initial_alpha * (1.0 - progress)    # learning rate: linear decay
            sigma = initial_sigma * math.exp(progress * exp_decay_scale)    # radius: exponential decay

            som_error = model.som.adapt(h, alpha, sigma, adapt_batch_size=64, stats=True)

            print('{}: som_error: {}; alpha: {}; sigma: {};'.format(step, som_error, alpha, sigma))

            wandb.log({'som_error': som_error, 'alpha': alpha, 'sigma': sigma })

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
                reconstruction = model(batch)
                fn = '{}_som_{:07d}.png'.format(experiment_name, step)
                show_and_save(fn, reconstruction, show=False, save=True)

            if step % wandb_log_interval == 0:
                # log reconstruction to wandb
                reconstruction = model(batch)
                img_grid = torchvision.utils.make_grid(reconstruction.detach().cpu())
                images = wandb.Image(img_grid, caption="Reconstruction")
                wandb.log({"reconstruction": images})

            step = step + 1
            if step > T:
                break



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

    count_parameters(model)
    
    print('loading auto-encoder checkpoint: ', opt.ae_checkpoint)
    checkpoint_data = torch.load(opt.ae_checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'])

    model.to(device)
   
    train_som(opt, model, device, data_loader)


if __name__ == '__main__':
    main()
