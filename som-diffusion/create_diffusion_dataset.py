import math
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from train_ae import SomAutoEncoder, load_file_list, FileListImageDataset, count_parameters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='device to use')
    parser.add_argument('--device-index', default=0, type=int, help='device index')
    parser.add_argument('--manual_seed', default=42, type=int, help='initialization of pseudo-RNG')
    parser.add_argument('--batch_size', default=96, type=int, help='batch size')

    parser.add_argument('--file_list_fn', default='imgnet_sm64_files.pth', type=str)
    parser.add_argument('--image_dir_path', default='/media/koepf/data_cache/imagenet_small/64x64/**/*', type=str)
    parser.add_argument('--image_fn_regex', default='.*\.png$', type=str)

    parser.add_argument('--checkpoint', default='experiments/som/som_2b_som_checkpoint_0010000.pth', type=str)
    parser.add_argument('--dataset_fn', default='diffusion_dataset.pth', type=str)
    parser.add_argument('--max_examples', default=-1, type=int)

    opt = parser.parse_args()
    return opt


def main():
    print('Using pytorch version {}'.format(torch.__version__))

    opt = parse_args()
    print('Options:', opt)

    device = torch.device(opt.device, opt.device_index)
    print('Device:', device)


    print('loading model checkpoint: ', opt.checkpoint)
    checkpoint_data = torch.load(opt.checkpoint, map_location=device)

    chkpt_opt = checkpoint_data['opt']
    embedding_dim = chkpt_opt.embedding_dim
    downscale_steps = chkpt_opt.downscale_steps
   
    # restore model
    model = SomAutoEncoder(embedding_dim=embedding_dim, downscale_steps=downscale_steps, pass_through_som=True)
    model.load_state_dict(checkpoint_data['model_state_dict'])

    test_batch = torch.rand(1, 3, 64, 64)
    test_latent = model.encoder(test_batch)
    print('latent size: ', test_latent.shape)

    count_parameters(model)

    model.to(device)

    file_names = load_file_list(opt.file_list_fn, opt.image_dir_path, opt.image_fn_regex)

    # data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = FileListImageDataset(file_names, transform)
    def remove_none_collate(batch):
        batch = list(filter(lambda x : x is not None, batch))
        return default_collate(batch)
    data_loader = DataLoader(ds, batch_size=opt.batch_size, shuffle=True, collate_fn=remove_none_collate, num_workers=4)

    encoded_batches = [] 

    c = 0
    for batch in data_loader:
        model.train()
        batch = batch.to(device)

        h = model.encode_2d(batch)

        encoded_batches.append(h.cpu())
        c += h.size(0)

        if opt.max_examples > 0 and c > opt.max_examples:
            break 
    
    encoded_batches = torch.cat(encoded_batches, dim=0)
    if opt.max_examples > 0 and encoded_batches.size(0) > opt.max_examples:
        encoded_batches = encoded_batches[:opt.max_examples]
    print('encoded_batches', encoded_batches.size())
    
    print('writing file: ', opt.dataset_fn)
    torch.save({
        'data': encoded_batches,
        'opt': chkpt_opt
    }, opt.dataset_fn)


if __name__ == '__main__':
    main()
