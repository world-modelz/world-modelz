import random
import argparse

import matplotlib.pyplot as plt

import torch
import torchvision 

from data.moving_mnist import MovingMNIST


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='device to use')
    parser.add_argument('--device-index', default=0, type=int, help='device index')
    parser.add_argument('--manual_seed', default=42, type=int, help='initialization of pseudo-RNG')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

    # moving MNIST dataset parameters
    parser.add_argument('--data_root', default='data', help='root directory for data')
    parser.add_argument('--image_width', type=int, default=128, help='the height / width of the input image to network')
    parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
    parser.add_argument('--num_digits', type=int, default=3, help='number of digits for moving mnist')
    parser.add_argument('--digit_size', type=int, default=48, help='size of single moving digit')

    opt = parser.parse_args()
    return opt


def show_batch(x, nrow=8):
    x = x.detach().cpu()
    grid = torchvision.utils.make_grid(x, nrow=nrow)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.show()
    plt.pause(0.1)


def main():
    print('Using pytorch version {}'.format(torch.__version__))

    opt = parse_args()
    print('Options:', opt)

    device = torch.device(opt.device, opt.device_index)
    print('Device:', device)

    torch.manual_seed(opt.manual_seed)

    # create data set
    train_data = MovingMNIST(
                train=True,
                data_root=opt.data_root,
                seq_len=opt.n_past+1,
                image_size=opt.image_width,
                deterministic=False,
                num_digits=opt.num_digits,
                digit_size=opt.digit_size)

    x = torch.from_numpy(train_data[random.randint(0, len(train_data)-1)])
    show_batch(x.permute(0,3,1,2))


if __name__ == '__main__':
    main()