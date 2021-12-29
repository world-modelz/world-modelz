# source: https://github.com/edenton/svg/blob/master/data/moving_mnist.py

import numpy as np
from numpy.random import randint
from torchvision import datasets, transforms

class MovingMNIST(object):
    
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=128, digit_size=48, deterministic=True):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.max_velocity = digit_size // 5
        self.step_length = 0.1
        self.digit_size = digit_size
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1 

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        min_velocity, max_velocity = -self.max_velocity, self.max_velocity+1
        x = np.zeros((self.seq_len,
                      image_size, 
                      image_size, 
                      self.channels),
                    dtype=np.float32)
        for n in range(self.num_digits):
            idx = randint(self.N)
            digit, _ = self.data[idx]

            sx = randint(image_size-digit_size)
            sy = randint(image_size-digit_size)
            dx = randint(min_velocity, max_velocity)
            dy = randint(min_velocity, max_velocity)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0 
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = randint(1, max_velocity)
                        dx = randint(min_velocity, max_velocity)
                elif sy >= image_size-digit_size:
                    sy = image_size-digit_size-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = randint(min_velocity, 0)
                        dx = randint(min_velocity, max_velocity)
                    
                if sx < 0:
                    sx = 0 
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = randint(1, max_velocity)
                        dy = randint(min_velocity, max_velocity)
                elif sx >= image_size-digit_size:
                    sx = image_size-digit_size-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = randint(min_velocity, 0)
                        dy = randint(min_velocity, max_velocity)
                   
                x[t, sy:sy+digit_size, sx:sx+digit_size, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x
