import torch
import numpy as np


class LossAwareSamplerEma:
    def __init__(self, num_histogram_buckets=100, uniform_p=0.01, alpha=0.9, warmup=10, jitter=True):
        assert num_histogram_buckets > 1
        self.num_histogram_buckets = num_histogram_buckets
        self.uniform_p = uniform_p
        self.alpha = alpha
        self.warmup = warmup
        self.jitter = jitter
        self._weights = torch.ones(num_histogram_buckets, requires_grad=False)
        self._counts = torch.zeros(num_histogram_buckets, dtype=torch.long, requires_grad=False)

    @torch.no_grad()
    def weights(self):
        if not self.warmed_up():
            return torch.ones(self.num_histogram_buckets)   # uniform sampling during warmup

        w = self._weights.clone() / self._weights.sum()
        w = (1-self.uniform_p) * w + self.uniform_p / self.num_histogram_buckets
        return w

    @torch.no_grad()
    def sample(self, batch_size):
        w = torch.multinomial(self.weights(), batch_size, replacement=True).float()
        if self.jitter:
            w = (w + torch.rand_like(w)) / self.num_histogram_buckets
        else:
            w = w / (self.num_histogram_buckets-1)
        return w
    
    @torch.no_grad()
    def update_with_losses(self, ts, losses):
        ts = ts.view(-1).cpu()
        losses = losses.view(-1).cpu()
        indices = (ts * self.num_histogram_buckets).long()
        self._counts.scatter_add_(0, indices, torch.ones_like(indices))
        for i,j in enumerate(indices):
            self._weights[j] = self._weights[j] * self.alpha + losses[i] * (1-self.alpha) 

    def warmed_up(self):
        return (self._counts > self.warmup).all()

    def weights_as_numpy_histogram(self):
        return ((self.weights() / (self.weights().sum()+1e-6)).numpy(), np.arange(self.num_histogram_buckets+1) / self.num_histogram_buckets)


def test():
    l = LossAwareSamplerEma(100)
    l.update_with_losses(torch.rand(10000), torch.rand(10000))
    print('counts', l._counts, l._counts.sum())
    print('warmed up', l.warmed_up())
    print('weights', l.weights())

    x = l.sample(10)
    print('samples', x)


if __name__ == '__main__':
    test()
