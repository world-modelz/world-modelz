import math

import torch
import torch.nn as nn
import torchvision


class SomLayer(nn.Module):
    def __init__(self, width, height, embedding_dim):
        super(SomLayer, self).__init__()

        assert(width > 0 and height > 0)
        assert(embedding_dim > 0)

        self.width = width
        self.height = height

        num_embeddings = width * height
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        x_indices = torch.arange(width, dtype=torch.float32).repeat(height, 1)
        y_indices = torch.arange(height, dtype=torch.float32).unsqueeze(-1).repeat(1, width)
        pos_map = torch.cat((x_indices.unsqueeze(-1), y_indices.unsqueeze(-1)), dim=-1).view(1, num_embeddings, 2)
        self.register_buffer('pos_map', pos_map)
        self.register_buffer('activation_count', torch.zeros(num_embeddings, dtype=torch.long))

    @torch.no_grad()
    def encode_indices_2d(self, bmu_indices):
      
        # generate x,y coordinates from indices
        bmu_indices = bmu_indices.unsqueeze(-1)
        best_match_y = bmu_indices.div(self.width, rounding_mode='trunc')
        best_match_x = bmu_indices.remainder(self.width)
        bmu_pos = torch.cat((best_match_x, best_match_y), dim=-1)
        bmu_pos = bmu_pos.float()

        bmu_pos_shape = bmu_pos.shape
        bmu_pos = bmu_pos.view(-1, 2)

        # subtract map center and add 0.5 to coordinates (pixel-centers)
        bmu_pos = bmu_pos - torch.tensor([self.width/2 - 0.5, self.height/2 - 0.5], device=bmu_pos.device)

        # device by half of width to get values in range [-1, 1] for all (x, y)
        bmu_pos[:,0].div_(self.width/2)
        bmu_pos[:,1].div_(self.height/2)

        bmu_pos = bmu_pos.view(bmu_pos_shape)

        return bmu_pos  # BxHxWx2

    @torch.no_grad()    
    def decode_indices_2d(self, bmu_pos):
        bmu_pos_shape = bmu_pos.shape
        assert(bmu_pos_shape[-1] == 2)

        bmu_pos = bmu_pos.clamp(-1, 1).view(-1, 2)
        bmu_pos[:,0].mul_(self.width/2)
        bmu_pos[:,1].mul_(self.height/2)

        bmu_pos = bmu_pos + torch.tensor([self.width/2, self.height/2], device=bmu_pos.device)
        bmu_pos = bmu_pos.long()
        
        bmu_indices = bmu_pos[:,1] * self.width + bmu_pos[:,0]
        bmu_indices = bmu_indices.view(bmu_pos_shape[:-1])

        return bmu_indices

    @torch.no_grad()
    def encode_2d(self, input):
        bmu_indices = self.encode(input)    # BxHxW
        return self.encode_indices_2d(bmu_indices)

    @torch.no_grad()    
    def decode_2d(self, bmu_pos):
        bmu_indices = self.decode_indices_2d(bmu_pos)
        return self.decode(bmu_indices)

    @torch.no_grad()
    def encode(self, input):
        assert(input.size(-1) == self.embedding_dim)

        # flatten input
        flat_input = input.reshape(-1, self.embedding_dim)

        # calculate distances (squared)
        # broadcasting variant requires more memory than matmul version
        #distances = (flat_input.unsqueeze(-1) - self.embedding.weight.t().unsqueeze(0)).pow(2).sum(dim=-2)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
  
        bmu_indices = torch.argmin(distances, dim=1)

        return bmu_indices.view(input.shape[:-1])

    @torch.no_grad()
    def decode(self, input):
        quantized = self.embedding.forward(input)
        return quantized

    def forward(self, input):
        bmu_indices = self.encode(input)
        quantized = self.decode(bmu_indices)
        quantized = input + (quantized - input).detach()    # pass through gradient
        diff = (quantized.detach() - input).pow(2).mean()
        return quantized, diff

    def add_stats(self, bmu_indices):
        bmu_indices = bmu_indices.view(-1)
        src = torch.ones_like(bmu_indices, dtype=torch.long)
        self.activation_count.scatter_add_(0, bmu_indices, src)

    def reset_stats(self):
        self.activation_count.zero_()

    @torch.no_grad()
    def adapt(self, x, alpha, sigma, adapt_batch_size=256, stats=True):
        assert(x.size(-1) == self.embedding_dim)
        alpha = max(0, alpha)
        sigma = max(1e-6, sigma)

        flat_input = x.reshape(-1, self.embedding_dim)
        num_flat_inputs = flat_input.size(0)

        error_sum = torch.tensor(0.0, device=x.device)
        for chunk_begin in range(0, num_flat_inputs, adapt_batch_size):
            chunk_end = min(chunk_begin + adapt_batch_size, num_flat_inputs)

            flat_input_chunk = flat_input[chunk_begin:chunk_end]
            bmu_indices = self.encode(flat_input_chunk)

            quantized = self.decode(bmu_indices)
            error_sum = error_sum + torch.sum((flat_input_chunk - quantized) ** 2)

            if stats:
                self.add_stats(bmu_indices)
            
            # generate x,y coordinates from indices
            bmu_indices = bmu_indices.unsqueeze(-1)
            best_match_y = bmu_indices.div(self.width, rounding_mode='trunc')
            best_match_x = bmu_indices.remainder(self.width)
            best_match_pos_chunk = torch.cat((best_match_x, best_match_y), dim=1)

            # compute (squared) distance of map positions to the position of the best matching unit 
            bmu_distance = self.pos_map - best_match_pos_chunk.unsqueeze(-2)
            bmu_distance_squared = torch.sum(bmu_distance ** 2, dim=-1)

            neighborhood = torch.exp(-bmu_distance_squared / (sigma ** 2)).unsqueeze(-1)

            # accumulate adaption directions
            delta = torch.mean(neighborhood * (flat_input_chunk.unsqueeze(-2) - self.embedding.weight.data.unsqueeze(0)), dim=0)
            self.embedding.weight.data.add_(alpha * delta)
        
        return error_sum / flat_input.numel()
        

def test_rgb_som():
    device = torch.device("cuda", 1)
    width,height = 128, 128
    s = SomLayer(width, height, 3)
    s.to(device)

    num_iter = 100

    sigma_begin = width / 2
    sigma_end = 1.0
    exp_decay_scale = math.log(sigma_end/sigma_begin)

    eta_begin = 0.2
    eta_end = 0.01

    T = num_iter
    fake_input = torch.rand(512*512, 3).to(device)
    for i in range(1, num_iter+1):
        
        perm = torch.randperm(fake_input.size(0), device=device)
        input = fake_input[perm]

        progress = (i-1)/T   # increases linearly from 0 to 1

        # learning rate: linear decay
        eta = eta_begin if eta_begin == eta_end else eta_begin * (1.0 - progress) + progress * eta_end
        eta = max(eta, 0)
        
        # radius: exponential decay
        sigma = sigma_begin * math.exp(progress * exp_decay_scale)

        error = s.adapt(input, alpha=eta, sigma=sigma, adapt_batch_size=1024)

        print('step: {}: sigma: {:.2f}; eta {:.2f}; error: {:.5f};'.format(i, sigma, eta, error.item()))
        if i % 10 == 0:
            img = s.embedding.weight.data.view(height, width, 3).permute(2,0,1)
            torchvision.utils.save_image(img, 'test{0}.png'.format(i))


if __name__ == '__main__':
    test_rgb_som()
