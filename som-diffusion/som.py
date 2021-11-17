import torch
import torch.nn as nn
import torchvision


class SomLayer(nn.Module):
    def __init__(self, width, height, embedding_dim, commitment_cost=0.25):
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
    def encode(self, input):
        assert(input.size(-1) == self.embedding_dim)

        # flatten input
        flat_input = input.view(-1, self.embedding_dim)

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
        return quantized

    def add_stats(self, bmu_indices):
        bmu_indices = bmu_indices.view(-1)
        src = torch.ones_like(bmu_indices, dtype=torch.long)
        self.activation_count.scatter_add_(0, bmu_indices, src)

    def reset_stats(self):
        self.activation_count.zero_()

    @torch.no_grad()
    def adapt(self, x, alpha, sigma, adapt_batch_size=256, stats=True):
        assert(x.size(-1) == self.embedding_dim)

        flat_input = x.view(-1, self.embedding_dim)
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
    initial_sigma = width / 2
    initial_alpha = 0.2

    fake_input = torch.rand(512*512, 3).to(device)
    for i in range(1, num_iter+1):
        
        perm = torch.randperm(fake_input.size(0), device=device)
        input = fake_input[perm]

        annealing_factor = 1.0 - i/num_iter
        alpha = initial_alpha * annealing_factor
        sigma = initial_sigma * annealing_factor
        
        error = s.adapt(input, alpha=alpha, sigma=sigma, adapt_batch_size=1024)

        print('step: {}: sigma: {:.2f}; alpha {:.2f}; error: {:.5f};'.format(i, sigma, alpha, error.item()))
        if i % 10 == 0:
            img = s.embedding.weight.data.view(height, width, 3).permute(2,0,1)
            torchvision.utils.save_image(img, 'test{0}.png'.format(i))


if __name__ == '__main__':
    test_rgb_som()
