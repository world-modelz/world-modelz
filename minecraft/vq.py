import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, num_latents=1, decay=0.99, eps=1e-5):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_latents = num_latents
        self.decay = decay
        self.eps = eps

        self.register_buffer('embedding', torch.randn(self.num_latents, self.num_embeddings, self.embedding_dim))
        self.register_buffer('cluster_size', torch.ones(self.num_latents, self.num_embeddings))
        self.register_buffer('latent_offsets', torch.arange(self.num_latents).mul(self.num_embeddings).unsqueeze(0), persistent=False)
        self.register_buffer('activation_count', torch.zeros(self.num_latents, self.num_embeddings), persistent=False)
        self.register_buffer('accumulated_error', torch.zeros(self.num_latents, self.num_embeddings), persistent=False)

        self.simple_update = False 
        self.laplace_smoothing = True
        
    def forward(self, input):
        # expected input shape is (batch x (latents x) embedding_dim)
        flat_input = input.reshape(-1, self.num_latents, self.embedding_dim)

        # calculate distances
        distances = (flat_input.unsqueeze(-1) - self.embedding.transpose(1, 2).unsqueeze(0)).pow(2).sum(dim=-2)

        # get indices of minimum distances
        encoding_indices = distances.argmin(dim=-1)
        quantized = self.decode(encoding_indices)
        embedding_errors = torch.sum((quantized - flat_input) ** 2, dim=2).detach()
        self.accumulated_error.scatter_add_(-1, encoding_indices.t(), embedding_errors.t())

        quantized = quantized.view_as(input)
        encodings = torch.zeros_like(distances).scatter(-1, encoding_indices.unsqueeze(-1), 1)

        # EMA update of embedding vectors
        if self.training:
            embeding_onehot_sum = encodings.sum(dim=0)
            self.activation_count.add_(embeding_onehot_sum)

            dw = encodings.permute(1, 2, 0) @ flat_input.transpose(0, 1)

            if self.simple_update:
                dw = dw / embeding_onehot_sum.unsqueeze(-1)
                mask = dw == dw     # only non-nan entries
                self.embedding.data[mask] = self.decay * self.embedding.data[mask] + (1.0 - self.decay) * dw[mask] 
            else:
                self.cluster_size.data.mul_(self.decay).add_(embeding_onehot_sum, alpha=(1 - self.decay))

                # Laplace smoothing of the cluster size
                if self.laplace_smoothing:
                    n = self.cluster_size.sum(dim=-1, keepdim=True)
                    cluster_size = (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
                else:
                    cluster_size = self.cluster_size

                # empirically directly using dw instead of calculating an EMA over it
                # (unexpectedly) quickly increases perplexity
                dw = dw / cluster_size.unsqueeze(-1)
                self.embedding.data.mul_(self.decay).add_(dw, alpha=(1 - self.decay))

        commitment_loss = F.mse_loss(quantized.detach(), input)

        # straight through estimator
        quantized = input + (quantized - input).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10) / self.num_latents))

        return quantized, encodings, commitment_loss, perplexity
    
    def codebook_distance(self, input, normalize=True):
        flat_input = input.reshape(-1, self.num_latents, self.embedding_dim)
        distances = (flat_input.unsqueeze(-1) - self.embedding.transpose(1, 2).unsqueeze(0)).pow(2).sum(dim=-2)
        if normalize:
            distances = distances / self.embedding_dim
        return distances

    def encode(self, input):
        distances = self.codebook_distance(input, False)
        encoding_indices = distances.argmin(dim=-1)
        return encoding_indices

    def decode(self, indices):
        encoding_indices = indices.view(-1, self.num_latents)
        flat_index = self.latent_offsets.add(encoding_indices).view(-1)
        quantized = self.embedding.view(self.num_latents * self.num_embeddings, self.embedding_dim)[flat_index]
        quantized = quantized.view(*indices.shape, self.embedding_dim)
        return quantized

    def reuse_inactive(self):
        total_reused = 0
        # move codebook entries with zero activity closer to highly active entries
        for i in range(self.num_latents):
            dead_mask = self.activation_count[i] == 0
            num_dead = dead_mask.count_nonzero().item()
            if num_dead > 0:
                v,j = self.activation_count[i].topk(num_dead)
                self.embedding[i][dead_mask] = self.embedding[i][dead_mask] * 0.1 + self.embedding[i][j] * 0.9
                #self.cluster_size[i][dead_mask] = self.cluster_size[i].mean()
                total_reused += num_dead
        return total_reused

    def reset_stats(self):
        self.activation_count.zero_()
        self.accumulated_error.zero_()


class VectorQuantizerEMA1(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA1, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        device = self._embedding.weight.device

        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.reshape(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
    
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w.data = self._ema_w * self._decay + (1 - self._decay) * dw
            
            self._embedding.weight.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        commitment_loss = torch.mean((quantized.detach() - inputs)**2)
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, encodings, commitment_loss, perplexity
