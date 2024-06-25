from torch import nn
import torch


class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    Taken from https://github.com/huggingface/diffusers/blob/v0.27.2/src/diffusers/models/normalization.py#L28
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.time_emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, z: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t = self.emb(timestep)
        z = z + t
        emb = self.linear(self.silu(z))
        scale, shift = torch.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x


class MLP(nn.Module):
    def __init__(self, width, num_blocks):
        super().__init__()
        blocks = [ResidualBlock(input_dim=width, hidden_dim=width, output_dim=width) for b in range(num_blocks)]
        self.mlp = nn.Sequential(*blocks)
    
    def forward(self, x, z, t):
        return self.mlp(x,z,t)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_num_timsteps):
        super().__init__()
        self.ln = AdaLayerNorm(input_dim, max_num_timsteps)
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, z, t):
        x = self.ln(x,z,t)
        x = self.linear(x)
        return x + self.out_half(x) # residual
    
    def out_half(self, x):
        return self.linear_out(self.silu(x))

class TokenDiffusion(nn.Module):
    def __init__(self,):
        super().__init__()
        pass

    def foward(self, z):
        pass