from cs336_basics.model.modules import Linear, Embedding, RMSNorm, MultiHeadSelfAttention
from cs336_basics.model.modules import SwiGLU as FFN
import torch
import torch.nn as nn

class Block(nn.Module):

    def __init__(
        self,
        d_model: int,        # Dimensionality of the transformer block inputs
        num_heads: int,      # Number of heads to use in multi-head self-attention
        d_ff: int,           # Dimensionality of the position-wise feed-forward inner layer
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        assert d_model % num_heads == 0, "d_moel % num_heads != 0"

        self.rmsnorm_layer1 = RMSNorm(self.d_model)
        self.rmsnorm_layer2 = RMSNorm(self.d_model)

        self.multihead_self_attention = MultiHeadSelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
        )

        self.ffn = FFN(d_model=self.d_model, d_ff=self.d_ff)

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)
    
    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        x = in_features
        x = x + self.multihead_self_attention(self.rmsnorm_layer1(x))
        x = x + self.ffn(self.rmsnorm_layer2(x))
        return x

class Transformer(nn.Module):

    def __init__(
        self,
        vocab_size: int, context_length: int, d_model: int, num_layers: int,num_heads: int, d_ff: int,
        rope_theta: float
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        assert d_model % num_heads == 0, "d_model % num_heads != 0"

        self.token_embedding = Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)
        self.layers = nn.ModuleList(
            [ 
                Block(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    max_seq_len=self.context_length,
                    theta=self.rope_theta
                )
                for _ in range(self.num_layers)]
        )
        self.output_norm = RMSNorm(d_model=self.d_model)
        self.output_embedding = Linear(in_features=self.d_model, out_features=self.vocab_size)
    
    def forward(self, x: torch.Tensor):
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_norm(x)
        x = self.output_embedding(x)
        return x
    
if __name__ == "__main__":
    model = Transformer(
        vocab_size=50257,
        context_length=1024,
        d_model=1600,
        num_heads=25,
        d_ff = 6400,
        num_layers=48,
        rope_theta=0.1
    )
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))