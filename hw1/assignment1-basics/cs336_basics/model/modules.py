import torch
import torch.nn as nn
from einops import einsum, rearrange

class Linear(nn.Module):
    
    def __init__(
        self,
        in_features: int,                         # final dimension of input
        out_features: int,                        # final dimension of output 
        device: torch.device | None = None,       # Device to store the parameters on
        dtype: torch.dtype | None = None          # Data type of the parameters
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weights = nn.Parameter(torch.empty((self.out_features, self.in_features), device=self.device, dtype=self.dtype))
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... in_features, out_features in_features -> ... out_features")

    def _init_weights(self):
        std = (2.0 / (self.in_features + self.out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3*std, b=3*std)

class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,                    # Size of thr vocabulary
        embedding_dim: int,                     # Dimension of the embedding vectors, i.e. d_model
        device: torch.device | None = None,     # Device to store the parameters on
        dtype: torch.dtype | None = None        # Data type of the parameters
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.embedding_weights = nn.Parameter(torch.empty((self.num_embeddings, self.embedding_dim), device=self.device, dtype=self.dtype))
        self._init_weights()
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()
        return self.embedding_weights[token_ids]

    def _init_weights(self):
        torch.nn.init.trunc_normal_(self.embedding_weights, mean=0.0, std=1.0, a=-3.0, b=3.0)

class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,                           # Hidden dimension of the model
        eps: float = 1e-5,                      # Epsilon value for numerical stability
        device: torch.device | None = None,     # Device to store the parameters on
        dtype: torch.dtype | None = None        # Data type of the parameters
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.gain = nn.Parameter(torch.empty((self.d_model,), device=self.device, dtype=self.dtype))
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape."""
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        out = einsum(x / rms, self.gain, "... d_model, d_model -> ... d_model")
        return out.to(dtype=in_dtype)

    def _init_weights(self):
        torch.nn.init.trunc_normal_(self.gain, mean=0.0, std=1.0, a=-3.0, b=3.0)

class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model: int,                          # Dimensionality of the feedforward input and output
        d_ff: int,                             # Dimensionality of the up-project happening internally to your swiglu
        device: torch.device | None = None,    # Device to store the parameters on
        dtype: torch.dtype | None = None       # Data type of the parameters
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1_weights = nn.Parameter(torch.empty((self.d_ff, self.d_model), device=self.device, dtype=self.dtype))
        self.w2_weights = nn.Parameter(torch.empty((self.d_model, self.d_ff), device=self.device, dtype=self.dtype))
        self.w3_weights = nn.Parameter(torch.empty((self.d_ff, self.d_model), device=self.device, dtype=self.dtype))
        self._init_weights()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape."""
        """SiLU(x) = x * sigmoid(x)"""
        """GLU(x, W1, W2) = sigmoid(W1 x) * (W2 x)"""
        """SwiGLU(x, W1, W2, W3) = W2(SiLU(W1 x) * (W3 x))"""
        a = einsum(self.w1_weights, x, "d_ff d_model, ... d_model -> ... d_ff")
        gate = a * torch.sigmoid(a)  # 门控单元 (0~1) 控制输出
        step2 = gate * einsum(self.w3_weights, x, "d_ff d_model, ... d_model -> ... d_ff")
        return einsum(self.w2_weights, step2, "d_model d_ff, ... d_ff -> ... d_model")

    def _init_weights(self):
        torch.nn.init.trunc_normal_(self.w1_weights, mean=0.0, std=1.0, a=-3.0, b=3.0)
        torch.nn.init.trunc_normal_(self.w2_weights, mean=0.0, std=1.0, a=-3.0, b=3.0)
        torch.nn.init.trunc_normal_(self.w3_weights, mean=0.0, std=1.0, a=-3.0, b=3.0)

class RoPE(nn.Module):

    def __init__(
        self,
        d_k: int,                           # dimension of the key/query vectors
        theta: float,                       # Θ value for the RoPE
        max_seq_len: int,                   # Maximum sequence length that will be inputted
        device: torch.device | None = None  # Device to store the buffer on
    ):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device

        # 旋转角度频率向量: (d_k/2,)
        self.freq_arrange = 1 / (self.theta ** (torch.arange(0, self.d_k, 2).to(dtype=torch.float32, device=self.device)/ self.d_k))
        # 把频率向量注册为buffer，以便在模型保存和加载时保留
        self.register_buffer(name="inv_freq", tensor=self.freq_arrange, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape."""
        # x: (B, H, S, Dh) 或 (B, S, D) 
        # 取决于在哪一层调用，在attention layer调用时有head维度，在ffn layer调用时没有head维度， 这里假设是(B, H, S, Dh)
        S, Dh = x.size(-2), x.size(-1)

        # 1). 准备 positions: 只用 (s,) 或 (1,s) 的形状即可广播
        """把每个位置的token索引整理成一个序列 便于计算相位与广播"""
        if token_positions is None:
            # 只要一个共享的位置序列即可
            token_positions = torch.arange(S, device=x.device)
        else:
            # 如果传进来是(B, S), 取一行或检查每行一致
            if token_positions.dim() == 2:
                token_positions = token_positions[0]
        
        # 2) 相位：(s, Dh/2)
        """theta[s] 就是位置 s 的 token 在 每一对维度（偶/奇一对）上要用的旋转角度集合"""
        theta = einsum(token_positions, self.inv_freq, "s, dk_half -> s dk_half")  # (s, Dh/2)，本质是做外积

        # 3) cos/sin: (s, Dh) -> (1, 1, s, Dh) 以便对(B, H, S, Dh)广播
        # repeat_interleave 的意思是：沿某个维度，把每个元素重复多次。
        cos = theta.cos().repeat_interleave(2, dim=-1) # (s, Dh)
        sin = theta.sin().repeat_interleave(2, dim=-1) # (s, Dh)

        # 动态加前置维度: cos/sin -> (1, ...,1,s,Dh) 来匹配x: (..., s, Dh)，这样能直接广播 x * cos + rorated_x * sin
        # 这里的 leading_dims 是为了适应输入 x 可能有不同数量的前置维度（batch size, head数等）
        leading_dims = (1,) * (x.dim() - 2)
        cos = cos.view(*leading_dims, S, Dh).to(dtype=x.dtype)
        sin = sin.view(*leading_dims, S, Dh).to(dtype=x.dtype)

        # 4) 旋转90度
        rotated_x = self.rotate_tensor(x)  # 保证只在最后一维做偶/奇位对调

        # 5) 应用 RoPE 公式：原本向量 *cos + 旋转90度的向量 * sin 相当于在二维平面上旋转
        return x * cos + rotated_x * sin
    
    def rotate_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """create a rotated tensor (x_2k, x_2k+1) -> (-x_2k+1, x_2k)"""
        # transformed x from (..., d_k) to (..., d_k/2, [x_2k, x_2k+1])
        x = rearrange(x, "... (s r) -> ... s r", r=2)

        x_even, x_odd = x.unbind(dim=-1)  # (..., d_k/2)

        # exchange even and get the inversed number of odd
        """相当于旋转90度: (x_2k, x_2k+1) -> (-x_2k+1, x_2k)"""
        x = torch.stack((-x_odd, x_even), dim=-1) # (..., d_k/2, 2)
        return rearrange(x, "... s r -> ... (s r)")  # (..., d_k)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute the softmax of the input tensor along the specified dimension."""
    
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x = torch.exp(x)
    return x / torch.sum(x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    queries: torch.Tensor, # (batch_size, ..., queries, d_k) cross attention时queries和keys/values长度可以不一样
    keys: torch.Tensor,  # (batch_size, ..., keys, d_k)
    values: torch.Tensor,  # (batch_size, ..., values, d_v)  keys and values have the same length
    mask: torch.Tensor | None = None  # (..., queries, keys)
) -> torch.Tensor:  # (batch_size, ... d_v)
    d_k = queries.size(-1)
    scores = einsum(queries, keys, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k ** 0.5) 

    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))

    atten_weights = softmax(scores, dim=-1)  # (..., queries, keys)
    output = einsum(atten_weights, values, "... queries keys, ... keys d_v -> ... queries d_v")
    return output

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, 
        d_model, num_heads, 
        position_embedding: nn.Module = RoPE, max_seq_len = None, theta = None, token_positions = None, 
        device = None, dtype = None, use_causal_mask = True
    ):
        """
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        use_causal_mask: bool Whether to apply causal masking.
        """
        super().__init__()
        self.pe = None
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_causal_mask = use_causal_mask
        assert d_model % num_heads == 0, 'number of heads donen\' match d_model'

        self.d_k = d_model // num_heads
        self.w_q = Linear(self.d_model, self.d_model, device, dtype)
        self.w_k = Linear(self.d_model, self.d_model, device, dtype)
        self.w_v = Linear(self.d_model, self.d_model, device, dtype)
        self.w_o = Linear(self.d_model, self.d_model, device, dtype)

        # 只有传入的三个与RoPE相关的参数都非None时才表示需要用RoPE
        if max_seq_len is not None and theta is not None and position_embedding is not None:
            self.pe = position_embedding(self.d_k, theta, max_seq_len, device)
        self.token_positions = token_positions
    
    def causal_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_i = self.w_q(x)
        k_i = self.w_k(x)
        v_i = self.w_v(x)
        q_i = rearrange(q_i, "B S (n_head d_k) -> B n_head S d_k", n_head=self.num_heads)
        k_i = rearrange(k_i, "B S (n_head d_k) -> B n_head S d_k", n_head=self.num_heads )
        v_i = rearrange(v_i, "B S (n_head d_k) -> B n_head S d_k", n_head=self.num_heads)
        
        if self.pe is not None:
            q_i = self.pe(q_i, self.token_positions)
            k_i = self.pe(k_i, self.token_positions)
        
        mask = None
        if self.use_causal_mask:
            mask = self.causal_mask(q_i.size(-2)).to(device=q_i.device)
        
        atten_score = scaled_dot_product_attention(q_i, k_i, v_i, mask)
        atten_score = rearrange(atten_score, "B n_head S d_k -> B S (n_head d_k)")
        output = self.w_o(atten_score)
        return output
    



