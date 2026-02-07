import torch
import math
import einops

class FlashAttentionPytorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        device = q.device
        B, N_q, d_model = q.shape
        _, N_k, _ = k.shape
        tile_size = 64
        T_q = math.ceil(N_q / tile_size)
        T_k = math.ceil(N_k / tile_size)
        O = torch.empty((B, N_q, d_model), device=device, dtype=torch.float32)
        L = torch.empty((B, N_q), device=device, dtype=torch.float32)

        for batch in range(B):
            Q_b, K_b, V_b = q[batch], k[batch], v[batch]
            for i in range(T_q):
                q_start, q_end = i * tile_size, min((i + 1) * tile_size, N_q)
                tiled_q_n = q_end - q_start
                Q_i = Q_b[q_start:q_end, :]
                O_i = torch.zeros((tiled_q_n, d_model), device=device, dtype=torch.float32)
                l_i = torch.zeros((tiled_q_n,), device=device, dtype=torch.float32)
                m_i = torch.full((tiled_q_n,), float('-inf'), device=device, dtype=torch.float32)

                for j in range(T_k):
                    k_start, k_end = j * tile_size, min((j + 1) * tile_size, N_k)
                    tiled_k_n = k_end - k_start
                    K_j, V_j = K_b[k_start:k_end, :], V_b[k_start:k_end, :]
                    S_i_j = einops.einsum(Q_i, K_j, "q_tile d, k_tile d -> q_tile k_tile")
                    S_i_j *= d_model**(-0.5)
                    if is_causal:
                        # 确保掩码注意力的正确性
                        diag = q_start - k_start  
                        causal_mask = torch.tril(torch.ones((tiled_q_n, tiled_k_n), dtype=torch.bool, device=device), diagonal=diag, )  
                        S_i_j = torch.masked_fill(S_i_j, ~causal_mask, float('-inf'))
                    
                    # online softmax
                    m_i_new = torch.maximum(m_i, S_i_j.max(1).values)
                    P_i = torch.exp(S_i_j - m_i_new.unsqueeze(-1))
                    l_i = torch.exp(m_i - m_i_new) * l_i + torch.sum(P_i, dim=1)
                    O_i = torch.diag(torch.exp(m_i - m_i_new)) @ O_i + P_i @ V_j
                    m_i = m_i_new

                O_i = torch.diag(1.0 / l_i) @ O_i  # 归一化处理
                l_i = m_i + torch.log(l_i)  # 存储logsum(exp) 为后向传播做准备
                # 写入全局内存(HBM)
                O[batch, q_start:q_end, :] = O_i.to(q.dtype)
                L[batch, q_start:q_end] = l_i
        
        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        raise NotImplementedError
    