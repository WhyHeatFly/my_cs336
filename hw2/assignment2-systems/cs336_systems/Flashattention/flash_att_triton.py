import triton
import triton.language as tl

import torch
import math

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,  # B_q, B_k
    scale,              # 1/sqrt(d_model)
    D: tl.constexpr,    # d_model
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    # Python 端启动代码
    # grid = (Query分块数量, Batch大小)
    grid = (triton.cdiv(N_q, BLOCK_SIZE_M), batch_size) 

    # 启动内核
    flash_fwd_kernel[grid](
    Q, K, V, ... 
    )
    """
    # 用来确定线程实例处理哪个batch的那一个query块
    query_tile_index = tl.program_id(0)  # Grid 第0维 (第一个参数) 的索引
    batch_index = tl.program_id(1)  # Grid 第1维 (第二个参数) 的索引
    
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    scaler = scale
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    Q_i = tl.load(Q_block_ptr, boundary_check=(1,0))
    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for j in range(T_k):
        K_block_ptr = tl.make_block_ptr(
            K_ptr + stride_kb * batch_index,
            shape=(D, N_KEYS),
            strides=(stride_kd, stride_kk),
            offsets=(0, j * K_TILE_SIZE),
            block_shape=(D, K_TILE_SIZE),
            order=(0, 1)
        )

        V_block_ptr= tl.make_block_ptr(
            V_ptr + stride_vb * batch_index,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(j * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0)
        )

        """
        boundary_check 是一个元组 (Dim0_Check, Dim1_Check):
        1 (True): 检查该维度，超出部分补 0。
        0 (False): 不检查该维度（假设它是安全的）。
        """
        K_j = tl.load(K_block_ptr, boundary_check=(1,0))
        V_j = tl.load(V_block_ptr, boundary_check=(0,1))

        S_ij = tl.dot(Q_i.to(tl.float32), K_j.to(tl.float32)) * scaler
        if is_causal:
            # Causal masking: query position >= key position
            # Use static ranges and add offsets
            q_indices = tl.arange(0, Q_TILE_SIZE)[:, None] + query_tile_index * Q_TILE_SIZE
            k_indices = tl.arange(0, K_TILE_SIZE)[None, :] + j * K_TILE_SIZE
            mask = q_indices >= k_indices
            S_ij = tl.where(mask, S_ij, float('-inf'))

        m_i_new = tl.maximum(m_i, tl.max(S_ij, 1))
        #Scaling the score
        P_ij = tl.exp(S_ij - m_i_new[:, None])
        #Update logexpsum
        m_scaler = tl.exp(m_i - m_i_new)
        l_i = m_scaler * l_i + tl.sum(P_ij, 1)
        #Update O
        o_i = o_i * m_scaler[:, None] + tl.dot(P_ij.to(tl.float32), V_j.to(tl.float32))
        m_i = m_i_new
    
    o_i = o_i * (1.0 / l_i[:, None])
    l_i = tl.log(l_i) + m_i

    tl.store(O_block_ptr, o_i.to(O_ptr.type.element_ty), boundary_check=(1,0))
    # boundary_check 不是“想开就能开”，而是“只能在 Triton 认为可能越界的维度上开”。
    tl.store(L_block_ptr, l_i.to(L_ptr.type.element_ty), boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        batch_size, N_q, D = q.shape
        _, N_k, _ = k.shape
        
        """
        据 hidden dimension (d_model) 的大小，动态调整分块大小,以防止 GPU 的共享内存 (Shared Memory / SRAM) 溢出。
        D越大 分块应该越小
        """
        if D <= 32:
            Q_TILE_SIZE, K_TILE_SIZE = 128, 128
        elif D <= 64:
            Q_TILE_SIZE, K_TILE_SIZE = 64, 64
        else:
            Q_TILE_SIZE, K_TILE_SIZE = 32, 32

        T_q = math.ceil(N_q / Q_TILE_SIZE)
        O_ptr = torch.empty_like(q)
        L_ptr = torch.empty((batch_size, N_q), dtype=torch.float32, device=q.device)

        flash_fwd_kernel[(T_q, batch_size)](
            q, k, v,
            O_ptr, L_ptr,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            O_ptr.stride(0), O_ptr.stride(1), O_ptr.stride(2),
            L_ptr.stride(0), L_ptr.stride(1),
            N_q, N_k,
            D**(-0.5),
            D, Q_TILE_SIZE, K_TILE_SIZE,
            is_causal
        )

        ctx.save_for_backward(q, k, v, O_ptr, L_ptr)
        ctx.i_causal = is_causal

        return O_ptr
    