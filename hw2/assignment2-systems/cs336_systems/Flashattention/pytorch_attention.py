import time
import timeit
import torch
import pandas as pd
from statistics import mean, stdev
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'cs336-basics'))

from cs336_basics.model.modules import scaled_dot_product_attention

batch_size = 8
num_heads = 1
d_models = [16, 32, 64, 128]
seq_lens = [256, 1024, 4096, 8192, 16384]
loop = 100
warm_up = 5

# @torch.compile 不进行图优化和算子融合
def pytorch_attention(d_model, seq_len, device):
    
    f_times, b_times = [], []
    mem_at_backward_start = None

    # 清理初始缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        Q = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32, device=device, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32, device=device, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32, device=device, requires_grad=True)
        
        # warm up process
        for _ in tqdm(range(warm_up), desc=f'[d={d_model}, seq={seq_len}] Warmup', leave=False):
            out = scaled_dot_product_attention(Q, K, V)
            out.sum().backward()
            Q.grad = K.grad = V.grad = None
        
        # forward process
        for _ in tqdm(range(loop), desc=f'[d={d_model}, seq={seq_len}] Forward', leave=False):
            torch.cuda.synchronize()
            start = timeit.default_timer()
            attn = scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize()
            end = timeit.default_timer()
            f_times.append(end - start)
            Q.grad = K.grad = V.grad = None
        
        # 题目要求：测量反向传播开始前的内存 (Memory in use before backward)
        # 此时 attn 已经计算完毕，且梯度图已在内存中
        mem_at_backward_start = torch.cuda.memory_allocated() / (1024 ** 2)

        # backward process
        for _ in tqdm(range(loop), desc=f'[d={d_model}, seq={seq_len}] Backward', leave=False):
            loss = attn.sum()
            torch.cuda.synchronize()
            back_start = timeit.default_timer()
            loss.backward(retain_graph=True)  # 保持图以便循环
            torch.cuda.synchronize()
            back_end = timeit.default_timer()
            b_times.append(back_end - back_start)
            Q.grad, K.grad, V.grad = None, None, None
        del Q, K, V
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print(f"OOM at d={d_model}, seq={seq_len}")
    
    result = {
        "seq_len": seq_len,
        "d_model": d_model,
        "forward_ms_mean": round(mean(f_times) * 1e3, 3) if len(f_times) >= 2 else "OOM",
        "forward_ms_std": round(stdev(f_times) * 1e3, 3) if len(f_times) >= 2 else "OOM",
        "backward_ms_mean": round(mean(b_times) * 1e3, 3) if len(b_times) >= 2 else "OOM",
        "backward_ms_std": round(stdev(b_times) * 1e3, 3) if len(b_times) >= 2 else "OOM",
        "fwd_peak_mem_MB": round(mem_at_backward_start, 1) if mem_at_backward_start else "OOM",
        "batch_size": batch_size,
    }
    return result

def main():
    res = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == 'cuda', "No GPU available!"
    for d_model in d_models:
        for seq_len in seq_lens:
            result = pytorch_attention(d_model, seq_len, device)
            res.append(result)
    res = pd.DataFrame(res)
    with open('attn_benchmark_results.md', 'w') as f:
        f.write(res.to_markdown())

if __name__ == "__main__":
    main()