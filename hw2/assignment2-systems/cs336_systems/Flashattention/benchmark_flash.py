''' Benchmarking script for FlashAttention-2 vs PyTorch attention '''

import torch
import triton.testing
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'cs336_systems'))

from cs336_systems.Flashattention.flash_att_triton import FlashAttentionTriton
import itertools  # 高效地生成各种“迭代组合”,不用写多重循环

def benchmark_attention(impl_name, attention_fn, seq_len, d_model, dtype, batch_size=1, is_causal=True):
    """Benchmark a single attention implementation"""

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    
    # Create random inputs
    q = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)

    # Warmup
    for _ in range(5):
        o = attention_fn(q, k, v, is_causal)
        if o.requires_grad:
            grad_out = torch.randn_like(o)
            o.backward(grad_out)
            q.grad = None
            k.grad = None
            v.grad = None
    
    # Benckmark forward
    def forward_fn():
        return attention_fn(q, k, v, is_causal)
    
    forward_time = triton.testing.do_bench(forward_fn)  # 用 Triton 自带的基准测试工具，测 forward_fn() 的 GPU 执行时间（延迟）

    # Benchmark backward_only
    o = attention_fn(q, k, v, is_causal)
    grad = torch.randn_like(o)

    def backward_only_fn():
        o.backward(grad, retain_graph=True)
        q.grad = k.grad = v.grad = None

    backward_time = triton.testing.do_bench(backward_only_fn)

    # Benchmark end_to_end
    def end_to_end_fn():
        o = attention_fn(q, k, v, is_causal)
        grad_out = torch.randn_like(o)
        o.backward(grad_out)
        q.grad=None
        k.grad=None
        v.grad=None

    # End-to-end time is approximately forward + backward
    end_to_end_time = triton.testing.do_bench(end_to_end_fn)

    return {
        'implementation': impl_name,
        'seq_len': seq_len,
        'd_model': d_model,
        'dtype': str(dtype),
        'forward_ms': forward_time,
        'backward_ms': backward_time,
        'end_to_end_ms': end_to_end_time
    }

def pytorch_attention(q, k, v, is_causal):
    """Reference PyTorch attention implemention"""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = q @ k.transpose(-2, -1) * scale

    if is_causal:
        seq_len = q.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v
    return out

def main():
    # Test configurations
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_models = [16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]

    results = []
    print("Starting benchmark...")
    print(f"Testing {len(seq_lengths)} sequence lengths, {len(d_models)} embedding dims, {len(dtypes)} dtypes")
    print(f"Total configurations: {len(seq_lengths) * len(d_models) * len(dtypes) * 2}")  # *2 for flash and pytorch

    for seq_len, d_model, dtype in itertools.product(seq_lengths, d_models, dtypes):
        print(f"\nTest seq_len={seq_len}, d_model={d_model}, dtype={dtype}")

        try:
            # Benchmark PyTorch attention
            print("   - Pytorch attention...")
            pytorch_result = benchmark_attention(
                'PyTorch', pytorch_attention, seq_len, d_model, dtype
            )
            results.append(pytorch_result)
            print(f"     Forward: {pytorch_result['forward_ms']:.3f}ms, Backward: {pytorch_result['backward_ms']:.3f}ms")
        except RuntimeError as e:
            print(f"  - Pytorch attention OOM: {e}")
            results.append({
                'implementation': 'Pytorch',
                'seq_len': seq_len,
                'd_model': d_model,
                'dtype': str(dtype),
                'forward_ms': 'OOM',
                'backward_ms': 'OOM',
                'end_to_end_ms': 'OOM'
            })

        try:
            # Benchmark FlashAttention
            print("   - FlashAttention-2...")
            flash_result = benchmark_attention(
                'FlashAttention-2', FlashAttentionTriton.apply, seq_len, d_model, dtype
            )
            results.append(flash_result)
            print(f"     Forward: {flash_result['forward_ms']:.3f}ms, Backward: {flash_result['backward_ms']:.3f}ms")
        except RuntimeError as e:
            print(f"  - FlashAttention-2 OOM: {e}")
            results.append({
                'implementation': 'FlashAttention-2',
                'seq_len': seq_len,
                'd_model': d_model,
                'dtype': str(dtype),
                'forward_ms': 'OOM',
                'backward_ms': 'OOM',
                'end_to_end_ms': 'OOM'
            })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv('flash_attention_benchmark.csv', index=False)
    print("\n" + "="*80)
    print("Benchmark complete! Results saved to flash_attention_benchmark.csv")
    print("="*80)

    # Print summary
    print("\nSample results:")
    print(df.head(20).to_string())

if __name__ == "__main__":
    main()