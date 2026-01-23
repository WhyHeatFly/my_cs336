import torch
from torch import Tensor
from jaxtyping import Float, Int
import math
from collections.abc import Iterable

def cross_entropy(out_logits: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]) -> Float[Tensor, ""]:
    """
    cross_entropy 的 Docstring: 计算交叉熵
    
    :param out_logits: Output of the Linear of the transformer, Shape is like (batch_size, vocab_size)
    :type out_logits: torch.Tensor
    :param targets: We are using next-word prediction, so the target is the next word of out input, shape is like (batch_size)
    :type targets: torch.Tensor
    :return: 说明
    :rtype: Tensor
    """
    # unsqueeze(dim) 给张量插入一个维度，dim表示在这个维度插入
    # gather(dim, index) 沿某个维度 把index索引对应的值捞出来 
    get_logits = out_logits.gather(dim=-1, index=targets.unsqueeze(-1)) # 得到对应targets的token的概率
    logsumexp = torch.logsumexp(input=out_logits, dim=-1, keepdim=True)  # 得到logsumexp
    loss = -get_logits + logsumexp  # (batch_size, 1)

    return torch.mean(loss, dim=0, keepdim=True)

def learning_rate_schedule(
    iter: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    assert cosine_cycle_iters > warmup_iters, "Invalid input for iteration striction"

    if iter < warmup_iters:
        return iter * max_learning_rate / warmup_iters
    elif warmup_iters <= iter <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos((iter - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    # l2范数就是把所有参数的梯度当成一个长向量以后，做“平方和再开根号”
    eps = 1e-6
    # 参数p不是单个标量元素，而是一个参数张量（torch.nn.Parameter），通常形状是向量/矩阵/更高维张量。
    grads = [p.grad for p in parameters if p.grad is not None]
    L2_norm = 0.0
    for grad in grads:
        L2_norm += grad.detach().pow(2).sum()
    L2_norm = torch.sqrt(L2_norm)
    if L2_norm >= max_l2_norm:
        for grad in grads:
            grad.data *= max_l2_norm/(L2_norm + eps)
