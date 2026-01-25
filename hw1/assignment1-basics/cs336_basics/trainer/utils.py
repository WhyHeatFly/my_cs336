from typing import IO, BinaryIO
import torch
from torch import Tensor
from jaxtyping import Float, Int
import math
from collections.abc import Iterable
import numpy.typing as npt
import os

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

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    N = len(dataset)
    starts = torch.randint(0, N - context_length, (batch_size,), device=device)
    # 每条序列的位置偏移索引：
    offsets = torch.arange(context_length, device=device)
    idx = starts[:, None] + offsets[None, :]  # (B, m)
    idx_t = idx + 1
    data = torch.as_tensor(dataset, dtype=torch.long).to(device=device)
    inputs = data[idx]
    targets = data[idx_t]

    return (inputs, targets)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    torch.save(
        obj={
            'model_state': model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "iteration": iteration
        },
        f = out
    )

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    ckp = torch.load(src, map_location='cpu')
    model.load_state_dict(ckp['model_state'])
    optimizer.load_state_dict(ckp['optimizer_state'])
    return ckp['iteration']


