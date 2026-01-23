import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math

class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data   # Get the gradient of loss with respect to p
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place
                state["t"] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):

    def __init__(self, params: Iterable[torch.nn.parameter.Parameter], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = {
            "alpha": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            alpha = group['alpha']
            beta1, beta2 = group['beta1'], group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                #Embedding layer always uses sparse martix. But AdamW doesn't support it 稀释梯度
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")
                
                # state = self.state[p]
                # prev_m = state.get('m', torch.zeros_like(grad))
                # state['m'] = beta1 * prev_m + (1 - beta1) * grad
                # prev_v = state.get('v', torch.zeros_like(grad))
                # state['v'] = beta2 * prev_v + (1 - beta2) * torch.square(grad)
                # t = state.get('t', 1)
                # alpha_t = alpha * math.sqrt(1 - beta2 ** t) / (1 - beta1**t)
                # p.data -= alpha_t * state['m'] / (torch.sqrt(state['v']) + eps)
                # p.data -= alpha * weight_decay * p.data
                # state['t'] = t + 1

                # --- init state --- 
                state = self.state[p]
                if len(state) == 0:
                    state['t'] = 1
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m, v, t = state['m'], state['v'], state['t']
                m.mul_(beta1).add_(grad, alpha= 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value= 1 - beta2)
                alpha_t = alpha * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                dnorm = v.sqrt().add_(eps)
                p.data.addcdiv_(m, dnorm, value=-alpha_t)

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-alpha * weight_decay)
                state['t'] += 1
                


        return loss


def main():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e3)

    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()

if __name__ == "__main__":
    main()