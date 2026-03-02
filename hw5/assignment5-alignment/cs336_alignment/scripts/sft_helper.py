import torch
import torch.nn.functional as F

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    Tokenize the prompt and output strings,and construct a mask that is 1 for the response tokens and 0 for
    other tokens(prompt or padding).

    Args:
        prompt_strs: list[str] List of prompt strings.
        output_strs: list[str] List of output strings.
        tokenizer:PreTrainedTokenizer Tokenizer to use for tokenization.

    Returns:
        dict[str,torch.Tensor].Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings.Then the returned dictionary should have the
        following keys:
            "input_ids":torch.Tensor of shape(batch_size, max(prompt_and_output_lens)-1):
            the tokenized prompt and output strings,with the final token sliced off.
            "labels":torch.Tensor of shape(batch_size, max(prompt_and_output_lens)-1):
            shifted input ids, i.e.,the input ids without the first token.
            "response_mask":torch.Tensor of shape(batch_size, max(prompt_and_output_lens)-1):
            a mask on the response tokens in the labels
    """
    assert len(prompt_strs) == len(output_strs), "invalid input or label dimensions"
    input_prompts_ids, output_ids = [], []

    for p in prompt_strs:
        p_id = tokenizer.encode(p, add_special_tokens=False)
        input_prompts_ids.append(torch.tensor(p_id))

    for o in output_strs:
        o_id = tokenizer.encode(o, add_special_tokens=False)
        output_ids.append(torch.tensor(o_id))
    
    prompt_and_output_lens = [len(prompt) + len(out) for prompt, out in zip(input_prompts_ids, output_ids)]
    D_output = max(prompt_and_output_lens) - 1

    # padding
    paded_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    input_ids = []
    labels = []
    response_mask = []
    for p_id, o_id in zip(input_prompts_ids, output_ids):
        input_id = torch.cat((p_id, o_id, torch.tensor([tokenizer.eos_token_id])), dim=-1)
        response_m = torch.cat((torch.zeros_like(p_id).to(dtype=torch.bool), torch.ones_like(o_id).to(dtype=torch.bool), torch.tensor([False])), dim=-1)
        slice_input_id = input_id[:-1]
        slice_output_id = input_id[1:]
        slcie_response_m = response_m[1:]
        pad_len = D_output - slice_input_id.shape[0]

        # 对于一维张量 (0, pad_len)表示左边补0，右边补pad_len
        padded_input_id = F.pad(input=slice_input_id, pad=(0, pad_len), value=paded_val)
        padded_output_id = F.pad(input=slice_output_id, pad=(0, pad_len), value=paded_val)
        response_mask_padded = F.pad(input=slcie_response_m, pad=(0, pad_len), value=False)

        input_ids.append(padded_input_id)
        labels.append(padded_output_id)
        response_mask.append(response_mask_padded)

    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'response_mask': torch.stack(response_mask)
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    normed_logits = F.softmax(logits, dim=-1)
    log_p = torch.log(normed_logits)
    return -torch.sum(normed_logits*log_p, dim=-1)

def get_response_log_probs(
    model:torch.nn.Module,
    input_ids:torch.Tensor,
    labels:torch.Tensor,
    return_token_entropy:bool=False,
) -> dict[str, torch.Tensor]:
    """
    Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    pred_logits = model(input_ids).logits
    log_probs_all = F.log_softmax(pred_logits, dim=-1)

    # Gather the log probabilities of the actual tokens in labels
    # labels shape: (batch_size, seq_length)
    # log_probs_all shape: (batch_size, seq_length, vocab_size)
    # We need to gather along the vocab dimension
    labels_expanded = labels.unsqueeze(-1)  # (batch_size, seq_length, 1)
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels_expanded).squeeze(-1)

    if return_token_entropy:
        entropy = compute_entropy(pred_logits)
    else:
        entorpy = None
    
    return {
        'log_probs': log_probs,
        'token_entropy': entropy
    }

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a cnostant,
    considering only the elements with mask value 1."""

    assert normalize_constant != 0, 'invalid constant for normalization!'
    """
    torch.where(condition, x, y) 的意思是：
    如果 condition 为 True → 取 x
    如果 condition 为 False → 取 y
    是一个逐元素 (elementwise) 操作。
    """
    sum_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
    return sum_tensor.sum(dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch"""

    # Forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    grad_tensor = torch.zeros_like(policy_log_probs, device=device)

    cross_entropy = masked_normalize(policy_log_probs, response_mask, dim=-1, normalize_constant=normalize_constant)
    loss = -cross_entropy.mean(dim=-1)
    loss /= gradient_accumulation_steps

    loss.backward()

    return (
        loss,
        {
            'loss': loss.detach()
        }
    )

