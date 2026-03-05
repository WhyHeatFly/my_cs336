from typing import Literal

import einops
import torch


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.
    
    Args:
    reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the ground
        truths, producing a dict with keys "reward", "format_reward", and "answer_reward".
    rollout_responses: list[str] Rollouts from the policy. The length of this list is 
        rollout_batch_size = n_prompts_per_rollout_batch * group_size.
    repeated_ground_truths: list[str] The ground truths for the examples. The length of this list is rollout_batch_size,
        because the ground truth for each example is repeated group_size times.
    group_size: int Number of responses per question (group).
    advantage_eps: float Small constant to avoid division by zero in normalization.
    normalize_by_std: bool if True, divide by the per-group standard deviation; otherwise subtract only the group mean.

    Returns:
        advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout response.
        raw_rewards shape (roloout_batch_size,). Unnormalized rewards for each rollout response.
        metadata your choice of other statistics to log(e.g.mean,std,max/min of rewards).
    """
    assert len(rollout_responses) == len(repeated_ground_truths), 'invalid input with inequal of labels and responses'

    # Compute raw rewards for each response
    raw_rewards_list = []
    for text_response, text_truth in zip(rollout_responses, repeated_ground_truths):
        reward_info = reward_fn(text_response, text_truth)
        r = reward_info['reward']
        raw_rewards_list.append(r)

    # Convert to tensor and reshape to (n_groups, group_size)
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    n_gourps = len(rollout_responses) // group_size
    grouped_rewards = raw_rewards.reshape(n_gourps, group_size)

    # Compute mean and std within each group (along dim=1)
    mean_r = torch.mean(grouped_rewards, dim=1, keepdim=True)  # (n_groups, 1)
    std_r = torch.std(grouped_rewards, dim=1, keepdim=True)    # (n_groups, 1)

    # Normalize within each group
    unnormalized_A = grouped_rewards - mean_r
    normalized_A = unnormalized_A / (std_r + advantage_eps)

    # Choose which advantages to return
    if normalize_by_std:
        A = normalized_A
    else:
        A = unnormalized_A
    
    # Flatten back to (rollout_batch_size,)
    A_flat = A.reshape(-1)
    raw_rewards_flat = raw_rewards

    return (
        A_flat,
        raw_rewards_flat,
        {
            'mean': mean_r.mean().item(), # overall mean of group means
            'std': std_r.mean().item(),   # overall mean of group stds
            'min': raw_rewards.min().item(),
            'max': raw_rewards.max().item()
        }
    )

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either the raw reward
        or an already-normalized advantage
    Args:
        raw_rewards_or_advantaged: torch.Tensor Shape (batch_size, 1), scalar reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for each token.
    
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per_token policy_gradient loss (to
        be aggregated across the batch and sequence dimensions in the training loop).
    """
    raw_rewards_or_advantages = einops.repeat(raw_rewards_or_advantages, 'b 1 -> b s', s=policy_log_probs.shape[-1])
    loss = - policy_log_probs * raw_rewards_or_advantages
    metadata = {}

    return (
        loss,
        metadata
    )

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    """
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs from the old policy.
        cliprange: float Clip parameter
    
    Returns:
        loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped loss.
        metadata logging whether each token was clipped or not
    """
    ratio = torch.exp(policy_log_probs - old_log_probs) # 新策略相对旧策略的概率比值 (batch_size, sequence_length)
    advantages = einops.repeat(advantages, 'b 1 -> b s', s=policy_log_probs.shape[-1])

    cliprange = einops.repeat(torch.tensor(cliprange).unsqueeze(0), '1 -> b s', b=ratio.shape[0], s=ratio.shape[1])

    # torch.where(condition, x, y) condition为True, 用x，否则，用y.
    def clip_func(ratio, cliprange):
        clip_scaler = torch.where(ratio > 1.0+cliprange, 1.0+cliprange, ratio) # 限制策略增加某动作概率的幅度不超过 1+ϵ
        clip_scaler = torch.where(clip_scaler < 1.0-cliprange, 1.0-cliprange, clip_scaler) # 限制策略降低某动作概率的幅度不超过 1−ϵ
        return clip_scaler
    
    clip_scaler = clip_func(ratio, cliprange).to(device=advantages.device)
    return (
        - torch.min(ratio*advantages, clip_scaler*advantages),
        {
            'ratio': ratio,
            'clip_scaler': clip_scaler,
        }
    )

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the policy being trained
        loss_type: desired policy-gradient loss type
        raw_rewards: Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantaged: Required for "reinforce_with_baseline" and "grpo_clip"; shape (batch_size, 1).
        old_log_probs: Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange: Required for "grpo_clip"
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss
        metadata
    """
    assert loss_type in ("no_baseline", "reinforce_with_baseline", "grpo_clip"), "wrong input loss type"
    if loss_type == "no_baseline":
        assert raw_rewards is not None, 'please input raw_rewards'
        loss_info = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, 'please input advantages'
        loss_info = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    else:
        assert advantages is not None and old_log_probs is not None and cliprange is not None, 'please fill in all required args, including advantages, old_log_probs and cliprange'
        loss_info = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

    loss, metadata = loss_info[0], loss_info[1]
    return (loss, metadata)

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    
    """
    Compute the mean of tensor along a given dimension, considering only those elements where mask == 1
    """
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """ Execute a forward-and-backward pass on a microbatch"""

    loss_per_token, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    l_mean = masked_mean(loss_per_token, response_mask, ) # batch dimension's loss
    l_mean /= gradient_accumulation_steps
    l_mean.backward()
    
    return (l_mean, metadata)