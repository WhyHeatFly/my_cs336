import os
import argparse
import torch
from typing import Optional

from cs336_basics.model.transformer import Transformer
from cs336_basics.tokenizer.get_tokenizer import Tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with a trained Transformer language model")

    # Model and tokenizer paths
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--vocab', type=str, required=True, help="Path to tokenizer vocabulary file")
    parser.add_argument('--merges', type=str, required=True, help="Path to tokenizer merges file")

    # Generation parameters
    parser.add_argument('--prompt', type=str, default="Once upon a time", help="Input prompt for generation")
    parser.add_argument('--max_tokens', type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperature for sampling(lower = more deterministic)")
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p threshold for nucleus sampling')
    parser.add_argument('--num_samples', type=int, default=1, help="Number of samples to generate")

    # Device
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, cuda, mps')

    # End-of-sequence token
    parser.add_argument('--eos_token', type=str, default='<|endoftext|>', help='End-of-sequence token')

    return parser.parse_args()

def load_model_and_tokenizer(checkpoint_path: str, vocab_path: str, merges_path: str, device):

    # Import tokenizer
    try:
        tokenizer = Tokenizer.from_files(vocab_path, merges_path)
    except ImportError:
        print("Warning: Could not import Tokenizer. You may need to implement or adjust import path")
        tokenizer = None
    
    # Load checkpoint to get model configuration
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model hyperparameters
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # Default configuration - adjust as needed
        config = {
            'vocab_size': 32000,
            'context_length': 256,
            'num_layers': 4,
            'd_model': 512,
            'num_heads': 16,
            'rope_theta': 10000.0,
            'd_ff': 1344
        }
        print("Warning: Using default model configuration. Adjust as needed.")
    
    # Create model
    model = Transformer(
        vocab_size=config.get('vocab_size', 32000),
        context_length=config.get('context_length', 256),
        num_layers=config.get('num_layers', 4),
        d_model=config.get('d_model', 512),
        num_heads=config.get('num_heads', 16),
        rope_theta=config.get('rope_theta', 10000.0),
        d_ff=config.get('d_ff', 1344)
    ).to(device)
    
    # Load model weights
    if 'model_state' in checkpoint:
        state = checkpoint['model_state']
    else:
        # 兼容某些人保存成 model_state_dict 的情况（可选）
        state = checkpoint.get('model_state_dict', checkpoint)

    # （可选）兼容 DataParallel/DDP 的 'module.' 前缀
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    
    return model, tokenizer

def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:

    # Apply temperature scaling
    scaled_logits = logits / temperature

    max_logits = torch.max(scaled_logits, dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(scaled_logits - max_logits)
    probabilities = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)

    return probabilities

def top_p_sampling(probabilities: torch.Tensor, p: float = 0.9) -> torch.Tensor:

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)

    # Compute cumulative probabilities 
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1) # 沿着某个维度，把前面所有元素一路加起来，输出同形状的张量
    
    # Create mask for tokens to keep (cumulative probability <= p)
    mask = cumulative_probs <= p

    # Alaways keep at least the top token
    mask[..., 0] = True  # 把vocab_size 的第一个元素置为True， 防止全0采样

    filtered_probs = sorted_probs * mask.float()

    # Renormalize
    filtered_probs = filtered_probs / torch.sum(filtered_probs, dim = -1, keepdim=True)

    # Create output tensor with same shape as input
    output_probs = torch.zeros_like(probabilities)
    # scatter（散射写入） 的原地版本（带 _ 表示原地修改 output_probs）
    output_probs.scatter_(-1, sorted_indices, filtered_probs) # sorted_indices 指定要写入的位置，把 filtered_probs 的值写回 output_probs 对应位置。

    return output_probs

def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cpu",
    eos_token: Optional[str] = "<|endoftext|>"
) -> str:
    
    model.eval()

    # Encode the prompt
    prompt_tokens = tokenizer.encode(prompt)
    
    # Convert to tensor and move to device
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Generate tokens one by one
    generated_tokens = prompt_tokens.copy()

    with torch.no_grad():
        for _ in range(max_tokens):

            # Forward pass through model
            logits = model(input_ids)

            # Get logits for the last token
            next_token_logits = logits[0, -1, :]  # shape:(vocab_size,)

            # Apply temperature scaling and softmax
            probabilities = softmax_with_temperature(next_token_logits, temperature)

            # Apply top_p sampling if specified
            if top_p < 1.0:
                probabilities = top_p_sampling(probabilities, top_p)
            
            # Sample next token 
            next_token = torch.multinomial(probabilities, num_samples=1).item()

            # Add to generated sequence
            generated_tokens.append(next_token)

            # Check for end-of-sequence token
            if eos_token is not None:
                decoded_token = tokenizer.decode([next_token])
                if decoded_token.strip() == eos_token.strip():
                    break
            
            # Update input for next iteration
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

            # Truncate if sequence gets too long
            if input_ids.size(1) > model.context_length:
                input_ids = input_ids[:, -model.context_length:]
    
    # Decode the generate tokens
    generated_text = tokenizer.decode(generated_tokens)

    return generated_text

def main():
    args = parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(
            checkpoint_path=args.checkpoint,
            vocab_path=args.vocab,
            merges_path=args.merges,
            device=device
        )
        print("Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return
    
    if tokenizer is None:
        print("Error: Tokenizer could not be loaded. Please check your tokenizer implementation.")
        return
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Generate text
    print(f"\nGenerating {args.num_samples} sample(s) with prompt: '{args.prompt}'")
    print(f"Parameters: max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p}")
    print("-" * 80)

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\nSample {i + 1}:")
            print("-" * 40)
        try:
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                eos_token=args.eos_token
            )

            print(generated_text)
            print("\n" + "=" * 80)

        except Exception as e:
            print(f"Error during generation: {e}")
            break
    
if __name__ == "__main__":
    main()