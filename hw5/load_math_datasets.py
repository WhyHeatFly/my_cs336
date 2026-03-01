import os

os.environ['HF_DATASETS_CACHE'] = './data/huggingface_cache'

from datasets import load_dataset

print("Downloading Countdown-Tasks-3to4 dataset...")
ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")

print("\nDataset downloaded successfully!")
print(f"Dataset info: {ds}")
print(f"Number of examples: {len(ds['train'])}")
print(f"\nFirst example:")
print(ds['train'][0])
