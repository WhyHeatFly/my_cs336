from typing import List, Callable
import random
from pathlib import Path

from get_tokenizer import Tokenizer

SPECIAL = "<|endoftext|>"

# 计算压缩比
def compression_ratio_bytes_per_token(
    docs: List[str],
    encode_fn: Callable[[str], List[int]],
) -> float:
    total_bytes, total_tokens = 0, 0
    for doc in docs:
        b = len(doc.encode('utf-8'))
        ids = encode_fn(doc)
        t = len(ids)
        if t == 0:
            continue
        total_bytes += b
        total_tokens += t
    if total_tokens == 0:
        raise RuntimeError("No tokens were produced from the documents.")
    return  total_bytes / total_tokens

def sample_docs_from_file(
    file_path: str,
    n_docs: int = 10,
    seed: int = 42,
    special_token: str = SPECIAL,
    max_char_per_doc: int = 20000,    
) -> List[str]:
    
    random.seed(seed)
    path = Path(file_path)
    assert path.exists(), f"File {file_path} does not exist."

    docs: List[str] = []
    buf = ""

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            buf += line
            
            while special_token in buf:
                doc, buf = buf.split(special_token, 1)
                doc = doc.strip()
                if doc:
                    docs.append(doc[:max_char_per_doc])
                
                if len(docs) >= 5000:
                    break
            if len(docs) >= 5000:
                break
    return random.sample(docs, n_docs)
            

def main():
    # 数据路径
    tiny_story_path = "../../../../data/TinyStoriesV2-GPT4-train.txt"
    owt_path = "../../../../data/owt_train.txt/owt_train.txt"

    
    tiny_vocab_path = "../train_bpe/bpe_output/vocab_on_tinystories_gpt2.json"
    owt_vocab_path = "../train_bpe/bpe_output/vocab_on_owt_gpt2.json"
    tiny_merges_path = "../train_bpe/bpe_output/merges_on_tinystories_gpt2.txt"
    owt_merges_path = "../train_bpe/bpe_output/merges_on_owt_gpt2.txt"

    
    # 采样10篇文档
    tiny_docs = sample_docs_from_file(tiny_story_path, n_docs=10, seed=42)
    owt_docs = sample_docs_from_file(owt_path, n_docs=10, seed=42)
    
    tiny_tokenizer = Tokenizer.from_files(tiny_vocab_path, tiny_merges_path, ["<|endoftext|>"])
    owt_tokenizer = Tokenizer.from_files(owt_vocab_path, owt_merges_path, ["<|endoftext|>"])

    tiny_ratio_on_tiny_text = compression_ratio_bytes_per_token(tiny_docs, tiny_tokenizer.encode)
    tiny_ratio_on_owt_text = compression_ratio_bytes_per_token(owt_docs, tiny_tokenizer.encode)
    owt_ratio_on_tiny_text = compression_ratio_bytes_per_token(tiny_docs, owt_tokenizer.encode)
    owt_ratio_on_owt_text = compression_ratio_bytes_per_token(owt_docs, owt_tokenizer.encode)
    print(f"TinyStories Tokenizer on TinyStories Text: {tiny_ratio_on_tiny_text:.2f} bytes/token")
    print(f"TinyStories Tokenizer on OWT Text: {tiny_ratio_on_owt_text:.2f} bytes/token")
    print(f"OWT Tokenizer on TinyStories Text: {owt_ratio_on_tiny_text:.2f} bytes/token")
    print(f"OWT Tokenizer on OWT Text: {owt_ratio_on_owt_text:.2f} bytes/token")
    
if __name__ == "__main__":
    main()