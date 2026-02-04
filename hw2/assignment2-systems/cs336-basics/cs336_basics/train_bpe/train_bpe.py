import os
from typing import List, Dict, Tuple
import regex as re
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenization(s: str, special_tokens: List[str]) -> list[str]:
    if not special_tokens:
        return re.findall(PAT, s)

    tokens = sorted(special_tokens, key=len, reverse=True)  # 按长度降序排序
    union = "|".join(re.escape(token) for token in tokens)
    # parts = re.split(union, s)  # 没有捕获组， 不保留分隔符
    # ({union}) 表示捕获分组，保留分隔符，即保留特殊token
    parts = re.split(f"({union})", s)  # 保留分隔符，即特殊token

    out = []
    st = set(special_tokens)
    for part in parts:
        if not part:
            continue
        if part in st:
            # 保留special token 作为一整个整体token（不要skip）
            out.append(part)
        else:
            # 普通文本，继续用PAT拆分
            out.extend(re.findall(PAT, part))
    
    return out

def _init_vocab(special_tokens: List[str]) -> Dict[int, bytes]:
    vocab = {}
    idx = 0
    for s in special_tokens:
        vocab[idx] = s.encode("utf-8")  # 存为bytes类型
        idx += 1
    
    for i in range(256):
        b = bytes([i])
        vocab[idx] = b
        idx += 1
    return vocab

def word_2_byte(word: str) -> Tuple[bytes, ...]:
    b = word.encode('utf-8')
    return tuple(bytes([x]) for x in b)

def _pairs_in_seq(seq: tuple[bytes, ...], special_bytes_set: set[bytes]):
    for a, b in zip(seq[:-1], seq[1:]):
        if a in special_bytes_set or b in special_bytes_set:
            continue
        yield (a, b)  # yield用在函数里，把函数变成一个生成器对象，每次调用生成器的__next__()方法时，函数会从上次yield语句后继续执行，直到遇到下一个yield语句

def _merge_seq(seq: tuple[bytes, ...], pair: Tuple[bytes, bytes], merged: bytes):
    a, b = pair
    out = []
    i, n = 0, len(seq)
    while i < n:
        if i < n - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    
    return tuple(out)

def run_train_bpe(
        input_path: str | os.PathLike, 
        vocab_size: int, 
        special_tokens: List[str],
        **kwargs
) -> Tuple[Dict[int, str], List[Tuple[str, str]]]:
    
    vocab = _init_vocab(special_tokens)
    special_bytes_set = {s.encode('utf-8') for s in special_tokens}

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    pre_tokenization_words = pre_tokenization(text, special_tokens)

    seq_counter = Counter()  # 统计所有byte序列出现的频次, Counter()是一个字典的子类计数器字典

    for word in pre_tokenization_words:
        if word in special_tokens:
            seq_counter[(word.encode('utf-8'),)] += 1  # special token 作为整体处理
        else:
            seq_counter[word_2_byte(word)] += 1  # 普通文本拆分为byte序列处理

    seqs = list(seq_counter.keys())  # seqs是所有不同的byte序列, 例如: [(b'h', b'e', b'l', b'l', b'o'), (b'w', b'o', b'r', b'l', b'd')]
    freqs = [seq_counter[s] for s in seqs]  # freqs是对应的频次列表

    # --- 初始化pair counts + inverted index ---
    pair_cnt = Counter()
    pair_to_words = defaultdict(set)

    for wi, seq in enumerate(seqs):  # wi是word index, seq是对应的byte序列
        for p in _pairs_in_seq(seq, special_bytes_set):
            pair_cnt[p] += freqs[wi]
            pair_to_words[p].add(wi)  # 记录包含该pair的word索引

    # --- 迭代合并 ---
    merges = []

    def pick_best_pair():
        # 先按照频次最大的选，再按照字典序最大的选
        return max(pair_cnt.items(), key = lambda kv: (kv[1], kv[0]))[0]
    
    while len(vocab) < vocab_size and pair_cnt:
        best = pick_best_pair()
        a, b = best
        merged = a + b

        merges.append(best)
        vocab[len(vocab)] = merged

        affected = list(pair_to_words.get(best, ()))  # 获取所有包含该pair的word索引

        if not affected:
            pair_cnt.pop(best, None)
            pair_to_words.pop(best, None)
            continue
        
        # 更新所有受影响的word
        for wi in affected:
            old_word = seqs[wi]
            word_freq = freqs[wi]

            # 移除旧pair计数
            for p in _pairs_in_seq(old_word, special_bytes_set):
                pair_cnt[p] -= word_freq
                if pair_cnt[p] <= 0:
                    pair_cnt.pop(p, None)
                pair_to_words[p].discard(wi)  # 从pair的word集合中移除该word索引
            
            # merge
            new = _merge_seq(old_word, best, merged)
            seqs[wi] = new

            # 添加新pair计数
            for p in _pairs_in_seq(new, special_bytes_set):
                pair_cnt[p] += word_freq
                pair_to_words[p].add(wi)
            
        # 移除已合并的pair
        pair_to_words.pop(best, None)
        pair_cnt.pop(best, None)
    
    return vocab, merges








 