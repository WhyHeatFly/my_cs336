import os
import math
import mmap
import regex as re
import multiprocessing as mp
from typing import List, Dict, Tuple, Iterable
from collections import Counter, defaultdict
from tqdm import tqdm
import time
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def gpt2_bytes_to_unicode() -> Dict[int, str]: # 0-255单字节映射到可编码的unicode字符
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    This modifies UTF-8 to make sure that all bytes are mapped to
    characters that are valid in UTF-8.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

_BYTES_TO_UNICODE = gpt2_bytes_to_unicode()
_UNICODE_TO_BYTES = {v : k for k, v in _BYTES_TO_UNICODE.items()}

def bytes_to_token_str(b: bytes) -> str:
    # 把任意bytes转成"remap unicode"串
    return "".join(_BYTES_TO_UNICODE[x] for x in b)

def token_str_to_bytes(s: str) -> bytes:
    # 把"remap unicode"串转回bytes
    return bytes([_UNICODE_TO_BYTES[c] for c in s])

def pre_tokenization(s: str, special_tokens: List[str]) -> list[str]:
    if not special_tokens:
        return re.findall(PAT, s)

    tokens = sorted(special_tokens, key=len, reverse=True)
    union = "|".join(re.escape(token) for token in tokens)
    parts = re.split(f"({union})", s)

    out = []
    st = set(special_tokens)
    for part in parts:
        if not part:
            continue
        if part in st:
            out.append(part)  # special token 保留
        else:
            out.extend(re.findall(PAT, part))
    return out

def _init_vocab(special_tokens: List[str]) -> Dict[int, bytes]:
    vocab = {}
    idx = 0
    for s in special_tokens:
        vocab[idx] = s.encode("utf-8")
        idx += 1
    for i in range(256):
        vocab[idx] = bytes([i])
        idx += 1
    return vocab

def word_2_byte(word: str) -> Tuple[bytes, ...]:
    b = word.encode("utf-8")
    return tuple(bytes([x]) for x in b)

def _pairs_in_seq(seq: tuple[bytes, ...], special_bytes_set: set[bytes]):
    for a, b in zip(seq[:-1], seq[1:]):
        if a in special_bytes_set or b in special_bytes_set:
            continue
        yield (a, b)

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

# ---------------------------
# 并行：worker 侧做“预分词+Counter”
# ---------------------------
_worker_special_tokens = None

def _init_worker(special_tokens: List[str]):
    # Windows spawn 下需要 initializer 给全局变量赋值
    global _worker_special_tokens
    _worker_special_tokens = special_tokens

def _count_chunk_worker(text_chunk: str) -> Counter:
    """
    对一个文本块做：
    1) pre_tokenization
    2) 统计 seq_counter（规则：special token -> (bytes,) ; 普通 -> tuple(bytes)）
    返回局部 Counter
    """
    st = _worker_special_tokens
    toks = pre_tokenization(text_chunk, st)
    c = Counter()
    st_set = set(st)
    for w in toks:
        if w in st_set:
            c[(w.encode("utf-8"),)] += 1
        else:
            c[word_2_byte(w)] += 1
    return c

# ---------------------------
# mmap 分块读取：保证不会切断 utf-8
# ---------------------------
def _iter_text_chunks_mmap(
    file_path: str | os.PathLike,
    chunk_bytes: int = 32 * 1024 * 1024,  # 32MB
):
    """
    mmap 按块迭代文本，并尽量复现 Python 文本模式的 universal newline 行为：
    - 不把 \r\n 切断
    - 把 \r\n 和 \r 统一成 \n
    """
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            n = len(mm)
            pos = 0
            carry = b""

            while pos < n:
                end = min(pos + chunk_bytes, n)

                # 关键：避免把 \r\n 拆成两块（chunk1 末尾是 \r，chunk2 开头是 \n）
                if end < n and end > 0 and mm[end - 1] == 13 and mm[end] == 10:  # 13=\r, 10=\n
                    end += 1

                raw = carry + mm[pos:end]
                pos = end

                # 处理 UTF-8 截断：把不完整字节留到下一块
                try:
                    s = raw.decode("utf-8")
                    carry = b""
                except UnicodeDecodeError as e:
                    valid = raw[:e.start]
                    carry = raw[e.start:]
                    s = valid.decode("utf-8", errors="ignore")

                if not s:
                    continue

                # 复现文本模式：统一换行
                s = s.replace("\r\n", "\n").replace("\r", "\n")
                # 统一异常换行符（LS/PS）
                s = s.replace("\u2028", "\n").replace("\u2029", "\n")

                yield s

            if carry:
                s = carry.decode("utf-8", errors="ignore")
                s = s.replace("\r\n", "\n").replace("\r", "\n")
                s = s.replace("\u2028", "\n").replace("\u2029", "\n")
                if s:
                    yield s

# ---------------------------
# 主函数：并行构建 seq_counter + 原逻辑合并
# ---------------------------
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = 8,
    chunk_bytes: int = 32 * 1024 * 1024,  # 分块大小，默认32MB
    pool_chunksize: int = 4,  # 给进程池的chunksize
    **kwargs,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

    vocab = _init_vocab(special_tokens)
    special_bytes_set = {s.encode("utf-8") for s in special_tokens}

    # ---------- 1) mmap + 多进程并行统计 seq_counter ----------
    # 说明：我们不再把全文件读入内存，而是迭代 chunk；
    # 每个 chunk 发给 worker 得到局部 Counter，然后主进程累加。
    seq_counter = Counter()

    # 预估 chunk 数用于 tqdm（不精确也没关系）
    file_size = os.path.getsize(input_path)  # 文件大小（字节）
    est_total = max(1, math.ceil(file_size / chunk_bytes))  # 预估块数

    # Windows 上建议使用 spawn，并且要在 main 里调用
    ctx = mp.get_context("spawn")  # 以"spawn"方式启动子进程

    with ctx.Pool(
        processes=num_processes,
        initializer=_init_worker,
        initargs=(special_tokens,),
    ) as pool:
        chunk_iter = _iter_text_chunks_mmap(input_path, chunk_bytes=chunk_bytes)

        # imap 可以边产生边消费，内存更稳
        for local_counter in tqdm(
            pool.imap(_count_chunk_worker, chunk_iter, chunksize=pool_chunksize),
            total=est_total,
            desc="Counting (pre-tokenize + freq)",
            unit="chunk",
            mininterval=0.5,
        ):
            seq_counter.update(local_counter)

    # ---------- 2) 后面保持你原逻辑：构建 pair_cnt/pair_to_words + 合并 ----------
    seqs = list(seq_counter.keys())
    freqs = [seq_counter[s] for s in seqs]

    pair_cnt = Counter()
    pair_to_words = defaultdict(set)

    for wi, seq in enumerate(seqs):
        for p in _pairs_in_seq(seq, special_bytes_set):
            pair_cnt[p] += freqs[wi]
            pair_to_words[p].add(wi)

    merges: List[Tuple[bytes, bytes]] = []

    def pick_best_pair():
        # 频次最大优先，频次相同按字典序最大（保持你原规则）
        return max(pair_cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]

    target_merges = vocab_size - len(vocab)
    pbar = tqdm(total=max(0, target_merges), desc="Training BPE merges", unit="merge", mininterval=0.5)

    while len(vocab) < vocab_size and pair_cnt:
        best = pick_best_pair()
        a, b = best
        merged = a + b

        merges.append(best)
        vocab[len(vocab)] = merged

        affected = list(pair_to_words.get(best, ()))
        if not affected:
            pair_cnt.pop(best, None)
            pair_to_words.pop(best, None)
            pbar.update(1)
            continue

        for wi in affected:
            old_word = seqs[wi]
            word_freq = freqs[wi]

            for p in _pairs_in_seq(old_word, special_bytes_set):
                pair_cnt[p] -= word_freq
                if pair_cnt[p] <= 0:
                    pair_cnt.pop(p, None)
                pair_to_words[p].discard(wi)

            new = _merge_seq(old_word, best, merged)
            seqs[wi] = new

            for p in _pairs_in_seq(new, special_bytes_set):
                pair_cnt[p] += word_freq
                pair_to_words[p].add(wi)

        pair_to_words.pop(best, None)
        pair_cnt.pop(best, None)

        pbar.update(1)

    pbar.close()
    return vocab, merges

def save_vocab_and_merges(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike
):
    # 1. 保存词汇表(json格式)
    # 有些字节不是合法的 utf-8 单字节，因此在保存的时候 errors='ignore' 会吞掉这些非法字节，导致编码失败
    # 因此采用gpt2的bytes_to_token_str方法进行转换，转成可编码的unicode字符串
    vocab_str = {idx: bytes_to_token_str(token) for idx, token in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=2)
    
    # 2. 保存合并规则(文本格式)
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            part1 = bytes_to_token_str(a)
            part2 = bytes_to_token_str(b)
            f.write(f"{part1} {part2}\n")

if __name__ == "__main__":

    train_path = "../../../../data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File not found: {train_path}")
    
    # 训练模型
    print("🚀 开始训练")
    start_time = time.time()
    vocab, merges = run_train_bpe(train_path, vocab_size, special_tokens)
    print(f"✅ 训练完成, 用时 {time.time() - start_time:.2f} 秒,  {(time.time() - start_time)/60:.2f} 分钟")

    # 保存结果
    output_dir = "./bpe_output"
    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, "vocab_on_tinystories_gpt2.json")
    merges_path = os.path.join(output_dir, "merges_on_tinystories_gpt2.txt")
    save_vocab_and_merges(vocab, merges, vocab_path, merges_path)
    print(f"✅ 结果已保存到目录: {output_dir}")
