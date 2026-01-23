import os
from typing import List, Tuple, Dict
import regex as re
from collections import Counter, defaultdict
import time
from tqdm import tqdm
import json
import math
import multiprocessing  # 多进程支持
import mmap  # 用于内存映射文件

# 正则表达式模式，参考GPT-2的分词规则
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

def _init_vocab(special_tokens: List[str]) -> Dict[int, bytes]:
    """
    _init_vocab: 初始化vocab字典, 包含special tokens和0-255的单字节字符
    
    :param special_tokens: 特殊字节列表，如["<|endoftext|>"]
    :type special_tokens: List[str]
    :return: 初始化后的vocab字典, 键为索引, 值为对应的bytes类型token
    :rtype: Dict[int, bytes]
    """
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

def pre_tokenization(
    text: str,
    special_tokens: List[str]
) -> List[str]:
    """
    pre_tokenization: 将训练文本进行预分词
    首先按照特殊token进行分割并保留特殊token, 然后对非特殊token部分按照gpt2的分割规则进行分词
    
    :param text: 训练文本
    :type text: str
    :param special_tokens: 特殊字节列表，如["<|endoftext|>"]
    :type special_tokens: List[str]
    :return: 经过预分词后的文本列表，如["This", " is", " a", "<|endoftext|>", " test", "."]
    :rtype: List[str]
    """
    if not special_tokens:
        return re.findall(PAT, text)

    tokens = sorted(special_tokens, key=len, reverse=True)  # 按长度降序排序
    union = "|".join(re.escape(token) for token in tokens)  # 构建正则表达式模式, 用于匹配特殊token, re.escape()转义特殊字符
    parts = re.split(f"({union})", text)

    out = []
    st = set(special_tokens)  # 用于快速查找特殊token
    for part in parts:
        if not part:
            continue
        if part in st:
            out.append(part)  # 保留特殊token
        else:
            out.extend(re.findall(PAT, part))  # 对非特殊token部分进行gpt2分词
    
    return out

def word_2_byte(word: str) -> Tuple[bytes, ...]:
    """
    word_2_byte: 将预分词得到的单词转换为对应的bytes序列
    
    :param word: 单词字符串
    :type word: str
    :return: 对应的bytes序列元组, 如"hello" -> (b'h', b'e', b'l', b'l', b'o')
    :rtype: Tuple[bytes, ...]
    """
    b = word.encode('utf-8')
    return tuple(bytes([x]) for x in b)

def _pairs_in_seq(
    seq: tuple[bytes, ...],
    special_bytes_set: set[bytes]
):
    """
    _pairs_in_seq: 生成序列中所有非特殊token的byte对
    
    :param seq: 字节序列, 例如: (b'h', b'e', b'l', b'l', b'o')
    :type seq: tuple[bytes, ...]
    :param special_bytes_set: 特殊字节的bytes集合
    :type special_bytes_set: set[bytes]
    :return: 说明
    :rtype: Tuple[bytes, bytes]
    """
    for a, b in zip(seq[:-1], seq[1:]):
        if a in special_bytes_set or b in special_bytes_set:
            continue
        yield (a, b)

def _merge_seq(
    old_word: tuple[bytes, ...],
    best: Tuple[bytes, bytes],
    merged: bytes
) -> tuple[bytes, ...]:
    """
    _merge_seq: 将旧的字节序列中所有出现的best对合并为merged
    
    :param old_word: 旧的字节序列, 包含要合并的byte对
    :type old_word: tuple[bytes, ...]
    :param best: 要合并的byte对
    :type best: Tuple[bytes, bytes]
    :param merged: 合并后的byte
    :type merged: bytes
    :return: 合并后的字节序列
    :rtype: tuple[bytes, ...]
    """
    a, b = best
    out = []
    i, n = 0, len(old_word)
    while i < n:
        if i < n - 1 and old_word[i] == a and old_word[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(old_word[i])
            i += 1
    
    return tuple(out)
# ------------------
# 并行：worker 侧做“预分词+Counter”
# ------------------

_worker_special_tokens = None  # 进程内的全局变量，worker 进程初始化时赋值

def _init_worker(special_tokens: List[str]):
    global _worker_special_tokens
    _worker_special_tokens = special_tokens

# ------------------
# mmap 分块读取：保证不会切断 utf-8 字符
# ------------------
def _iter_text_chunks_mmap(
    file_path: str | os.PathLike,
    chunk_bytes: int = 32 * 1024 * 1024
):
    """
    mmap 按块迭代文本
    - 不把\r\n等换行符切断
    - 把\r\n 和 \r 统一成 \n
    """
    # "rb" 模式读取二进制，避免换行符转换，读出来是 bytes
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            pos, n = 0, len(mm)
            carry = b""  # 上一块的残留部分

            while pos < n:
                end = min(pos + chunk_bytes, n)

                # 避免把 \r\n 拆成两块（chunk1 末尾是 \r，chunk2 开头是 \n）
                # 因为Windows的换行是\r\n
                if end < n and end > 0 and mm[end - 1] == 13 and mm[end] == 10: # 13=\r, 10=\n
                    end += 1 
                
                raw = carry + mm[pos:end]
                pos = end
                # 处理 utf-8 截断: 把不完整字节留到下一块
                # 因为 utf-8 是变长编码，可能一个字符占1-4个字节，如"中"是3个字节，可能会在中间切断
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
                # 统一异常换行符（LS/PS）为普通换行
                s = s.replace("\u2028", "\n").replace("\u2029", "\n")

                yield s
            
            # 处理最后的残留部分
            if carry:
                s = carry.decode("utf-8", errors="ignore")
                s = s.replace("\r\n", "\n").replace("\r", "\n")
                s = s.replace("\u2028", "\n").replace("\u2029", "\n")
                if s:
                    yield s

def _count_chunk_worker(text_chunk: str) -> Counter:
    """
    对一个文本块做：
    1) pre_tokenization
    2) 统计 seq_counter (规则: special token -> (bytes,) ; 普通 -> tuple(bytes))
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

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = 8,
    chunk_bytes: int = 32 * 1024 * 1024,  # 分块大小，默认32MB
    pool_chunksize: int = 4,  # 给进程池的chunksize
    **kwargs,
) -> Tuple[Dict[int, str], List[Tuple[bytes, bytes]]]:
    vocab = _init_vocab(special_tokens)

    # 特殊字节的bytes集合, 用来在后续处理中判断是否为特殊token
    special_token_bytes_set = {s.encode('utf-8') for s in special_tokens}

    # ---------- 1) mmap + 多进程并行统计seq_counter ----------
    # 不把整个文件读到内存，而是迭代chunk
    # 每个 chunk 发给 worker 得到局部 Counter, 然后主进程累加

    seq_counter = Counter()  # 统计所有seq_byte出现的频率

    # 预估 chunk 数用于进度条显示
    file_size = os.path.getsize(input_path)  # 文件大小（字节）
    est_total = max(1, math.ceil(file_size / chunk_bytes))

    # 显式选择启动子进程的方式，兼容Windows
    ctx = multiprocessing.get_context("spawn")  # 以"spawn"方式启动子进程

    with ctx.Pool(
        processes=num_processes,
        initializer=_init_worker,
        initargs=(special_tokens,),
    ) as pool:
        chunk_iter = _iter_text_chunks_mmap(input_path, chunk_bytes=chunk_bytes)

        # imap 边产生边消费
        for local_counter in tqdm(
            pool.imap(_count_chunk_worker, chunk_iter, chunksize=pool_chunksize),
            total=est_total,
            desc="Counting (pre-tokenize + freq)",
            unit="chunk",
            mininterval=0.5,
        ):
            seq_counter.update(local_counter)

    seqs = list(seq_counter.keys())  # seqs是所有不同的byte序列, 例如: [(b'h', b'e', b'l', b'l', b'o'), (b'w', b'o', b'r', b'l', b'd')]
    freqs = [seq_counter[s] for s in seqs]  # freqs是对应的频次列表

    # --- 初始化pair_cnt ---
    pair_cnt = Counter() # 统计所有byte对出现的频次
    pair_to_words = defaultdict(set)  # 记录每个byte对出现在哪些byte序列中

    for wi, seq in enumerate(seqs):
        for p in _pairs_in_seq(seq, special_token_bytes_set):
            pair_cnt[p] += freqs[wi]
            pair_to_words[p].add(wi)

    # --- 迭代合并 BPE训练 ---
    merges: List[Tuple[bytes, bytes]] = []

    def pick_best_pair() -> Tuple[bytes, bytes]: 
        # 先按照频次最大的选，再按照字典序最大的选
        return max(pair_cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
    
    target_merges = vocab_size - len(vocab)
    pbar = tqdm(total=max(0, target_merges), desc="Training BPE merges", unit="merge", mininterval=0.5)

    while len(vocab) < vocab_size and pair_cnt:
        best = pick_best_pair()
        a, b = best
        merged = a + b

        merges.append(best)
        vocab[len(vocab)] = merged

        affected_word_indices = list(pair_to_words.get(best, ()))  # 获取所有包含该pair的word索引

        if not affected_word_indices:
            pair_cnt.pop(best, None)
            pair_to_words.pop(best, None)
            pbar.update(1)
            continue

        # 更新所有受影响的word
        for wi in affected_word_indices:
            old_word = seqs[wi]
            word_freq = freqs[wi]

            # 移除旧pair计数
            for p in _pairs_in_seq(old_word, special_token_bytes_set):
                pair_cnt[p] -= word_freq
                if pair_cnt[p] <= 0:
                    pair_cnt.pop(p, None)
                pair_to_words[p].discard(wi)
            
            # merge
            new = _merge_seq(old_word, best, merged)
            seqs[wi] = new

            # 添加新pair计数
            for p in _pairs_in_seq(new, special_token_bytes_set):
                pair_cnt[p] += word_freq
                pair_to_words[p].add(wi)
        
        # 移除已合并的pair
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

    train_path = "../../../../data/owt_train.txt/owt_train.txt"
    vocab_size = 32000
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

    vocab_path = os.path.join(output_dir, "vocab_on_owt_gpt2.json")
    merges_path = os.path.join(output_dir, "merges_on_owt_gpt2.txt")
    save_vocab_and_merges(vocab, merges, vocab_path, merges_path)
    print(f"✅ 结果已保存到目录: {output_dir}")



