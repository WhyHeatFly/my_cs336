import os
from typing import List, Tuple, Dict, Set
import regex as re


def gpt2_bytes_to_unicode():
    """
    将字节转化为Unicode字符, 调用函数直接返回字典{字节:unicode字符}
    核心作用: 把所有 0~255 的字节映射到不会被 tokenizer 分词器“拆开”的 Unicode 字符，保证 BPE 分词对所有字节稳定可逆。
    因为所有的字符都要用0~255表示
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("i"), ord("¬") + 1 ))
        + list(range(ord("®"), ord("ÿ") + 1))
    )  # 安全可见的unicode字符

    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs  # 接收任意数量的“关键字参数”, 并把它打包成一个字典传入函数中
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    input_path: str Path to a text file with BPE tokenizer training data.
    vocab_size: int A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, 
    vocabulary items produced from merging, and any special tokens).
    special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.

    Your BPE training function should return the resulting vocabulary and merges:
    vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
    merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), 
    representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
    """

    # 将字节转化为Unicode字符，调用函数直接返回字典{字节:unicode字符}  
    _BYTES_TO_UNICODE_MAP = gpt2_bytes_to_unicode()
    # print(_BYTES_TO_UNICODE_MAP)
    # 将unicode字符转换为字节，{unicode:字节}
    token_str_to_bytes = {v : bytes([k]) for k, v in _BYTES_TO_UNICODE_MAP.items()}
    
    # 先校验参数，更好地增强函数的鲁棒性
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size 必须是一个正整数")

    # step1: 初始化词表，基础词汇表包含所有的256个基础bytes，对应ASCII码范围是0~255
    vocab: Dict[int, bytes] = {i : bytes([i]) for i in range(256)}
    current_next_id: int = 256
   
    # 用集合来高效检查特殊字符是否存在词汇表中
    existing_byte_values: Set[bytes] = set(vocab.values())
    
    # step2： 添加特殊符号到词汇表
    for sp_token in special_tokens:
        if len(vocab) >= vocab_size:  # 如果词汇表满了 就不再添加
            break 
        sp_token_bytes = sp_token.encode("utf-8")  # 将特殊符号字符串转为字节串
        if sp_token_bytes not in existing_byte_values:
            vocab[current_next_id] = sp_token_bytes  # 将新的字节串添加到词汇表
            existing_byte_values.add(sp_token_bytes)
            current_next_id += 1  # 更新下一个token ID
    
    # step3: pre_tokenization  预分词 本项目来源于和鲸社区，使用转载需要标注来源 若不加处理地跨文本合并字节，可能导致仅因标点不同而语义相近的词被拆分为完全不同的一组标记，例如“dog!”和“dog.”会被视为完全无关的标记，不利于模型学习其语义一致性
    # 记载训练的语料库
    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        text=""  # 文本不存在， 则视为空
    
    # 对语料库的文段进行简单的预分词，按要求保留空格分隔文本
    raw_words: list[str] = re.findall(r'\s*\S+', text)
    # 把"单词"转化为
    # step4: comput BPE merge 计算BPEmerge

run_train_bpe("1", 1, ['<|endoftext|>'])