from typing import List, Dict, Tuple, Iterable, Iterator, Set
import regex as re

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

class Tokenizer:

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] | None = None):
        """
        __init__ 的 Docstring: 初始化Tokenizer实例
        
        :param self: 说明
        :param vocab: int->bytes的词汇表映射
        :type vocab: Dict[int, bytes]
        :param merges: 字节对的合并规则列表
        :type merges: List[Tuple[bytes, bytes]]
        :param special_tokens: 特殊标记列表
        :type special_tokens: List[str] | None
        """
        self.vocab: Dict[int, bytes] = vocab  # id->bytes_combination
        self.merges: Dict[Tuple[bytes, bytes]] = merges # 合并规则
        self.special_tokens: List[str] = special_tokens if special_tokens is not None else []
        # we also need a dict to map bytes_combination->id for encoding
        self.bytes_2_id_vocab: Dict[bytes, int] = {v : k for k, v in vocab.items()}
        # merge -> rank dict for quick lookup
        self.merges_rank: Dict[Tuple[bytes, bytes], int] = {merge : idx for idx, merge in enumerate(self.merges)}

        if self.special_tokens:
            self.special_tokens_bytes: List[bytes] = [token.encode('utf-8') for token in self.special_tokens]
        else:
            self.special_tokens_bytes = []
        
        # add special tokens
        for sp_bytes in self.special_tokens_bytes:
            if sp_bytes not in self.bytes_2_id_vocab:
                new_id = len(self.vocab)
                self.vocab[new_id] = sp_bytes
                self.bytes_2_id_vocab[sp_bytes] = new_id

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] | None = None) -> "Tokenizer":
        """
        from_files: construct and return a Tokenizer from a serialized vocabulary and list of merges and a list of special tokens
        
        :param cls: 类方法的第一个参数，表示类本身，作用是用于创建类的实例
        :param vocab_filepath: 词汇表文件路径
        :type vocab_filepath: str
        :param merges_filepath: 合并规则文件路径
        :type merges_filepath: str
        :param special_tokens: 用户自定义的特殊标记列表
        :type special_tokens: List[str] | None
        """
        import json

        with open(vocab_filepath, 'r', encoding='utf-8', errors='replace') as vf:
            raw_vocab = json.load(vf)
        
        norm_vocab: Dict[int,bytes] = {}
        for k, v in raw_vocab.items():
            kid = int(k)
            norm_vocab[kid] = token_str_to_bytes(v)
        
        # load and normalize merges: ensure tuples of bytes
        norm_merges: List[Tuple[bytes, bytes]] = []
        with open(merges_filepath, 'r', encoding='utf-8', errors='replace') as mf:
            for line in mf:
                line = line.rstrip('\n')  # 去除行尾换行符
                if not line:
                    continue
                a_str, b_str = line.rsplit(" ", 1)
                norm_merges.append((token_str_to_bytes(a_str), token_str_to_bytes(b_str)))

        
        return cls(norm_vocab, norm_merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        """
        encode 的 Docstring: Encode an input text into a sequence of token IDs.
        
        :param self: 说明
        :param text: 说明
        :type text: str
        :return: 说明
        :rtype: List[int]
        """
        # step1. 预分词
        pre_tokenization_words: List[str] = self._pre_tokenization(text, self.special_tokens)

        # step2. 对每个预分词结果进行编码
        result_ids: List[int] = []
        for word in pre_tokenization_words:
            if word in self.special_tokens:
                result_ids.append(self.bytes_2_id_vocab[word.encode('utf-8')])
            else:
                result_ids.extend(self._encode_word(word))
        return result_ids
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        encode_iterable 的 Docstring: Given an iterable of strings(e.g.,a Python file handle),return a generator that lazily yields
        token IDs.This is required for memory-efficient tokenization of large files that we cannot directly load into memory
        
        :param self: 说明
        :param iterable: 说明
        :type iterable: Iterable[str]
        :return: 说明
        :rtype: Iterator[int]
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: List[int]) -> str:
        """
        decode 的 Docstring: Decode a sequence of token IDs  into text.
        
        :param self: 说明
        :param ids: 说明
        :type ids: List[int]
        :return: 说明
        :rtype: str
        """
        bytes_list = b''.join(self.vocab.get(id_) for id_ in ids)
        return bytes_list.decode('utf-8', errors='replace')
    
    def _encode_word(self, word: str) -> List[int]:
        """辅助函数，编码与分词后的单词
        _encode_word 的 Docstring: Encode a pretoken single word (normal text, not special tokens) into a sequence of token IDs using BPE merges.
        
        :param self: 说明
        :param word: 说明
        :type word: str
        :return: 说明
        :rtype: List[int]
        """
        def word_to_bytes(word: str) -> Tuple[bytes, ...]:
            b = word.encode('utf-8')
            return tuple(bytes([x]) for x in b)
        
        pre_token_word_bytes: Tuple[bytes, ...] = word_to_bytes(word)
        pre_token_word_bytes_after_merge: Tuple[bytes, ...] = self._apply_merge(pre_token_word_bytes)

        tokens_ids: List[int] = []
        for word_bytes in pre_token_word_bytes_after_merge:
            token_id = self.bytes_2_id_vocab[word_bytes]
            tokens_ids.append(token_id)
        
        return tokens_ids

    def _apply_merge(self, pre_token_bytes: Tuple[bytes, ...]) -> List[bytes]:
        # 应用BPE合并规则
        def get_pairs(word_bytes: List[bytes]) -> Set[Tuple[bytes, bytes]]:
            pairs = set()
            prev_byte = word_bytes[0]
            for byte in word_bytes[1:]:
                pairs.add((prev_byte, byte))
                prev_byte = byte
            return pairs
        
        word_bytes: List[bytes] = list(pre_token_bytes)
        word_bytes_pair: Set[Tuple[bytes, bytes]] = get_pairs(word_bytes)

        if not word_bytes_pair:
            return word_bytes
        
        while True:
            # find the minimum rank pair
            bigram: Tuple[bytes, bytes] = min(word_bytes_pair, key=lambda pair: self.merges_rank.get(pair, float('inf')))
            if bigram not in self.merges_rank:
                break
            idx = 0
            new_word_bytes: List[bytes] = []
            first, second = bigram
            while idx < len(word_bytes):
                try:
                    first_nearest: int = word_bytes.index(first, idx)
                except ValueError:
                    new_word_bytes.extend(word_bytes[idx:])
                    break
                else:
                    new_word_bytes.extend(word_bytes[idx:first_nearest])
                    idx = first_nearest
                    if word_bytes[idx] == first and idx + 1 < len(word_bytes) and word_bytes[idx + 1] == second:
                        new_word_bytes.append(first + second)
                        idx += 2
                    else:
                        new_word_bytes.append(word_bytes[idx])
                        idx += 1
            word_bytes = new_word_bytes
            if len(word_bytes) == 1:
                break
            else:
                word_bytes_pair = get_pairs(word_bytes)
        return word_bytes

    def _pre_tokenization(self, text: str, special_tokens: List[str]) -> List[str]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        if not special_tokens:
            return re.findall(PAT, text)
        
        tokens: List[str] = sorted(special_tokens, key=len, reverse=True)
        union: str = "|".join(re.escape(token) for token in tokens)
        parts: List[str] = re.split(f"({union})", text)
        out: List[str] = []
        for part in parts:
            if part in special_tokens:
                out.append(part)
            else:
                out.extend(re.findall(PAT, part))
        return out

if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        vocab_filepath=r"C:\Users\why\WayHeatFly\cs336\hw1\assignment1-basics\cs336_basics\chapter1\train_bpe\bpe_output\vocab_on_owt.json",
        merges_filepath=r"C:\Users\why\WayHeatFly\cs336\hw1\assignment1-basics\cs336_basics\chapter1\train_bpe\bpe_output\merges_on_owt.txt",
        special_tokens=["<|endoftext|>"]
    )
