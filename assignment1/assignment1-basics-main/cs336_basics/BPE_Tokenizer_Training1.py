import os
from pathlib import Path  # 用来处理文件路径的库
import time
from typing import Union, List, Tuple, Dict, DefaultDict, Any
import mmap  # 用于内存映射文件
import random
import regex as re
from tqdm import tqdm
import multiprocessing  # 用于多进程处理
from collections import defaultdict


# gpt2预分词的正则表达式
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

"""
    多进程工作:
    global_worker_byte_map = None # 全局变量用于多进程, 用于在子进程中共享字节映射表, 
    但是python的多进程不能共享普通变量,因此这里是的策略是：在主进程启动子进程的时候，调用 initializer 初始化子进程中的全局变量
"""
global_worker_byte_map = None # 全局变量用于多进程, 用于在子进程中共享字节映射表, 但是python的多进程不能共享普通变量

"""
每个子进程启动时执行：
函数 init_worker
将主进程传入的byte_map(即bytes_to_unicode_map)
存入该子进程自己的全局变量 global_worker_byte_map
"""
# 初始化worker进程
def init_worker(byte_map: Dict[int, str]):
    global global_worker_byte_map
    global_worker_byte_map = byte_map

# worker 执行的预分词任务
def pre_tokenize_worker(doc: str) -> List[List[str]]:
    return pre_tokenize_document(doc, global_worker_byte_map)

def pre_tokenize_document(doc: str, bytes_to_unicode_map: Dict[int, str]) -> List[List[str]]:
    """
    预分词处理单个文件,把一个文档(字符串)拆成GPT-2预分词序列, 每个词最终会转化成BPE训练所需的"unicode字符序列"
    tokens = re.findall(GPT2_SPLIT_PATTERN, doc, flags=re.UNICODE) 按照gpt2的分词模式进行分词
    token_unicode = ''.join(bytes_to_unicode_map[b] for b in token.encode('utf-8')) 每个token先转化为字节序列(b), 再转化为unicode字符
    """
    tokens = re.findall(GPT2_SPLIT_PATTERN, doc, flags=re.UNICODE)
    sequences = []
    for token in tokens:
        token_unicode = ''.join(bytes_to_unicode_map[b] for b in token.encode('utf-8'))
        sequences.append(list(token_unicode))
    return sequences

def parallel_pre_tokenize(
        documents: List[str],
        num_processes: int,
        bytes_to_unicode_map: Dict[int, str],
) -> List[List[str]]:
    """并行预分词优化"""
    if num_processes <= 1:
        return [seq for doc in documents for seq in pre_tokenize_document(doc, bytes_to_unicode_map)]
    
    with multiprocessing.Pool(
        num_processes,  # 多进程的进程数
        initializer=init_worker,
        initargs=(bytes_to_unicode_map,)  # init_worker的参数列表
    ) as pool:
        results = list(tqdm(
            pool.imap(pre_tokenize_worker, documents, chunksize=50),  # pool.imp() 并行处理documents里的str，对documents里的每个doc并行执行pre_tokenize_worker(doc)
            desc="预分词",  # tqdm的参数 进度条左侧显示的描述文字
            mininterval=1  # 进度条最少每秒刷新一次
        ))
    
    return [seq for doc_sequences in results for seq in doc_sequences]

def load_and_sample_data(
        file_path: str,
        sample_size: int = 2000,
        special_token: str = "<|endoftext|>",  # 用special_token 作为分隔符切隔文档
) -> str:
    """
    内存映射文件方式加载并采样文档, 把磁盘上的文件直接映射到内存地址空间，这样读取文件内容就像访问内存一样快，不需要传统的 read() 拷贝。
    mmap.mmap()创建一个内存映射对象，把文件的内容直接映射到内存，让你像操作数组一样访问文件，而不用 read() 去拷贝数据。
    f.fileno() 返回文件在操作系统中的文件描述符, mmap需要这个整数来知道要映射哪个文件
    0 表示映射整个文件, 如果只想映射前1MB, 可以写1024 * 1024
    access mmap的访问模式 mmap.ACCESS_READ 只读
    with ... as mm: python中的上下文管理器 在用完后会自动关闭 减少内存泄漏风险
    """
    try:
        with open(file_path, "r+", encoding='utf-8', errors='ignore') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                documents = []
                start = 0
                while start < len(mm):
                    end = mm.find(special_token.encode('utf-8'), start)
                    if end == -1:
                        doc = mm[start:].decode('utf-8', errors='repalce').strip()
                        if doc:
                            documents.append(doc)
                        break

                    doc = mm[start:end].decode('utf-8', errors='replace').strip()
                    if doc:
                        documents.append(doc)
                    start = end + len(special_token)
                
                if len(documents) > sample_size:
                    documents = random.sample(documents, sample_size)
                
                return special_token.join(documents)
    except Exception as e:
        raise IOError(f"加载数据集失败： {e}")

def gpt2_bytes_to_unicode_local():
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

"""
    defaultdict(int) 数据结构, 区别于普通dict, 访问不存在的键时自动创建默认值,以int作为工厂函数默认值为0, 以list作为工厂函数, defaultdict 会为每个缺失的键创建一个空列表
    DefualtDict: 是类型注解(typing) 只是用来告诉 IDE / 静态检查工具： “这是一个 key=tuple, value=int 的 defaultdict,
    变量和注解类型不一致不会报错
    Any: 任何类型都可以
"""
class BPEIndex:
    """高效索引结构用于BPE合并"""
    def __init__(self, sequences: List[List[str]]):
        self.sequences = sequences  # 存储所有的文本序列
        self.pair_counts: DefaultDict[Tuple[str, str], int] = defaultdict(int)  # 统计字节对频率
        self.pair_positions: DefaultDict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)  # 记录字节对出现的位置
        self.heap = []  # 最大堆(存最高频率字节对)
        self.heap_entrie: Dict[Tuple[str, str], Any] = {}  # 堆条目快速访问
        
        

def run_train_bpe(
        input_path: Union[str, os.PathLike],
        vocab_size: int,
        special_tokens: List[str] = ["<|endoftext|>"],
        num_processes: int = 8,
        sample_size: int = 2000,
        **kwargs,
) -> Tuple[dict[int, bytes], List[Tuple[bytes, bytes]]]:
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
     # 参数验证
     base_vocab_size = 256 + len(special_tokens)
     if vocab_size < base_vocab_size:
         raise ValueError(f"vocab_size至少需{base_vocab_size}")
     
     # 1. 字节到unicode的映射
     bytes_to_unicode_map = gpt2_bytes_to_unicode_local()
     unicode_to_bytes_map = {v : bytes([k]) for k, v in bytes_to_unicode_map.items()}
     
     # 2. 初始化词汇表
     vocab = {i : bytes([i]) for i in range(256)}
     next_token_id = 256
     existing_bytes = set(vocab.values())

     # 3. 添加特殊token
     for sp_token in special_tokens:
         if len(vocab) > vocab_size:
             break
         
         sp_token_bytes = sp_token.encode("utf-8")
         if sp_token_bytes not in existing_bytes and len(vocab):
             vocab[next_token_id] = sp_token_bytes
             existing_bytes.add(sp_token_bytes)
             next_token_id += 1
     
     # 4. 加载并采样文档
     print(f"📖 从{input_path} 加载并采样 {sample_size} 个文档...")
     text = load_and_sample_data(input_path, sample_size, special_tokens[0])

     # 5. 划分文档
     """
     re.escape() 的作用是 把字符串中的特殊字符全部转义, 例如 <|endoftext|> 中的 | 会在正则中被解释成“或”，所以必须转义。
     "|".join(escaped_tokens) 构建正则split模型 这样正则匹配就可以匹配任意一个special_token
     """
     escaped_tokens = [re.escape(sp_token) for sp_token in special_tokens]  # 返回 "<\\|endoftext\\|>"
     split_pattern = "|".join(escaped_tokens)
     # 按特殊token切分文档 切分之后 特殊token不会在文档本身
     documents = [part for part in re.split(split_pattern, text) if part]

     # 6. 并行预分词
     sequences = parallel_pre_tokenize(documents, num_processes, bytes_to_unicode_map)
     
     print(f"✅ 预分词完成，得到{len(sequences):} 个token序列")
     
     # 7. 初始化索引结构
     print("🦴 构建BPE索引...")
     bpe_index = BPEIndex(sequences)
     merges = []
     vocab_process = len(vocab)
     total_merge = vocab_size - vocab_process

     return 1, 2

if __name__ == "__main__":
    # 配置参数
    config = {
        "vocab_size": 10000,
        "special_tokens": ["<|endoftext|>", "<|pad|>", "<|unk|>"],
        "num_processes": 8,  # 多进程的进程数
        "sample_size": 2000,  # 采样测试的数量
    }

    # 训练集的路径
    train_path = "C:/Users/why/Desktop/cs336/assignment1/data/TinyStoriesV2-GPT4-train.txt"
    # 验证集的路径
    valid_path = "C:/Users/why/Desktop/cs336/assignment1/data/TinyStoriesV2-GPT4-valid.txt"
    
    """
    from pathlib import Path
    Path(train_path) 会产生一个Path对象, 表示某个文件路径
    .exists() 用来检查文件路径是否存在, 返回True/False
    raise 用来手动抛出异常
    FileNotFoundError是python内置错误类型, 表示文件未找到
    """
    # 检验路径是否合法
    if not Path(train_path).exists():
        raise FileNotFoundError(f"训练集文件 {train_path} 不存在")
    if not Path(valid_path).exists():
        raise FileNotFoundError(f"验证集文件 {valid_path} 不存在")
    
    # 训练分词器
    print("🚀 开始训练")
    start_time = time.time()

    train_vocab, train_megers = run_train_bpe(train_path, **config)