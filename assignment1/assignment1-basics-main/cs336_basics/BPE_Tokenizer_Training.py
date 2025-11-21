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
import heapq  # 用于实现堆的数据结构


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
            pool.imap(pre_tokenize_worker, documents, chunksize=500),  # pool.imp() 并行处理documents里的str，对documents里的每个doc并行执行pre_tokenize_worker(doc)
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
                        doc = mm[start:].decode('utf-8', errors='replace').strip()
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
        + list(range(ord("¡"), ord("¬") + 1 ))
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
        self.pair_positions: DefaultDict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)  # 记录字节对出现的位置, (seq_idx, pos)
        self.heap = []  # 最大堆(存最高频率字节对)
        self.heap_entries: Dict[Tuple[str, str], Any] = {}  # 堆条目快速访问

        # 初始化索引 一次性统计所有相邻字节对得出现位置和频率——将时间复杂度由O(N^2) 降到O(NlogN)
        for seq_idx, sequence in enumerate(self.sequences):
            for pos in range(len(sequence) - 1):
                pair = (sequence[pos], sequence[pos + 1])
                self.pair_counts[pair] += 1
                self.pair_positions[pair].append((seq_idx, pos))

        # 构建堆, 将高频字节对(>1次)加入最大堆，让get_most_frequent_pair()能O(1)时间复杂度获得最高频对
        for pair, count in self.pair_counts.items():
            if count > 1:
                entry = [-count, pair]  # python的heapq是最小堆，所以用负数存储频率
                heapq.heappush(self.heap, entry)
                self.heap_entries[pair] = entry  # 例如: {('a', 'b'): [-5, ('a', 'b')]}

    def get_most_frequent(self) -> Tuple[str, str]:
        """快速返回当前最高频率字节对"""
        while self.heap:
            neg_count, pair = self.heap[0]  # 查看堆顶元素
            # 检查pair是否仍然有效 Lazy deletion（惰性删除）：不直接修改堆，而是标记旧 entry 为无效，遇到时 pop 掉
            if pair not in self.heap_entries:
                heapq.heappop(self.heap)  # 无效则弹出堆顶
                continue

            current_count = self.pair_counts.get(pair, 0)

            # 检查计数是否匹配且大于1
            if -neg_count == current_count and current_count > 1:
                return pair
            # 否则移除无效条目
            heapq.heappop(self.heap)
            if pair in self.heap_entries:  # 确保条目存在
                del self.heap_entries[pair]
        
        return None  # 堆为空时返回None
    
    def merge_pair(self, pair: Tuple[str, str], new_token: str) -> int:
        """合并字符对并更新索引"""
        # Lazy deletion（惰性删除）：可能pair已经不在序列中或者pair_positions已经为空
        if pair not in self.pair_positions or not self.pair_positions[pair]:
            return 0
        
        # 按序列和位置分组
        positions_by_seq = defaultdict(list)
        for seq_idx, pos in self.pair_positions[pair]:
            positions_by_seq[seq_idx].append(pos)
        
        merge_count = 0
        for seq_idx, positions in positions_by_seq.items():
            seq = self.sequences[seq_idx]
            # 按位置倒序排序
            positions.sort(reverse=True)
            last_merged_pos = -2

            for pos in positions:
                # 检查是否已经被前面的合并影响
                if pos >= len(seq) - 1 or pos <= last_merged_pos:
                    continue
                if seq[pos] != pair[0] or seq[pos + 1] != pair[1]:
                    continue
                
                # 执行合并
                seq[pos] = new_token
                del seq[pos + 1]
                merge_count += 1
                last_merged_pos = pos

                # 更新左侧pair
                if pos > 0:
                    left_pair = (seq[pos - 1], pair[0])
                    self._update_pair_count(left_pair, -1)

                    new_left_pair = (seq[pos - 1], new_token)
                    self._update_pair_count(new_left_pair, 1)
                    self._add_position(new_left_pair, seq_idx, pos - 1)
                
                # 更新右侧pair
                if pos < len(seq) - 1:
                    right_pair = (pair[1], seq[pos + 1])
                    self._update_pair_count(right_pair, -1)

                    new_right_pair = (new_token, seq[pos + 1])
                    self._update_pair_count(new_right_pair, 1)
                    self._add_position(new_right_pair, seq_idx, pos)

        # 清理已合并的pair
        if pair in self.pair_counts:
            del self.pair_counts[pair]
        if pair in self.pair_positions:
            del self.pair_positions[pair]
        if pair in self.heap_entries:
            # 标记为无效, 稍后清理
            self.heap_entries[pair] = None

        return merge_count
    
    def _update_pair_count(self, pair: Tuple[str, str], delta: int):
        """更新字符对计数"""
        if delta == 0:
            return 
        
        # 确保pair在字典中
        if pair not in self.pair_counts:
            self.pair_counts[pair] = 0
        
        new_count = self.pair_counts[pair] + delta
        self.pair_counts[pair] = new_count

        # 确保计数不为负
        if new_count < 0:
            new_count = 0
            self.pair_counts[pair] = 0
        
        if pair in self.heap_entries and self.heap_entries[pair] is not None:
            # 更新堆条目
            self.heap_entries[pair][0] = -new_count
            heapq.heapify(self.heap)
        elif new_count > 1:  # 只添加计数大于1的pair
            # 新建堆条目
            entry = [-new_count, pair]
            heapq.heappush(self.heap, entry)
            self.heap_entries[pair] = entry

    def _add_position(self, pair: Tuple[str, str], seq_idx: int, pos: int):
        """添加新位置到索引"""
        self.pair_positions[pair].append((seq_idx, pos))

def run_my_train_bpe(
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
         
         if sp_token_bytes not in existing_bytes and len(vocab) < vocab_size:
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

     # 8. BPE训练主循环
     print(f"🚀 开始BPE训练, 目标合并数: {total_merge}")
     process_bar = tqdm(total=total_merge, desc="训练BPE", unit="合并", mininterval=0.5)
     
     while vocab_process < vocab_size:
         best_pair = bpe_index.get_most_frequent()
         if best_pair is None:
             print("\n ⚠️ 没有更多有效的字符对可供合并, 提前结束训练")
             break
         
         # 创建新token
         new_token_str = best_pair[0] + best_pair[1]
         p1_bytes = unicode_to_bytes_map[best_pair[0]]
         p2_bytes = unicode_to_bytes_map[best_pair[1]]
         new_token_bytes = p1_bytes + p2_bytes

         # 执行合并
         merge_count = bpe_index.merge_pair(best_pair, new_token_str)
         if merge_count == 0:
             continue
         
         # 更新词汇表
         if new_token_bytes not in existing_bytes:
             vocab[next_token_id] = new_token_bytes
             existing_bytes.add(new_token_bytes)
             merges.append((p1_bytes, p2_bytes))
             next_token_id += 1
             vocab_process += 1
             process_bar.update(1)
        
         # 更新映射表
         unicode_to_bytes_map[new_token_str] = new_token_bytes
     
     process_bar.close()

     return vocab, merges

def evaluate_tokenizer(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], test_text: str):
    """简单评估分词器效果"""
    print("\n🔍 分词器评估")
    sample_text = test_text[:200] + "..." if len(test_text) > 200 else test_text
    print(f"样例文本: {sample_text}")

    # 简单统计
    unique_tokens = set(vocab.values())
    print(f"词汇表大小: {len(vocab):,}")
    print(f"唯一token数: {len(unique_tokens):,}")
    print(f"合并操作数: {len(merges):,}")

if __name__ == "__main__":
    # 配置参数
    config = {
        "vocab_size": 500,
        "special_tokens": ["<|endoftext|>", "<|pad|>", "<|unk|>"],
        "num_processes": 8,  # 多进程的进程数
        "sample_size": 2000,  # 采样测试的数量
    }

    # 训练集的路径
    train_path = r"..\..\data\TinyStoriesV2-GPT4-train.txt"
    # 验证集的路径
    valid_path = r"..\..\data\TinyStoriesV2-GPT4-valid.txt"
    
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

    train_vocab, train_merges = run_my_train_bpe(train_path, **config)
    print(f"\n✅ 训练完成! 耗时: {time.time() - start_time:.2f}秒")

    # 小规模验证（使用验证集的10%）
    print("\n💡 小规模验证")
    valid_config = config.copy()
    valid_config['sample_size'] = int(20) 

    valid_vocab, valid_merges = run_my_train_bpe(valid_path, **valid_config)
    # print(valid_vocab)
    # print(valid_merges)
    # 分析结果
    print("\n 训练结果")
    print(f"训练词汇表大小: {len(train_vocab):,}")
    print(f"训练合并操作数：{len(train_merges):,}")
    print(f"验证词汇表大小: {len(valid_vocab):,}")
    print(f"验证合并操作数: {len(valid_merges):,}")

    # 比较词汇表重叠率
    train_tokens = set(train_vocab.values())
    valid_tokens = set(valid_vocab.values())
    overlap = train_tokens & valid_tokens
    print(f"\n 词汇表重叠率: {len(overlap)/len(train_tokens):.1%}")

    # 加载验证集样例进行评估
    with open(valid_path, 'r', encoding='utf-8') as f:
        valid_text = f.read(1000)  # 读取前1000字符用于评估
    
    evaluate_tokenizer(train_vocab, train_merges, valid_text)

    import json  # 保存成json格式

    def save_vocab_merges(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], vocab_path: str, merges_path: str):
        """保存词汇表和合并列表到文件"""
        # 1. 保存词汇表(JSON格式)
        vocab_str = {token.decode('utf-8', errors='replace') : idx for idx, token in vocab.items()}
        with open(vocab_path, 'w', encoding='utf-8') as f:
            # json.dump(obj, file) 将python对象以JSON格式写入文件， ensure_ascii=False 允许写出非ASCII字符, indent=2 格式化缩进两个空格
            json.dump(vocab_str, f, ensure_ascii=False, indent=2)

        # 2. 保存合并列表(文本格式)
        with open(merges_path, 'w', encoding='utf-8') as f:
            for merge in merges:
                part1 = merge[0].decode('utf-8', errors='replace')
                part2 = merge[1].decode('utf-8', errors='replace')
                f.write(f"{part1} {part2}\n")
    
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, "gpt2_vocab.json")
    merges_path = os.path.join(output_dir, "gpt2_merges.txt")

    save_vocab_merges(train_vocab, train_merges, vocab_path, merges_path)
    print(f"✅ 词汇表已保存至: {vocab_path}")
    print(f"✅ 合并列表已保存至: {merges_path}")

    # 内存分析
    import psutil  # 系统库, 用来获取CPU信息、内存使用情况、磁盘IO、进程状态、网络流量等系统级监控信息
    process = psutil.Process()  # 获取当前Python进程对象
    mem_usage = process.memory_info().rss / (1024 ** 3)  # GB memory_info()返回一个对象,包含当前进程的各种内存数据 rss 实际占用的物理内存
    print(f"峰值内存使用：{mem_usage:.2f} GB")
