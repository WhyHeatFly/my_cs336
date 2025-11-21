import os
import time
import random
import mmap
import heapq
import regex as re
from pathlib import Path
from typing import List, Dict, Tuple, DefaultDict
from collections import defaultdict
import multiprocessing
from tqdm import tqdm

# GPT-2 预分词正则
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

global_worker_byte_map = None

def init_worker(byte_map: Dict[int, bytes]):
    global global_worker_byte_map
    global_worker_byte_map = byte_map

# -------------------- 预分词 --------------------
def pre_tokenize_document(doc: str, bytes_to_bytes_map: Dict[int, bytes], token_cache: Dict[str, List[bytes]]):
    tokens = re.findall(GPT2_SPLIT_PATTERN, doc, flags=re.UNICODE)
    sequences = []
    for token in tokens:
        if token not in token_cache:
            token_cache[token] = [bytes_to_bytes_map[b] for b in token.encode('utf-8')]
        sequences.append(token_cache[token])
    return sequences

def pre_tokenize_worker(doc: str):
    token_cache = {}
    return pre_tokenize_document(doc, global_worker_byte_map, token_cache)

def parallel_pre_tokenize(documents: List[str], num_processes: int, bytes_to_bytes_map: Dict[int, bytes]):
    if num_processes <= 1:
        token_cache = {}
        return [seq for doc in documents for seq in pre_tokenize_document(doc, bytes_to_bytes_map, token_cache)]
    with multiprocessing.Pool(num_processes, initializer=init_worker, initargs=(bytes_to_bytes_map,)) as pool:
        results = list(tqdm(pool.imap(pre_tokenize_worker, documents, chunksize=200), desc="预分词", mininterval=1))
    return [seq for doc_sequences in results for seq in doc_sequences]

# -------------------- bytes 映射 --------------------
def gpt2_bytes_to_bytes_local():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256+n)
            n+=1
    cs = [bytes(n) for n in cs]
    return dict(zip(range(256), cs))

# -------------------- 数据加载 --------------------
def load_and_sample_data(file_path: str, sample_size: int = 2000, special_token: str = "<|endoftext|>") -> str:
    try:
        with open(file_path, "r+", encoding='utf-8', errors='ignore') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                documents = []
                start = 0
                while start < len(mm):
                    end = mm.find(special_token.encode('utf-8'), start)
                    if end == -1:
                        doc = mm[start:].decode('utf-8', errors='replace').strip()
                        if doc: documents.append(doc)
                        break
                    doc = mm[start:end].decode('utf-8', errors='replace').strip()
                    if doc: documents.append(doc)
                    start = end + len(special_token)
                if len(documents) > sample_size:
                    documents = random.sample(documents, sample_size)
                return special_token.join(documents)
    except Exception as e:
        raise IOError(f"加载数据集失败: {e}")

# -------------------- BPE 索引 --------------------
class BPEIndex:
    def __init__(self, sequences: List[List[bytes]]):
        self.sequences = sequences
        self.pair_counts: DefaultDict[Tuple[bytes,bytes], int] = defaultdict(int)
        self.pair_positions: DefaultDict[Tuple[bytes,bytes], List[Tuple[int,int]]] = defaultdict(list)
        self.heap = []
        self.heap_entries: Dict[Tuple[bytes,bytes], Tuple[int,int,Tuple[bytes,bytes]]] = {}
        self.counter = 0  # 用于保证 heap 顺序

        for seq_idx, seq in enumerate(self.sequences):
            for pos in range(len(seq)-1):
                pair = (seq[pos], seq[pos+1])
                self.pair_counts[pair] += 1
                self.pair_positions[pair].append((seq_idx,pos))

        for pair,count in self.pair_counts.items():
            if count>1:
                entry = (-count, self.counter, pair)
                heapq.heappush(self.heap, entry)
                self.heap_entries[pair] = entry
                self.counter+=1

    def get_most_frequent(self):
        while self.heap:
            neg_count, _, pair = self.heap[0]
            if pair not in self.heap_entries or self.heap_entries[pair] is None:
                heapq.heappop(self.heap)
                continue
            current_count = self.pair_counts.get(pair,0)
            if -neg_count==current_count and current_count>1:
                return pair
            heapq.heappop(self.heap)
            if pair in self.heap_entries: del self.heap_entries[pair]
        return None

    def merge_pair(self, pair: Tuple[bytes,bytes], new_token: bytes):
        if pair not in self.pair_positions or not self.pair_positions[pair]:
            return 0
        positions_by_seq = defaultdict(list)
        for seq_idx,pos in self.pair_positions[pair]:
            positions_by_seq[seq_idx].append(pos)
        merge_count = 0
        for seq_idx, positions in positions_by_seq.items():
            seq = self.sequences[seq_idx]
            positions.sort(reverse=True)
            last_merged = -2
            for pos in positions:
                if pos>=len(seq)-1 or pos<=last_merged: continue
                if seq[pos]!=pair[0] or seq[pos+1]!=pair[1]: continue
                seq[pos] = new_token
                del seq[pos+1]
                merge_count+=1
                last_merged=pos
                if pos>0:
                    self._update_pair_count((seq[pos-1],pair[0]),-1)
                    self._update_pair_count((seq[pos-1],new_token),1)
                    self._add_position((seq[pos-1],new_token), seq_idx, pos-1)
                if pos<len(seq)-1:
                    self._update_pair_count((pair[1],seq[pos+1]),-1)
                    self._update_pair_count((new_token,seq[pos+1]),1)
                    self._add_position((new_token,seq[pos+1]), seq_idx, pos)
        if pair in self.pair_counts: del self.pair_counts[pair]
        if pair in self.pair_positions: del self.pair_positions[pair]
        if pair in self.heap_entries: self.heap_entries[pair]=None
        return merge_count

    def _update_pair_count(self, pair: Tuple[bytes,bytes], delta: int):
        if delta==0: return
        if pair not in self.pair_counts: self.pair_counts[pair]=0
        self.pair_counts[pair]=max(0,self.pair_counts[pair]+delta)
        if pair in self.heap_entries and self.heap_entries[pair] is not None:
            count,_ ,_ = self.heap_entries[pair]
            self.heap_entries[pair] = (-self.pair_counts[pair], _, pair)
        elif self.pair_counts[pair]>1:
            entry = (-self.pair_counts[pair], self.counter, pair)
            heapq.heappush(self.heap, entry)
            self.heap_entries[pair] = entry
            self.counter+=1

    def _add_position(self, pair: Tuple[bytes,bytes], seq_idx: int, pos: int):
        self.pair_positions[pair].append((seq_idx,pos))

# -------------------- BPE 训练 --------------------
def run_my_train_bpe(input_path: str, vocab_size:int, special_tokens:List[str], num_processes:int=8, sample_size:int=2000):
    bytes_to_bytes_map = gpt2_bytes_to_bytes_local()

    vocab = {i:bytes([i]) for i in range(256)}
    next_token_id=256
    existing_bytes = set(vocab.values())
    for sp in special_tokens:
        sp_bytes = sp.encode("utf-8")
        if sp_bytes not in existing_bytes:
            vocab[next_token_id]=sp_bytes
            existing_bytes.add(sp_bytes)
            next_token_id+=1
        if next_token_id>=vocab_size: break

    text = load_and_sample_data(input_path, sample_size, special_tokens[0])
    escaped_tokens = [re.escape(t) for t in special_tokens]
    split_pattern = "|".join(escaped_tokens)
    documents = [p for p in re.split(split_pattern, text) if p]

    sequences = parallel_pre_tokenize(documents, num_processes, bytes_to_bytes_map)
    bpe_index = BPEIndex(sequences)
    merges=[]
    total_merge=vocab_size-next_token_id
    process_bar = tqdm(total=total_merge, desc="训练 BPE", unit="merge")
    while next_token_id<vocab_size:
        best_pair = bpe_index.get_most_frequent()
        if best_pair is None: break
        new_token_bytes = best_pair[0]+best_pair[1]
        merge_count = bpe_index.merge_pair(best_pair, new_token_bytes)
        if merge_count==0: continue
        if new_token_bytes not in existing_bytes:
            vocab[next_token_id]=new_token_bytes
            existing_bytes.add(new_token_bytes)
            merges.append(best_pair)
            next_token_id+=1
            process_bar.update(1)
    process_bar.close()
    return vocab, merges

# =========================
# ⚡ 示例
# =========================
if __name__=="__main__":
    train_path = r"..\..\data\TinyStoriesV2-GPT4-train.txt"
    vocab, merges = run_my_train_bpe(train_path, 1000, ["<|endoftext|>","<|pad|>","<|unk|>"])
    print(f"词汇表大小: {len(vocab)}, merge数: {len(merges)}")
