import os
from typing import BinaryIO

# 将一个大文件切割成多个块，且每个块的边界都在特殊标记special_token处
def find_chunk_boundaries(
    file: BinaryIO,  # 以二进制方式打开的文件对象
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:  # 返回文件中每个块的起始位置列表
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # 确保特殊标记是字节
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring" 

    # Get total file size in bytes
    # seek(offset, whence): offset偏移量, whence: 偏移的基准点 os.SEEK_SET文件开头, os.SEEK_END文件末尾
    file.seek(0, os.SEEK_END)  # 将文件指针移动到文件末尾
    file_size = file.tell()  # 获取当前文件指针位置，即文件大小
    file.seek(0) # 将文件指针重新移动到文件开头

    chunk_size = file_size // desired_num_chunks  # 每个块的均匀理想大小

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)] # 初始猜测的块边界位置列表
    chunk_boundaries[-1] = file_size  # 确保最后一个边界在文件末尾

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1): # 对每个块边界进行调整，第一个边界一定是文件开头0，最后一个边界一定是文件末尾file_size
        initial_position = chunk_boundaries[bi] # 获取当前块边界的初始位置
        file.seek(initial_position)  # Start at boundary guess 
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk 

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"": # 如果读到文件末尾
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
with open(..., "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
