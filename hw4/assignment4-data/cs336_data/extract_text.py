from resiliparse.extract.html2text import extract_plain_text
import resiliparse.parse

"""ArchiveIterator: 用来 迭代整个 WARC 文件中的记录。"""
"""WarcRecordType: 定义了 WARC 文件中记录的类型，可以选择只处理某些类型的记录"""
from fastwarc.warc import ArchiveIterator, WarcRecordType
from pathlib import Path

# WARC 文件路径
warc_path = Path("data/CC/example.warc.gz/CC-MAIN-20250417135010-20250417165010-00065.warc.gz")

# 对应的 WET 文件路径
wet_path = Path("data/CC/example.warc.wet.gz/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz")

output_path = Path("warc_extracted_text.txt")

""" 用于从** 包含原始 HTML 的字节串 (byte string) **中提取文本"""
def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        decoded_text = html_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            # 检测文本的编码方式
            coding_type = resiliparse.parse.encoding.detect_encoding(html_bytes)
            decoded_text = html_bytes.decode(coding_type)
        except (UnicodeDecodeError, LookupError, TypeError):
            # 编码检测失败，尝试用忽略错误的方式编码
            try:
                decoded_text = html_bytes.decode('utf-8', errors='ignore')
            except:
                return None
    
    try:
        """extract_plain_text: 把一段 HTML 文档中的“可见文本”提取出来，尽量去掉标签、脚本和部分模板内容，输出纯文本。"""
        clean_text = extract_plain_text(decoded_text)
        return clean_text
    except:
        return None
    
def main():
    # 提取 WARC 中的 HTML 文本
    warc_text = []
    with warc_path.open("rb") as f:
        for record in ArchiveIterator(f):
            if record.record_type != WarcRecordType.response:
                continue
            
            if not record.http_headers:
                continue  # 有些 response 可能没有 HTTP 内容

            html_bytes = record.reader.read()
            text = extract_text_from_html_bytes(html_bytes)
            
            if text.strip():
                warc_text.append(text.strip())

    
    # 提取 WET 文件的文本
    wet_text = []
    import gzip
    with gzip.open(wet_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            wet_text.append(line.strip())
    
    # 写入磁盘（一篇文档一行）
    with output_path.open("w", encoding="utf-8") as out:
        for doc in warc_text:
            out.write(doc.replace("\n", " ") + "\n")

    print(f"Saved {len(warc_text)} documents to {output_path}")
    
    # 取前几条看看差异
    print("=== WARC 提取 ===")
    for t in warc_text[:3]:
        print(t[:500], "\n---")

    print("=== WET 文件 ===")
    for t in wet_text[:3]:
        print(t[:500], "\n---")

if __name__ == "__main__":
    main()