from resiliparse.extract.html2text import extract_plain_text
import resiliparse.parse

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