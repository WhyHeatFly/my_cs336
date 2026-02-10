import regex as re
from pathlib import Path
import random


warc_path = Path(__file__).parent / "warc_extracted_text.txt"

def mask_emails(text: str) -> tuple[str, int]:
    EAMIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+"
    REPLACE_STRING = '|||EMAIL_ADDRESS|||'
    count = 0
    def replacer(match):
        nonlocal count
        count += 1
        return REPLACE_STRING
    # re.sub(pattern, repl, string): 在 string 中查找所有匹配 pattern 的子串，并用 repl 指定的方式进行替换。
    new_text = re.sub(EAMIL_PATTERN, replacer, text)
    return (new_text, count)

def mask_phone_numbers(text: str) -> tuple[str, int]:
    # 匹配美国手机号码的各种格式
    # 包括：0123456789,(012)-345-6789,(012) 345 6789, 012-345-6789
    PHONE_PATTERNS = [
        r'\b[2-9]\d{9}\b',  # 10位连续数字
        r'\([2-9]\d{2}\)-\d{3}-\d{4}',  # (xxx)-xxx-xxxx
        r'\([2-9]\d{2}\)\s\d{3}\s\d{4}',  # (xxx) xxx xxxx  
        r'[2-9]\d{2}-\d{3}-\d{4}'  # xxx-xxx-xxxx
    ]

    REPLACED_STR = '|||PHONE_NUMBER|||'
    result_text = text

    total_count = 0
    for pattern in PHONE_PATTERNS:
        # re.subn返回 (替换后的字符串, 替换次数)
        new_text, count = re.subn(pattern=pattern, repl=REPLACED_STR, string=result_text)
        result_text = new_text
        total_count += count
    
    return (result_text, total_count)

def mask_ips(text: str) -> tuple[str, int]:
    # IPv4地址匹配模式 (0-255.0-255.0-255.0-255)
    IPV4_PATTERN = r'(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\d)'
    REPLACED_STR = "|||IP_ADDRESS|||"
    new_text, cnt = re.subn(IPV4_PATTERN, REPLACED_STR, text)
    return (new_text, cnt)

def random_sample(path, k):
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < k:
                samples.append(line)
            else:
                j = random.randint(0, i)
                if j < k:
                    samples[j] = line

    return samples

def main():
    samples = random_sample(warc_path, 20)

    for i, original in enumerate(samples):
        print(original)
        print(mask_emails(original))
        print(mask_phone_numbers(original))
        print(mask_ips(original))
        print("===============\n")
        
if __name__ == "__main__":
    main()