import re
def convert_cot_to_think_answer(text: str):
    m = re.search(r"####\s*([^\n]+)\s*$", text)
    if m:
        ans = m.group(1).strip()
        prefix = text[: m.start()].rstrip()
        return f"{prefix} </think> <answer>{ans}</answer>"

    # Fallback: try to capture a trailing number at end of text
    m_num = re.search(r"(-?\d+(?:\.\d+)?)\s*$", text)
    if m_num:
        ans = m_num.group(1)
        prefix = text[: m_num.start()].rstrip()
        return f"{prefix} </think> <answer>{ans}</answer>"

    return text

import pandas as pd

def extract_math_answer(answer: str) -> str:
    ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"

data_path = r"C:\Users\why\WayHeatFly\cs336\hw5\assignment5-alignment\data\SFT\gsm8k\main\test-00000-of-00001.parquet"
df = pd.read_parquet(data_path)
# print(df.head())
data_len = len(df)

for question, text in zip(df['question'][:10], df['answer'][:10]):
    print(convert_cot_to_think_answer(text))
    
    print(extract_math_answer(text))
    print("="*30)
