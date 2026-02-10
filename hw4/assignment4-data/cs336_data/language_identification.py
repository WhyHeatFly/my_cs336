import fasttext
from typing import Any
from pathlib import Path
from itertools import islice

_MODEL_PATH = Path(__file__).parent / "lid.176.bin"

warc_path = Path(__file__).parent / "warc_extracted_text.txt"

def identify_language(text: str) -> tuple[Any, float]:
    classifier = fasttext.load_model(str(_MODEL_PATH))
    cleaned_text = text.replace('\n', '').replace('\r', '')
    # 返回 ['__label__en'] [0.98]
    label, confidence = classifier.predict(cleaned_text, k=1)

    if label:
        res_label = label[0].replace('__label__', '')

    res_conf = confidence[0]
    return (res_label, res_conf)

def main():
    with open(warc_path, 'r', encoding='utf-8') as f:
        for line in islice(f, 20):
            res = identify_language(line)
            print(res)
        
if __name__ == "__main__":
    main()
