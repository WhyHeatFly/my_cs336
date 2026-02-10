import fasttext
from pathlib import Path
from typing import Any
import random

_NSFW_MODEL_PATH = "cs336_data/jigsaw_fasttext_bigrams_nsfw_final.bin"
_TOXIC_MODEL_PATH = "cs336_data/jigsaw_fasttext_bigrams_hatespeech_final.bin"

warc_path = Path(__file__).parent / "warc_extracted_text.txt"


def classify_nsfw(text: str) -> tuple[Any, float]:
    text = text.replace("\n", " ").strip()
    NSFW_classifier = fasttext.load_model(_NSFW_MODEL_PATH)
    label, conf = NSFW_classifier.predict(text, k=1)
    nsfw_label, nsfw_conf = label[0].replace('__label__', ''), conf[0]
    return (nsfw_label, nsfw_conf)

def classify_toxic_speech(text: str) -> tuple[Any, float]:
    text = text.replace("\n", " ").strip()
    TOXIC_classifier = fasttext.load_model(_TOXIC_MODEL_PATH)
    label, conf = TOXIC_classifier.predict(text, k=1)
    toxic_label, toxic_conf = label[0].replace('__label__', ''), conf[0]
    return (toxic_label, toxic_conf)

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

    for i, sample in enumerate(samples):
        print(sample)
        print(classify_nsfw(sample))
        print(classify_toxic_speech(sample))
        print("========================\n")

if __name__ == "__main__":
    main()