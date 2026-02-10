from nltk import word_tokenize
import regex as re

def gropher_quality_filter(text: str) -> bool:
    words = word_tokenize(text)
    res = True
    
    # first rule: Contain less than 50 or more than 100,000 words
    if len(words) < 50 or len(words) > 100000:
        res = False
    
    # Second rule: Have a mean length outside the range of 3 to 10 characters
    mean_len = sum(len(word) for word in words) / len(words)
    if mean_len < 3 or mean_len > 10:
        res = False
    
    # Third rule: Have more than 30% of lines ending with an ellipsii ("...")
    lines = text.split('\n')
    non_empty_lines = [l.strip() for l in lines if len(l) != 0]
    if non_empty_lines:
        ellipsis_line = sum([1 for l in non_empty_lines if l.endswith('...')])
        ratio = ellipsis / len(non_empty_lines)
        if ratio > 0.3:
            res = False
    
    # Fourth rule: Contain less than 80% of words with at least one alphabetic character
    ALPHABETIC_PATTERN = re.compile(r"[A-Za-z]")
    alpha_words_num = sum([1 for word in words if ALPHABETIC_PATTERN.search(word)])
    alpha_words_ratio = alpha_words_num / len(words)
    if alpha_words_ratio < 0.8:
        res = False

    return res
