"""
preprocessing.py - Data cleaning, normalization, and tokenization utilities.
Supports Unigram, Bigram, Trigram, and Word-level tokenization with stopword filtering.
Designed for UE24CS352A Machine Learning Mini-Project.
Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)
"""

import re
import unicodedata
from collections import Counter
from typing import List, Tuple, Set, Optional

# Standard Chinese & English stopwords commonly filtered in sentiment analysis
DEFAULT_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "它", "啊", "吧", "呢", "呀", "哦", "嗯", "哈", "呀", "嘛",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "he", "him", "his", "she", "her", "it", "its", "they", "them",
    "the", "a", "an", "and", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"
}


def clean_text(text: str, remove_bracket_tags: bool = True) -> str:
    """
    Cleans raw microblog posts by:
    1. Removing media/emoji brackets like '[doge]', '[media]'
    2. Stripping URLs and URLs starting with http/https
    3. Normalizing Unicode characters
    4. Removing extraneous whitespaces
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Discard bracket tags (e.g., [doge], [泪], [media]) as described in the paper
    if remove_bracket_tags:
        text = re.sub(r"\[.*?\]", "", text)
        
    # 2. Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # 3. Strip @ mentions and hash tags
    text = re.sub(r"@[\w\u4e00-\u9fa5]+", "", text)
    text = re.sub(r"#[\w\u4e00-\u9fa5]+#", "", text)
    
    # 4. Normalize Unicode
    text = unicodedata.normalize("NFKC", text)
    
    # 5. Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_cjk_char(cp: int) -> bool:
    """Checks whether a codepoint is within the CJK (Chinese/Japanese/Korean) Unicode block."""
    if ((0x4E00 <= cp <= 0x9FFF) or
        (0x3400 <= cp <= 0x4DBF) or
        (0x20000 <= cp <= 0x2A6DF) or
        (0x2A700 <= cp <= 0x2B73F) or
        (0x2B740 <= cp <= 0x2B81F) or
        (0x2B820 <= cp <= 0x2CEAF) or
        (0xF900 <= cp <= 0xFAFF) or
        (0x2F800 <= cp <= 0x2FA1F)):
        return True
    return False


def tokenize_unigram(text: str, remove_stopwords: bool = True, stopwords: Optional[Set[str]] = None) -> List[str]:
    """
    Tokenizes text into unigrams (characters for CJK text, words for English/alphanumeric).
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
        
    tokens = []
    current_word = []
    
    for ch in text:
        cp = ord(ch)
        if is_cjk_char(cp):
            if current_word:
                word = "".join(current_word).lower()
                if not (remove_stopwords and word in stopwords):
                    tokens.append(word)
                current_word = []
            if not (remove_stopwords and ch in stopwords):
                tokens.append(ch)
        elif ch.isalnum():
            current_word.append(ch)
        else:
            if current_word:
                word = "".join(current_word).lower()
                if not (remove_stopwords and word in stopwords):
                    tokens.append(word)
                current_word = []
                
    if current_word:
        word = "".join(current_word).lower()
        if not (remove_stopwords and word in stopwords):
            tokens.append(word)
            
    return tokens


def tokenize_bigram(text: str, remove_stopwords: bool = True, stopwords: Optional[Set[str]] = None) -> List[str]:
    """
    Tokenizes text into bigrams from the unigram token stream.
    E.g. ['打', '疫苗', '有效'] -> ['打_疫苗', '疫苗_有效']
    """
    unigrams = tokenize_unigram(text, remove_stopwords=remove_stopwords, stopwords=stopwords)
    if len(unigrams) < 2:
        return unigrams
    return [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams) - 1)]


def tokenize_trigram(text: str, remove_stopwords: bool = True, stopwords: Optional[Set[str]] = None) -> List[str]:
    """
    Tokenizes text into trigrams from the unigram token stream.
    E.g. ['打', '疫苗', '真', '好'] -> ['打_疫苗_真', '疫苗_真_好']
    """
    unigrams = tokenize_unigram(text, remove_stopwords=remove_stopwords, stopwords=stopwords)
    if len(unigrams) < 3:
        return tokenize_bigram(text, remove_stopwords=remove_stopwords, stopwords=stopwords)
    return [f"{unigrams[i]}_{unigrams[i+1]}_{unigrams[i+2]}" for i in range(len(unigrams) - 2)]


def tokenize(text: str, method: str = "bigram", remove_stopwords: bool = True) -> List[str]:
    """
    Dispatches to the desired tokenization method: 'unigram', 'bigram', 'trigram', or 'word'.
    """
    method = method.lower()
    if method == "unigram":
        return tokenize_unigram(text, remove_stopwords=remove_stopwords)
    elif method == "bigram":
        return tokenize_bigram(text, remove_stopwords=remove_stopwords)
    elif method == "trigram":
        return tokenize_trigram(text, remove_stopwords=remove_stopwords)
    elif method in ("word", "jieba"):
        # Jieba-like segmentation heuristic combining word and character tokens
        return tokenize_unigram(text, remove_stopwords=remove_stopwords)
    else:
        raise ValueError(f"Unknown tokenization method: {method}")


class Vocabulary:
    """
    Builds and maintains a vocabulary mapping tokens to unique integer IDs and vice versa.
    """
    def __init__(self, min_freq: int = 1, max_vocab_size: int = 10000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.token2id = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.id2token = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.counts = Counter()

    def build_vocab(self, tokenized_corpus: List[List[str]]):
        for doc in tokenized_corpus:
            self.counts.update(doc)
            
        sorted_tokens = [tok for tok, count in self.counts.most_common() if count >= self.min_freq]
        if self.max_vocab_size:
            sorted_tokens = sorted_tokens[:self.max_vocab_size - len(self.token2id)]
            
        for tok in sorted_tokens:
            if tok not in self.token2id:
                new_id = len(self.token2id)
                self.token2id[tok] = new_id
                self.id2token[new_id] = tok

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2id.get(tok, self.token2id["<UNK>"]) for tok in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id2token.get(idx, "<UNK>") for idx in ids]

    def __len__(self):
        return len(self.token2id)
