import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    import hashlib
    import re

    neighbors = {
        "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg",
        "y": "tugh", "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
        "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc",
        "g": "ftyhbv", "h": "gyujnb", "j": "huikmn", "k": "jiolm",
        "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb",
        "b": "vghn", "n": "bhjm", "m": "njk",
    }
    protected = {
        "no", "not", "never", "neither", "nor", "nothing", "nobody", "nowhere",
        "without", "cannot", "can't", "don't", "doesn't", "didn't", "isn't",
        "wasn't", "weren't", "aren't", "won't", "wouldn't", "couldn't", "shouldn't",
        "haven't", "hasn't", "hadn't",
    }
    text = example["text"]
    digest = hashlib.sha256(("42\0" + text).encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    parts = []
    last = 0
    for match in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text):
        parts.append(text[last:match.start()])
        word = match.group()
        if len(word) >= 4 and "'" not in word and word.lower() not in protected:
            if rng.random() < 0.4:
                position = rng.randrange(1, len(word) - 1)
                letter = rng.choice(neighbors[word[position].lower()])
                if word[position].isupper():
                    letter = letter.upper()
                word = word[:position] + letter + word[position + 1:]
        parts.append(word)
        last = match.end()
    parts.append(text[last:])
    example["text"] = "".join(parts)

    ##### YOUR CODE ENDS HERE ######

    return example
