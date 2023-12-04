from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np

UNK = "<UNK>"


class Vocab:
    """
    reads file and builds a vocabulary
    """

    def __init__(self, file, min_freq=5):
        self.word2idx = defaultdict(self.create_unk_defaultdict)
        self.idx2word = []
        self.freqs = {}
        self.min_freq = min_freq
        self.file = file
        if self.file is not None:
            self._build_vocab()
            self.prepare_frequences()

    def create_unk_defaultdict(self):
        return self.word2idx[UNK]

    @property
    def unk_id(self):
        return self.word2idx[UNK]
    
    @staticmethod
    def from_list(words: List[str]) -> Vocab:
        self = Vocab(None)
        for i, word in enumerate(words):
            self.word2idx[word] = i
            self.idx2word.append(word)
            self.freqs[word] = None
        assert not UNK in self.word2idx
        self.word2idx[UNK] = len(self.word2idx)
        self.idx2word.append(UNK)
        self.freqs[UNK] = 0
        return self

    def _build_vocab(self, max_words=100000):
        """
        builds vocabulary from file
        """
        with open(self.file, "r") as f:
            for line in f:
                for word in line.split():
                    self.freqs[word] = self.freqs.get(word, 0) + 1
        it = 0
        for word, freq in sorted(self.freqs.items(), key=lambda x: x[1], reverse=True):
            # sort ids by frequency (decreasing)
            if freq >= self.min_freq and it < max_words:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word.append(word)
                it += 1
        # add UNK token
        assert not UNK in self.word2idx
        self.word2idx[UNK] = len(self.word2idx)
        self.idx2word.append(UNK)
        self.freqs[UNK] = 0

    def save(self, file):
        """
        saves vocabulary to file
        """
        with open(file, "w") as f:
            for id in range(len(self.idx2word)):
                word = self.idx2word[id]
                f.write(f"{word} {self.freqs[word]} {self.word2idx[word]}\n")

    @staticmethod
    def from_hypernyms(file) -> Vocab:
        self = Vocab(None)
        self.normalized_freqs = []
        i = 0
        with open(file, "r") as f:
            f.readline()  # skip first line
            for line in f:
                lemma, pos, hypernyms = line.strip().split(",")
                lemma = lemma.strip()
                hypernyms = hypernyms.strip().split("|")
                hypernyms = [w for h in hypernyms for w in h.split(";") if w != ""]
                for word in [lemma, *hypernyms]:
                    # format: word x1 x2 ... xn
                    self.word2idx[word] = i
                    self.idx2word.append(word)
                    self.freqs[word] = None
                    self.normalized_freqs.append(1.0)
                    i += 1
        assert not UNK in self.word2idx
        self.word2idx[UNK] = len(self.word2idx)
        self.idx2word.append(UNK)
        self.freqs[UNK] = 0
        return self

    def __len__(self):
        return len(self.idx2word)

    def prepare_frequences(self):
        freqs = np.array(
            list(self.freqs[self.idx2word[idx]] for idx in range(len(self)))
        )
        denom = freqs.sum()
        self.normalized_freqs = freqs / denom
