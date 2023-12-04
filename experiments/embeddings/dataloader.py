import logging
from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from vocab import Vocab

logger = logging.getLogger("dataloader.py")
logging.basicConfig(level=logging.INFO)


def load_vocab(path, n_stop_words=100):
    vocab = Vocab.from_file(path)

    from collections import Counter

    context_counts = Counter(vocab.freqs)
    context_counts = sorted(context_counts.items(), key=lambda x: x[1], reverse=True)
    context_counts = context_counts[:n_stop_words]
    context_words = [x[0] for x in context_counts]
    context_freqs = [x[1] for x in context_counts]

    return vocab, set(context_words)


class HypernymTree:
    def __init__(self, dataset: List[Tuple[str, str]]):
        """
        Builds a tree from the dataset: from parent to leaves
        """
        self.dataset = dataset
        self.tree, self.parents, self.children, self.siblings = self._build_tree()

    def print(self):
        print(self.tree)

    def is_root(self, word):
        return word in self.tree

    def get_subtree(self, word) -> list[str]:
        """
        Recursively get all nodes of word"""
        return self._get_subtree(word)

    def _get_subtree(self, word) -> list[str]:
        if word not in self.children:
            return [word]
        children = self.children[word]
        subtree = [word]
        for child in children:
            subtree += self._get_subtree(child)
        return subtree

    def _build_tree(self):
        print("building tree")
        # first find roots
        roots = set()
        parents = {}
        children = {}
        siblings = {}
        for child, parent in self.dataset:
            roots.add(parent)
            parents[child] = parent
            if parent not in children:
                children[parent] = set()
            children[parent].add(child)

        roots = roots - set([child for child, parent in self.dataset])
        roots = list(roots)

        # get_siblings:
        for root in roots:
            siblings[root] = set()
        for parent in children:
            cur_children = children[parent]
            for child in cur_children:
                siblings[child] = cur_children - set([child])

        # build tree
        tree = {}
        for root in roots:
            tree[root] = self._build_tree_from_root(root)
            parents[root] = None

        return tree, parents, children, siblings

    def _build_tree_from_root(self, root):
        tree = {}
        for child, parent in self.dataset:
            if parent == root:
                tree[child] = self._build_tree_from_root(child)
        return tree

    def get_candidates(self, word):
        # candidates for negative samples
        to_exclude = [word]
        while word in self.parents:
            word = self.parents[word]
            to_exclude.append(word)  # exclude all ancestors
        to_exclude.extend(self.get_subtree(word))  # exclude all descendants

        candidates = []
        for root in self.tree:
            candidates.extend(self.get_subtree(root))
        candidates = list(set(candidates) - set(to_exclude))
        return candidates


class HypernymsDataset:
    # positives: (hypernym_id, lemma_id)
    # negatives: add lemma_ids
    def __init__(
        self,
        file,
        n_negatives=5,
        debug=False,
        only_selected_words=False,
        pickle_path=None,
    ):
        self.only_selected_words = only_selected_words  # filter out source words that are not in the selected words list
        self.pickle_path = pickle_path

        self.file = file
        self.dataset, self.vocab = self._build_dataset(debug)
        self.tree = HypernymTree(
            [
                (self.vocab.idx2word[hypernym_id], self.vocab.idx2word[lemma_id])
                for hypernym_id, lemma_id in self.dataset
            ]
        )
        self.tree.print()
        print("parents", self.tree.parents)
        print("children", self.tree.children)
        print("siblings", self.tree.siblings)
        self.n_negatives = n_negatives

    def read_csv(self, skip_row=True) -> List[Tuple[str, str]]:
        with open(self.file, "r") as f:
            if skip_row:
                f.readline()
            data = []
            for line in f.readlines():
                child, parent = line.strip().split(",")
                data.append((child.strip(), parent.strip()))
        return data

    def read_wordnet(self, skip_row=True) -> List[Tuple[str, str]]:
        with open(self.file, "r") as f:
            if skip_row:
                f.readline()  # skip header
            data = []
            for line in tqdm(f.readlines()):
                lemma, pos, hypernyms = line.strip().split(",")
                lemma = lemma.strip()
                hypernyms = hypernyms.strip().split("|")
                hypernyms = [w for h in hypernyms for w in h.split(";") if w != ""]

                for hypernym in hypernyms:
                    data.append((lemma, hypernym))
        return data

    def _build_dataset(self, debug) -> Tuple[List[Tuple[str, str]], Vocab]:
        # the file format: csv file with columns: lemma, POS, hypernyms
        # hypernyms are separated by |
        # leave only those hypernims and lemmas that are single words
        # and are in vocab

        # dataset: (hypernym_id, lemma_id)
        # RETURN dataset, vocab

        n_not_in_vocab = 0
        n_not_single_word = 0
        n_success = 0

        self.dataset = []
        # get all unique parent, child
        all_words = set()
        for child, parent in self.read_csv(skip_row=True):
            all_words.add(child)
            all_words.add(parent)
            self.dataset.append((child, parent))
        # build vocab
        vocab = Vocab.from_list(list(all_words))
        # convert dataset to idx
        self.dataset = [
            (vocab.word2idx[child], vocab.word2idx[parent])
            for child, parent in self.dataset
        ]
        self.dataset = np.array(self.dataset)

        logger.info(f"Dataset size: {len(self.dataset)}")
        logger.info(f"Skipped {n_not_in_vocab} hypernyms not in vocab")
        logger.info(f"Skipped {n_not_single_word} hypernyms that are not single words")
        logger.info(f"Success: {n_success}")
        return self.dataset, vocab

    def __getitem__(self, idx):
        child, parent = self.dataset[idx]
        # sample negatives
        candidates = self.tree.get_candidates(self.vocab.idx2word[child])
        candidates = list(candidates)
        if len(candidates) == 0:
            negs = [-1] * self.n_negatives
        else:
            negs = np.random.choice(candidates, size=self.n_negatives, replace=True)
        negs = np.array([self.vocab.word2idx[neg] for neg in negs])
        return child, parent, negs

    def __len__(self):
        return len(self.dataset)
