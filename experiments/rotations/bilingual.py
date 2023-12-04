"""
Bilingual dictionary alignment.
Dinu data: https://zenodo.org/record/2654864#.XX_GZCVS9TZ

possibly mirrored by Artetxe: https://github.com/artetxem/vecmap/blob/master/get_data.sh

X = english
Y = italian
find Q st Y = QX.

"""

import numpy as np
import torch

path = "."
enf = "EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt"
itf = "IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt"


def load_bilingual():
    with open(path + "OPUS_en_it_europarl_train_5K.txt") as f:
        pairs = [line.strip().split() for line in f]

    unique_en = set(w for w, _ in pairs)
    unique_it = set(w for _, w in pairs)

    en_emb = {}
    it_emb = {}

    with open(path + enf) as f:
        next(f)  # skip header

        for line in f:
            word, vec = line.strip().split(" ", maxsplit=1)
            if word in unique_en:
                en_emb[word] = np.fromstring(vec, dtype=np.double, sep=" ")

    # with open(path + itf, encoding='utf-8') as f:
    with open(path + itf, errors="replace") as f:
        next(f)  # skip header

        for line in f:
            word, vec = line.strip().split(" ", maxsplit=1)
            if word in unique_it:
                it_emb[word] = np.fromstring(vec, dtype=np.double, sep=" ")

    en_array = np.row_stack([en_emb[w] for w, _ in pairs])
    it_array = np.row_stack([it_emb[w] for _, w in pairs])

    en_t = torch.from_numpy(en_array)
    it_t = torch.from_numpy(it_array)
    torch.save((en_t, it_t), "dinu_emb_pairs.pt")

    # save embeddings_dict
    en_emb = {w: torch.from_numpy(v) for w, v in en_emb.items()}


def main():
    load_bilingual()


if __name__ == "__main__":
    main()
