"""Variational inference for Procrustes

Let Q be an RV in SO(n) such that:

    p(Q) = 1/vol(SO(n))

    p(y |x, Q) = N(Qx, I_n)

        so -logp(Y | X, Q) = 1/2 || Y - XQ ||^2 + mn/2 log(2pi)

    we fit q(Q; params)

    to maximize the ELBO

    ln p(Y|X) >= E_q [ log ( p(Y,Q|X) / q(Q) ) ]
               = E_q [ log p(Y | X, Q) ] - KL[q(Q) : p(Q)]
               = E_q [ log p(Y | X, Q) ] - E_q [ log q(Q) - log p(Q) ]
               = E_q [ log p(Y | X, Q) ] - E_q [ log q(Q) ] + E_q [log p(Q) ]
               = E_q [ log p(Y | X, Q) ] - E_q [ log q(Q) ] + log 1/vol(SO(n))
               = E_q [ log p(Y | X, Q) ] - E_q [ log q(Q) ] - log vol(SO(n))

    or, really, minimize the neg-ELBO

   -ln p(Y|X) <= E_q [ -log p(Y | X, Q) ] + E_q [ log q(Q) ] + log vol(SO(n))

"""

import math
import pickle as pk
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import logm, orthogonal_procrustes
from scipy.stats import special_ortho_group
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from wbgauss.scale_operators import LogDiagScale
from wbgauss.special_orthogonal import PoleWBGaussian
from wbgauss.utils.skew_sym import log_volume_so_n

PI = math.pi
LOG_2PI = math.log(2 * math.pi)


def evaluate(path, params=None, en_PCA=None, it_PCA=None, Qdet_np=None):
    # load params
    if params is None:
        loc_param, scale_param = torch.load("params.pt")
    else:
        loc_param, scale_param = params
    # load point solution
    if Qdet_np is None:
        Qdet_np = pk.load(open("Qdet_np.pkl", "rb"))
    # load PCA weights
    if en_PCA is None:
        en_PCA = pk.load(open("en_PCA.pkl", "rb"))
    if it_PCA is None:
        it_PCA = pk.load(open("it_PCA.pkl", "rb"))

    loc = torch.matrix_exp(loc_param - loc_param.T)
    scale = LogDiagScale(scale_param)
    wbg = PoleWBGaussian(loc, scale, alpha=1.1)
    n_samples = 30
    Qs = []
    for i in range(n_samples):
        Q_sample, _log_qQ = wbg.sample_and_log_prob()
        Qs.append(Q_sample.detach().numpy())
    Qs = np.array(Qs)

    # evaluate on the test data
    ###########################

    path = "."
    enf = "EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt"
    itf = "IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt"

    # load test data
    with open(path + "OPUS_en_it_europarl_test.txt") as f:
        pairs = [line.strip().split() for line in f]

    test_en = list(w for w, _ in pairs)
    test_it = list(w for _, w in pairs)
    # count the number of occurences of each word
    test_en_counts = Counter(test_en)
    test_it_counts = Counter(test_it)

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

    test_en_emb = np.array([en_emb[w] for w in test_en])
    test_it_emb = np.array([it_emb[w] for w in test_it])
    print("test shape", test_en_emb.shape)

    # project test data
    test_en_emb = en_PCA.transform(test_en_emb)
    test_it_emb = it_PCA.transform(test_it_emb)

    print("transformed test shape", test_en_emb.shape)

    # test_en_emb, test_it_emb
    test_accs = []
    for Q_sample in Qs:
        test_accs.append(test_accuracy(Q_sample, test_en_emb, test_it_emb))
    test_accs = np.array(test_accs)
    print("test accs", test_accs)
    print("mean", np.mean(test_accs))
    print("std", np.std(test_accs))
    # save the accuracies in a readable way
    # format: en, it, acc_0, acc_1, ...
    with open("accs.txt", "w") as f:
        # save test_accs
        for i in range(len(test_accs)):
            f.write(f"{test_accs[i]} ")

    print("static accuracy", test_accuracy(Qdet_np, test_en_emb, test_it_emb))

    dists_all = []
    for Q_sample in Qs:
        it = test_en_emb @ Q_sample
        print(np.linalg.norm(it - test_it_emb, axis=-1).mean())
        dists_all.append(np.linalg.norm(it - test_it_emb, axis=-1))
    dists_all = np.array(dists_all)
    # save the distances in a readable way
    # format: en, it, dist_0, dist_1, ...
    with open("dists.txt", "w") as f:
        for i in range(len(test_en)):
            f.write(f"{test_en[i]} {test_it[i]} ")
            for j in range(n_samples):
                f.write(f"{dists_all[j, i]} ")
            f.write("\n")

    stds = np.std(dists_all, axis=0)
    # word pairs with highest std
    idx = np.argsort(stds)
    most_uncertain = [(test_en[i], test_it[i], stds[i]) for i in idx[-10:]]
    print("most uncertain")
    for en, it, std in most_uncertain:
        print(
            f'"{en}"',
            f'"{it}"',
            std,
            "the number of occs (en, it):",
            test_en_counts[en],
            test_it_counts[it],
        )
    most_certain = [(test_en[i], test_it[i], stds[i]) for i in idx[:10]]
    print("most certain")
    for en, it, std in most_certain:
        print(
            f'"{en}"',
            f'"{it}"',
            std,
            "the number of occs (en, it):",
            test_en_counts[en],
            test_it_counts[it],
        )

    print("deterministic")
    it = test_en_emb @ Qdet_np
    print(np.linalg.norm(it - test_it_emb, axis=-1).mean())

    for en, it in pairs[:10]:
        print("-----------------")
        print(f'"{en}"', f'"{it}"')
        # get embeddings of en, it
        cur_en_emb = en_emb[en]
        cur_it_emb = it_emb[it]
        # project embeddings
        cur_en_emb = en_PCA.transform(cur_en_emb.reshape(1, -1))
        cur_it_emb = it_PCA.transform(cur_it_emb.reshape(1, -1))
        # get distance
        print(
            "point distance", np.linalg.norm(cur_en_emb @ Qdet_np - cur_it_emb, axis=-1)
        )

        dists = []
        for Q_sample in Qs:
            dists.append(np.linalg.norm(cur_en_emb @ Q_sample - cur_it_emb, axis=-1))
        dists = np.array(dists)
        print("mean", dists.mean())
        print("std", dists.std())


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--n_components",
        type=int,
        default=20,
        help="number of components for dim reduction of embeddings",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")

    parser.add_argument(
        "--full_dim",
        action="store_true",
        help="no dimentionality reduction (challenging)",
    )
    parser.add_argument("--toy", action="store_true")
    parser.add_argument(
        "--init", action="store_true", help="initialize with deterministic Procrustes"
    )

    args = parser.parse_args()
    toy = args.toy

    torch.set_default_dtype(torch.double)
    torch.set_default_dtype(torch.double)
    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.RandomState(1)

    print(args)

    # generate dataset
    ##################

    if toy:
        n = 10
        n_samples = 100
        sigma = 0.1  # noise after rotation

        Qtrue_np = special_ortho_group.rvs(n, random_state=rng)
        X_np, *_ = make_blobs(n_samples, n, centers=n, random_state=rng)
        Y_np = X_np @ Qtrue_np + sigma * rng.randn(n_samples, n)
        X = torch.from_numpy(X_np)
        Y = torch.from_numpy(Y_np)

    else:
        X, Y = torch.load("dinu_emb_pairs.pt")

        full_dimentionality = args.full_dim
        n_components = args.n_components

        X_np = X.numpy()
        Y_np = Y.numpy()

        class PCA_dummy:
            def __init__(self):
                pass

            def fit_transform(self, X):
                return X

            def transform(self, X):
                return X

        en_PCA = PCA_dummy()
        it_PCA = PCA_dummy()

        if not full_dimentionality and n_components < X.shape[1]:
            print("reducing dimentionality")
            en_PCA = PCA(n_components=n_components)
            it_PCA = PCA(n_components=n_components)
            X_np = en_PCA.fit_transform(X.numpy())
            Y_np = it_PCA.fit_transform(Y.numpy())
            # save PCA transforms
            pk.dump(en_PCA, open("en_PCA.pkl", "wb"))
            pk.dump(it_PCA, open("it_PCA.pkl", "wb"))

        if n_components == 300:
            assert np.allclose(X_np, X.numpy())

        X = torch.from_numpy(X_np)
        Y = torch.from_numpy(Y_np)

        print("X.shape", X.shape)
        print("Y.shape", Y.shape)

        n_samples, n = X.shape

    # deterministic procrustes
    ##########################

    # error on the same magnitude as noise
    Qdet_np, _ = orthogonal_procrustes(X_np, Y_np)
    # save orthogonal procrustes solution
    pk.dump(Qdet_np, open("Qdet_np.pkl", "wb"))

    # check that qdet_np is in SO(n)
    assert np.allclose(Qdet_np @ Qdet_np.T, np.eye(n))
    if toy:
        print("Deterministic Procrustes error")
        print(np.linalg.norm(Qdet_np - Qtrue_np))

    print("Fitting error (minimzation obj)")
    determ_error = 0.5 * np.sum((Y_np - X_np @ Qdet_np) ** 2)
    print(determ_error)

    # variational inference
    #######################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # move data to device
    X = X.to(device)
    Y = Y.to(device)

    alpha = 1.1
    n_iter = 5000
    lr = args.lr
    # n_iter = 100000
    # lr = 0.1

    if toy:
        Qtrue = torch.from_numpy(Qtrue_np)

    if args.init:
        # init good solution
        A = logm(Qdet_np)
        A[np.triu_indices_from(A)] = 0
        loc_param_ = torch.from_numpy(A).to(device)
        print("init from point solution")
    else:
        # random init
        loc_param_ = torch.randn(n, n, device=device)

    loc_param = torch.nn.Parameter(loc_param_)

    scale_param_ = torch.randn(n * (n - 1) // 2, device=device) + 1
    scale_param = torch.nn.Parameter(scale_param_)

    log_vol = log_volume_so_n(n)

    opt = torch.optim.Adam([loc_param, scale_param], lr)
    print(f"{n=}")

    elbos = []
    errors = []
    locs = []

    mf_errors = []
    init_scale = None

    for it in tqdm(range(n_iter)):
        opt.zero_grad()

        loc = torch.matrix_exp(loc_param - loc_param.T)
        scale = LogDiagScale(scale_param, rcond=1e-100)

        if it == 0:
            init_scale = torch.diagonal(scale.as_tensor().detach())

        wbg = PoleWBGaussian(loc, scale, alpha=alpha, validate_args=True)

        # approximate expectation of log likelihood E_q [ p(y | x) ]
        Q_sample, log_qQ = wbg.sample_and_log_prob(shape=())

        # -elbo = E_q [ -log p(Y | X, Q) ] + E_q [ log q(Q) ] + log vol(SO(n))

        # E_q[-logp(Y|Q,X)]: one big gaussian NLL
        fitting_error = 0.5 * (Y - X @ Q_sample).square().sum()
        neg_ll = fitting_error + (n_samples * n) / 2 * LOG_2PI

        neg_elbo = neg_ll + log_qQ + log_vol
        neg_elbo.backward()
        opt.step()

        elbos.append(neg_elbo.detach().item())
        errors.append(fitting_error.detach().item())

        mf_error = 0.5 * (Y - X @ loc).square().sum().detach()
        mf_errors.append(mf_error)

        if toy:
            locs.append(torch.norm(loc - Qtrue).detach().item())

        if it % 100 == 0:
            tqdm.write(
                f"iter {it} -elbo {neg_elbo.item():.2f} "
                f"error {fitting_error.item():.2f} "
                f"mf error {mf_error.item():.2f}"
            )

        if int(n_iter * 0.5) == it or int(n_iter * 0.75) == it:
            # reduce learning rate
            for param_group in opt.param_groups:
                param_group["lr"] = lr / 10

    scale_ = torch.diag(scale.as_tensor().detach())
    print(
        "cosine similarity init_scale and scale",
        torch.nn.functional.cosine_similarity(init_scale, scale_, dim=-1),
    )

    plt.figure()
    plt.title("elbo")
    plt.plot(elbos)
    plt.semilogy()
    plt.savefig("elbo.png")
    plt.figure()
    plt.title("error")
    plt.plot(errors)
    plt.plot(mf_errors, ls=":", label="mean")
    plt.axhline(determ_error)
    plt.semilogy()
    plt.legend()
    plt.savefig("error.png")
    if toy:
        plt.figure()
        plt.plot(locs)
        plt.semilogy()
        plt.title("location error")
    # save plot in file
    plt.show()

    # save the params
    path = f"models/params_{args.lr}_{args.n_components}.pt"
    print(path)
    torch.save((loc_param, scale_param), path)
    evaluate(path, (loc_param, scale_param), en_PCA, it_PCA, Qdet_np)


def test_accuracy(rotation_matrix, en_emb, it_emb) -> float:
    # compute the accuracy
    # rotate the english embeddings
    en_emb = en_emb @ rotation_matrix
    # compute the cosine similarity
    sim = np.sum(en_emb[:, None] * it_emb[None, :], axis=-1)
    # take the argmax
    argmax = np.argmax(sim, axis=-1)
    # compute the accuracy
    accuracy = (argmax == np.arange(len(argmax))).mean()
    return accuracy


if __name__ == "__main__":
    main()
    # evaluate()
