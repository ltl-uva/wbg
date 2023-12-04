from math import pi

import geoopt
# fond embedded pdf
import matplotlib
import matplotlib.pyplot as plt
import torch

from wbgauss.sphere import LogDiagScale, PoleWBGaussian

matplotlib.rc("pdf", fonttype=42)


def tangent_loss(dist, y):
    return dist.cross_fy(y)


def wrapped_loss(dist, y):
    return dist.wrapped_cross_fy(y)


def name_loss_latex(loss_func):
    if loss_func == tangent_loss:
        return "tangent $L^x$"
    else:
        return "wrapped $L^x$"


def fit(alpha=1.1):
    d = 3
    n_samples = 10000
    n_iter = 5000

    torch.manual_seed(42)
    torch.set_default_dtype(torch.double)

    sphere = geoopt.manifolds.SphereExact()
    true_loc = sphere.random_uniform(d)
    true_scale = -torch.ones(d - 1)
    true_scale[-1] = -4
    true_scale[-2] = -3

    true_wbg = PoleWBGaussian(loc=true_loc, scale=LogDiagScale(true_scale), alpha=alpha)

    Y = true_wbg.rsample(shape=(n_samples,))

    print(sphere.dist(Y, true_loc).max())
    plt.figure(figsize=(5, 5))
    plt.axhline(0, ls=":", color="k")

    loc_init_ = sphere.random_uniform(d)
    scale_init_ = torch.randn(d - 1) - 1
    for loss_func in (tangent_loss, wrapped_loss):
        print()
        print()
        print(f"{loss_func=}")

        print("Loss at true parameters:", loss_func(true_wbg, Y).mean().item())

        loc = geoopt.tensor.ManifoldParameter(loc_init_.clone(), manifold=sphere)
        scale = torch.nn.Parameter(scale_init_.clone())
        parameters = [loc, scale]
        opt = geoopt.optim.RiemannianAdam(parameters, lr=0.01)

        w2_dists = []

        for it in range(n_iter):
            opt.zero_grad()
            wbg = PoleWBGaussian(loc=loc, scale=LogDiagScale(scale), alpha=alpha)

            loss = loss_func(wbg, Y).mean()
            loss.backward()
            w2_dists.append(true_wbg.wasserstein_distance(wbg).item())
            if it % 100 == 0:
                print(f"{it=} {loss.item()=}")
            opt.step()

        plt.plot(w2_dists, label=f"{loss_func=}")
        torch.save(
            [w2_dists, wbg, true_wbg], rf"w2_{alpha=}_{name_loss_latex(loss_func)}.pt"
        )

    plt.legend()
    plt.semilogx()
    plt.show()


def main():
    for alpha in 1.01, 1.1, 1.25, 1.5:
        fit(alpha=alpha)


if __name__ == "__main__":
    main()
