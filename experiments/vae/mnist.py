import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import (Timer, VAEMonitor, WarmupScheduler, get_csv_logger,
                   get_wandb_logger, set_default_tensor_type, set_seed)

from wbgauss.sphere import (EmbeddedWBGaussian, LogDiagScale, LogDiagScaleProj,
                            PoleWBGaussian)
from wbgauss.utils.hyperspherical import HypersphericalUniform, VonMisesFisher


class ModelVAE(torch.nn.Module):
    def __init__(
        self,
        h_dim,
        z_dim,
        activation=F.relu,
        distribution="wbgauss_amb",
        i_dim=784,
        device="cpu",
        alpha=1.1,
    ):
        """
        ModelVAE initializer
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        :param i_dim: dimension of the input
        :param device: string, either `cpu` or `cuda`
        :param alpha: float, alpha parameter for BetaGaussian
        :param good_sigma_init: bool, whether to use good sigma initialization for WBetaGaussian
        """
        super(ModelVAE, self).__init__()

        self.z_dim, self.activation, self.distribution = z_dim, activation, distribution
        self.i_dim = i_dim

        # 2 hidden layers encoder
        self.fc_e0 = nn.Linear(i_dim, h_dim * 2)
        self.fc_e1 = nn.Linear(h_dim * 2, h_dim)

        if self.distribution == "wbgauss":
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, z_dim - 1)
        elif self.distribution == "wbgauss_amb":
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, z_dim)
        elif self.distribution == "vmf":
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, 1)
        else:
            raise NotImplementedError

        # 2 hidden layers decoder
        self.fc_d0 = nn.Linear(z_dim, h_dim)
        self.fc_d1 = nn.Linear(h_dim, h_dim * 2)
        self.fc_logits = nn.Linear(h_dim * 2, i_dim)
        self.device = device

        self.alpha = alpha  # for BetaGaussian

    def encode(self, x):
        # 2 hidden layers encoder
        x = self.activation(self.fc_e0(x))
        x = self.activation(self.fc_e1(x))

        if self.distribution == "vmf":
            # compute mean and concentration of the von Mises-Fisher
            z_mean = self.fc_mean(x)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.fc_var(x)) + 1
        elif self.distribution == "wbgauss" or self.distribution == "wbgauss_amb":
            z_mean = self.fc_mean(x)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            z_var = self.fc_var(x)  # afterward we will use sigmoid constraints
        else:
            raise NotImplementedError

        return z_mean, z_var

    def decode(self, z):
        x = self.activation(self.fc_d0(z))
        x = self.activation(self.fc_d1(x))
        x = self.fc_logits(x)
        return x

    def reparameterize(self, z_mean, z_var):
        if self.distribution == "vmf":
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1, device=self.device)
            # todo replace p_z with uniform normal
        elif self.distribution == "wbgauss":
            q_z = PoleWBGaussian(
                z_mean,
                LogDiagScale(z_var),
                alpha=self.alpha,
            )
            p_z = HypersphericalUniform(self.z_dim - 1, device=self.device)
        elif self.distribution == "wbgauss_amb":
            q_z = EmbeddedWBGaussian(
                z_mean,
                LogDiagScaleProj(z_var, z_mean),
                alpha=self.alpha,
            )
            p_z = HypersphericalUniform(self.z_dim - 1, device=self.device)
        else:
            raise NotImplemented

        return q_z, p_z

    def forward(self, x):
        z_mean, z_var = self.encode(x)
        assert not torch.any(torch.isnan(z_var))
        q_z, p_z = self.reparameterize(
            z_mean,
            z_var,
        )
        z = q_z.rsample()
        x_ = self.decode(z)

        return (z_mean, z_var), (q_z, p_z), z, x_

    @property
    def dtype(self):
        return self.fc_e0.weight.dtype

    def register_step(self, step):
        self.step = step


def log_likelihood(model, x, n=10, i_dim=784):
    """
    :param model: model object
    :param optimizer: optimizer object
    :param n: number of MC samples
    :return: MC estimate of log-likelihood
    """

    z_mean, z_var = model.encode(x.reshape(-1, i_dim))
    q_z, p_z = model.reparameterize(z_mean, z_var)

    z = q_z.rsample(torch.Size([n]))
    x_mb_ = model.decode(z)
    log_p_z = p_z.log_prob(z)

    if isinstance(p_z, torch.distributions.Normal):
        log_p_z = log_p_z.sum(-1)

    log_p_x_z = -nn.BCEWithLogitsLoss(reduction="none")(
        x_mb_, x.reshape(-1, i_dim).repeat((n, 1, 1))
    ).sum(-1)

    log_q_z_x = q_z.log_prob(z)

    if isinstance(q_z, torch.distributions.Normal):
        log_q_z_x = log_q_z_x.sum(-1)

    return ((log_p_x_z + log_p_z - log_q_z_x).t().logsumexp(-1) - np.log(n)).mean()


def kl_divergence(q_z, z, p_z):
    """KL divergence between q(z|x) and p(z)
    param q_z: q(z|x)
    param z: latent variable (reuse random sample)
    param p_z: prior distribution
    """
    log_q = q_z.log_prob(z)
    if len(log_q.shape) == 2:  # n_samples, batch_size
        log_q = log_q.mean(dim=0)
    return log_q + p_z.entropy()


def train(
    model,
    optimizer,
    train_loader,
    i_dim=784,
    kl_factor=1.0,
    device="cpu",
    step=None,
    scheduler=None,
    epoch=None,
    clip_gradients=None,
):
    timer = Timer()
    timer.start()
    monitor = VAEMonitor()
    # get curent lr of optimizer
    for i, (x_mb, y_mb) in enumerate(train_loader):
        model.register_step(step)
        # move x_mb, y_mb to device
        x_mb = x_mb.to(device)
        y_mb = y_mb.to(device)

        optimizer.zero_grad()
        scheduler.step(epoch)

        # dynamic binarization
        x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).to(
            model.dtype
        )
        (z_mean, z_var), (q_z, p_z), z, x_mb_ = model(x_mb.reshape(-1, i_dim))

        monitor.update(z_mean, z)

        loss_recon = (
            nn.BCEWithLogitsLoss(reduction="none")(x_mb_, x_mb.reshape(-1, i_dim))
            .sum(-1)
            .mean()
        )

        if model.distribution == "normal":
            loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
        elif model.distribution == "wnormal":
            loss_KL = q_z.kl_divergence(z, p_z).mean()
        elif model.distribution == "vmf":
            loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
        elif model.distribution == "wbgauss" or model.distribution == "wbgauss_amb":
            loss_KL = kl_divergence(q_z, z, p_z).mean()  # reuse the samples
        else:
            raise NotImplemented

        loss = loss_recon + loss_KL * kl_factor

        loss.backward()
        if clip_gradients is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradients)
        optimizer.step()

    print("epoch took", timer.end(), "seconds")
    return step, {
        "||mean_z||_2": torch.norm(monitor.mean_z(), dim=0).item(),
        "std from loc": monitor.mean_dist().item(),
        "step": step,
        "lr": optimizer.param_groups[0]["lr"],
    }


def sample(model, p_z, i_dim=784):
    with torch.no_grad():
        samples = p_z.sample(torch.Size([64]))
        print(samples.shape)
        x_ = model.decode(samples)
        print(x_.shape)
        return x_.reshape(-1, 1, 28, 28).data.cpu()


def test(
    model,
    optimizer,
    test_loader,
    i_dim=784,
    draw=False,
    device="cpu",
    name="test",
    log_folder=".",
):
    print_ = defaultdict(list)
    p_z_ = None
    for x_mb, y_mb in test_loader:
        # move x_mb, y_mb to device
        x_mb = x_mb.to(device)
        y_mb = y_mb.to(device)

        # dynamic binarization
        x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).to(
            model.dtype
        )

        _, (q_z, p_z), z, x_mb_ = model(x_mb.reshape(-1, i_dim))
        n = 500
        z = q_z.rsample(torch.Size([n]))  # sample n times
        p_z_ = p_z

        print_["recon loss"].append(
            float(
                nn.BCEWithLogitsLoss(reduction="none")(x_mb_, x_mb.reshape(-1, i_dim))
                .sum(-1)
                .mean()
                .data
            )
        )
        if model.distribution == "vmf":
            print_["KL"].append(
                float(torch.distributions.kl.kl_divergence(q_z, p_z).mean().data)
            )
        elif model.distribution == "wbgauss" or model.distribution == "wbgauss_amb":
            print_["KL"].append(float(kl_divergence(q_z, z, p_z).mean().data))
        else:
            raise NotImplemented

        print_["ELBO"].append(-print_["recon loss"][-1] - print_["KL"][-1])
        print_["LL"].append(float(log_likelihood(model, x_mb, n=500, i_dim=i_dim).data))

    print_ = {k: np.mean(v) for k, v in print_.items()}

    if draw:
        samples = sample(model, p_z_, i_dim=i_dim)
        # draw samples with matplotlib
        save_image(samples, os.path.join(log_folder, f"{model.distribution}.png"))

    return print_


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--distribution",
        type=str,
        default="wbgauss",
        choices=["vmf", "wbgauss", "wbgauss_amb"],
        help="distribution of latent space",
    )

    parser.add_argument(
        "--kl_factor", type=float, default=1.0, help="weight of KL term"
    )
    parser.add_argument(
        "--z_dim", type=int, default=3, help="dimension of latent space"
    )
    parser.add_argument(
        "--h_dim", type=int, default=128, help="dimension of hidden layer"
    )
    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--warmup", type=int, default=100, help="warmup epochs")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--double", action="store_true", help="use double precision")
    parser.add_argument("--name", default="test", help="name of experiment")
    parser.add_argument("--wandb", action="store_true", help="use wandb as logger")
    parser.add_argument(
        "--clip_gradients", type=float, default=0.01, help="gradient clipping"
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="patience for early stopping"
    )

    # Wrapped Beta Gaussian params
    parser.add_argument(
        "--alpha", default=1.1, type=float, help="alpha for BetaGaussian (=2 - beta)"
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    device, dtype = set_default_tensor_type(args.double)
    set_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data", train=True, download=True, transform=transforms.ToTensor()
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator(device=device),
        drop_last=True,
        num_workers=1,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data", train=False, download=True, transform=transforms.ToTensor()
        ),
        batch_size=args.batch_size,
        num_workers=1,
        drop_last=True,
    )

    def run(model, optimizer, name):
        print(f"##### {name} #####", flush=True)
        if args.wandb:
            import wandb

            wblogger = get_wandb_logger(name=args.name, config=args)
            # wandb.watch(model, log="all")
        csv_logger = get_csv_logger(log_folder="vae/logs/", name=args.name, config=args)
        log_folder = csv_logger.log_folder
        step = 0
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=args.warmup,
            final_lr=optimizer.param_groups[0]["lr"],
        )
        best_LL = -np.inf
        n_no_improvement = 0
        for epoch in range(args.n_epochs):
            step, train_metrics = train(
                model,
                optimizer,
                train_loader,
                device=device,
                kl_factor=args.kl_factor,
                scheduler=scheduler,
                epoch=epoch,
                clip_gradients=args.clip_gradients,
            )
            print(f"Epoch {epoch} train metrics", train_metrics, flush=True)
            metrics = test(
                model,
                optimizer,
                test_loader,
                device=device,
                name=args.name,
            )
            metrics["epoch"] = epoch
            csv_logger.log(metrics)
            print(metrics, flush=True)
            print(flush=True)
            if args.wandb:
                wblogger.log(train_metrics)
                wblogger.log(metrics)

            # early stopping
            if metrics["LL"] > best_LL + 1e-3:
                best_LL = metrics["LL"]
                n_no_improvement = 0
            else:
                n_no_improvement += 1
                if n_no_improvement >= args.patience:
                    print(f"Early stopping at epoch {epoch}", flush=True)
                    break

        print("Drawing samples...")
        metrics = test(
            model,
            optimizer,
            test_loader,
            draw=True,
            device=device,
            name=args.name,
            log_folder=log_folder,
        )
        # save model to log_folder
        torch.save(model, os.path.join(log_folder, "model.pt"))

    if args.distribution in {"vmf", "wbgauss", "wbgauss_amb"}:
        args.z_dim = args.z_dim + 1

    # wrapped gaussian VAE
    model = ModelVAE(
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        distribution=args.distribution,
        device=device,
        alpha=args.alpha,  # wrapped beta gaussian param
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    run(model, optimizer, args.distribution)


if __name__ == "__main__":
    main()
