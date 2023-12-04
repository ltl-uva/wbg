import random
import timeit
import warnings
from typing import Optional

import numpy as np
import torch


class Timer:
    def __init__(self):
        self.start_ = None
        self.end_ = None

    def start(self):
        if torch.cuda.is_available():
            self.start_ = torch.cuda.Event(enable_timing=True)
            self.end_ = torch.cuda.Event(enable_timing=True)
            self.start_.record()
        else:
            # start cpu timer
            self.start_ = timeit.default_timer()

    def end(self):
        if torch.cuda.is_available():
            self.end_.record()
            torch.cuda.synchronize()
            # return time in seconds
            return self.start_.elapsed_time(self.end_) / 1000
        else:
            return timeit.default_timer() - self.start_


def get_wandb_logger(**kwargs):
    import wandb

    class WandbLogger:
        def __init__(self, project="mnist-wbg-v2", **kwargs):
            wandb.init(project=project, **kwargs)

        def log(self, logs):
            wandb.log(logs)

    return WandbLogger(**kwargs)


def get_csv_logger(log_folder, name, config):
    import csv
    import os

    class CSVLogger:
        def __init__(self, log_folder, name, config):
            self.log_folder = f"{log_folder}/{name}/"
            self.log_file = os.path.join(self.log_folder, f"log.csv")
            if not os.path.exists(self.log_folder):
                os.makedirs(self.log_folder)

            # save config
            with open(os.path.join(self.log_folder, "config.txt"), "w") as f:
                f.write(str(config))

            with open(self.log_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "LL", "recon_loss", "kl_loss", "ELBO"])

        def log(self, logs):
            with open(self.log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        logs["epoch"],
                        logs["LL"],
                        logs["recon loss"],
                        logs["KL"],
                        logs["ELBO"],
                    ]
                )

    return CSVLogger(log_folder, name, config)


def set_default_tensor_type(double=True):
    if torch.cuda.is_available():
        if double:
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
            dtype = torch.float64
        else:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            dtype = torch.float32
        warnings.warn("Using GPU: cuda device")
        device = torch.device("cuda")
        torch.multiprocessing.set_start_method("spawn")
    else:
        if double:
            torch.set_default_tensor_type(torch.DoubleTensor)
            dtype = torch.float64
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
            dtype = torch.float32
        device = torch.device("cpu")
        warnings.warn("Using CPU")
    return device, dtype


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class VAEMonitor:
    def __init__(self):
        self.z = None
        self.dist = None
        self.n = 0

    @torch.no_grad()
    def update(self, z_mean: torch.Tensor, z: torch.Tensor):
        bs, _ = z.shape
        if self.z is None:
            self.z = z.sum(0)
            self.dist = torch.std(z - z_mean, dim=-1).sum(0)
            self.n = bs
        else:
            self.z += z.sum(0)
            self.dist += torch.std(z - z_mean, dim=-1).sum(0)
            self.n += bs

    @torch.no_grad()
    def mean_z(self):
        mean_z = self.z / self.n if self.z is not None else None
        return mean_z

    @torch.no_grad()
    def mean_dist(self):
        mean_dist = self.dist / self.n if self.dist is not None else None
        return mean_dist


# warmup scheduler for epochs
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, final_lr=0.001):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.final_lr = final_lr
        self.current_lr = 0

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            self.current_lr = self.final_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.current_lr
        else:
            self.current_lr = self.final_lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.current_lr
        return self.current_lr

    def get_lr(self):
        return self.current_lr

    def get_final_lr(self):
        return self.final_lr
