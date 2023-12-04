import os

import geoopt
import torch
from dataloader import HypernymsDataset
from model import SIMS, WBG_TYPES, ProbabilisticEmbeddings
from torch.utils.data import DataLoader
from tqdm import tqdm
from vocab import Vocab

Logger = None
DEBUG = False


class Trainer:
    def __init__(
        self,
        model: ProbabilisticEmbeddings,
        vocab: Vocab,
        optimizer,
        name="hypern",
        logger: Logger = None,
        save_dir="embeddings/save_dir",
        clip_gradients=None,
    ):
        self.model = model
        self.vocab = vocab
        self.optimizer = optimizer
        self.name = name
        self.logger = logger
        self.save_dir = save_dir
        self.clip_gradients = clip_gradients

    def train(self, dataloader: DataLoader, epochs: int = 1, debug=False):
        save_dir = self.save_dir
        self.model.train()

        cur_batch = 0
        for epoch in range(epochs):
            for i, (child, parent, negs) in enumerate(
                tqdm(
                    dataloader,
                    desc=f"Epoch {epoch}",
                    total=len(dataloader),
                    mininterval=10,
                )
            ):
                parent = parent.to(self.model.device)
                child = child.to(self.model.device)
                negs = negs.to(self.model.device)

                self.optimizer.zero_grad()

                for c, p, n in zip(child, parent, negs):
                    print("child", self.vocab.idx2word[c.item()], end="\t")
                    print("parent", self.vocab.idx2word[p.item()], end="\t")
                    print("negs", [self.vocab.idx2word[ni.item()] for ni in n])

                src_dist = self.model.similarity.get_source(child)
                tgt_dist = self.model.similarity.get_target(parent)

                negs_dist = self.model.similarity.get_target(
                    negs.transpose(0, 1).unsqueeze(1)
                )  # 1000 x negs x batch_size x dim
                KL_tgt = src_dist.kl_fy(tgt_dist, 100)
                KL_neg = src_dist.kl_fy(negs_dist, 100)

                regularization = src_dist.tangent_dist.scale.trace_inv.mean()
                NEGATIVE_WEIGHT = 0.1
                loss = (
                    KL_tgt - NEGATIVE_WEIGHT * torch.logsumexp(KL_neg, dim=0)
                ).mean()

                print("loss:", loss.item())
                print("regularization:", regularization.item())
                print("KL_tgt:", KL_tgt.mean().item())
                print("KL_neg:", KL_neg.mean().item())

                loss.backward()
                self.optimizer.step()

                if i % 100 == 0:
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

                cur_batch += 1

            with torch.no_grad():
                if epoch % 10 == 0:
                    if not debug:
                        print("saving model")
                        self.model.save(f"{save_dir}/{self.name}_{epoch}.pth")
        self.model.eval()
        if not debug:
            self.model.save(f"{save_dir}/{self.name}_last.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Dataset arguments
    parser.add_argument("--file", type=str, default="data/simple_hypernym.csv")
    parser.add_argument("--n_negatives", type=int, default=3)
    parser.add_argument("--debug", action="store_true")

    # Loss arguments
    parser.add_argument("--sim", type=str, default="kl_fy", choices=SIMS.keys())
    parser.add_argument(
        "--margin", type=float, default=1.0
    )  # margin for maxmargin loss

    # Model arguments
    parser.add_argument("-s", "--embedding_dim", type=int, default=3)
    parser.add_argument(
        "--scale_init",
        type=float,
        default=-1.0,
        help="mean of log-scale initialization",
    )

    # Optimizer arguments
    parser.add_argument("--batch_size", type=int, default=11)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=30)

    # Other arguments
    parser.add_argument("--name", type=str, default="toy_hypernyms")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--clip_gradients", type=float, default=10000.0, help="gradient clipping"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.3, help="alpha for wrapped Gaussian"
    )
    parser.add_argument(
        "--wbg_type",
        choices=WBG_TYPES,
        default="pole",
        help="type of wrapped beta gaussian parametrization",
    )
    parser.add_argument("--wbg_loss", default="kl_fy")

    args = parser.parse_args()
    torch.set_default_tensor_type(torch.DoubleTensor)

    args.name += f".{args.wbg_type}"

    dataset = HypernymsDataset(
        args.file, debug=args.debug, n_negatives=args.n_negatives
    )
    print("Dataset size:", len(dataset))
    n_workers = min(2, torch.get_num_threads())
    print(f"Using {n_workers} workers")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=n_workers
    )

    vocab = dataset.vocab

    for i in range(len(dataset)):
        print(dataset[i])

    sim = SIMS[args.sim](
        vocab,
        args.embedding_dim,
        alpha=args.alpha,
        wbg_type=args.wbg_type,
        wbg_loss=args.wbg_loss,
        scale_init=args.scale_init,
    )

    model = ProbabilisticEmbeddings(vocab, args.embedding_dim, similarity=sim)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.lr)
    wandb_logger = None
    save_dir = "save_hypernyms_exp"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    trainer = Trainer(
        model,
        vocab,
        optimizer,
        name=args.name,
        save_dir=save_dir,
        clip_gradients=args.clip_gradients,
    )
    trainer.train(dataloader, epochs=args.epochs, debug=args.debug)
