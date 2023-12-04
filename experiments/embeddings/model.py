from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from geoopt.manifolds.sphere import SphereExact
from geoopt.tensor import ManifoldParameter
from vocab import Vocab

from wbgauss.sphere import (EmbeddedWBGaussian, LogDiagScale, LogDiagScaleProj,
                            PoleWBGaussian, WBGBase)
from wbgauss.utils.sphere import sphere_distance

SPARSE = False


class WordEmbedding(nn.Embedding):
    """General class for word embeddings, can be used for location, and scale embeddings"""

    def __init__(self, vocab: Vocab, emb_dim, **kwargs):
        super().__init__(len(vocab), emb_dim, **kwargs)
        self.embedding_dim = emb_dim
        self.vocab = vocab

    def init_from_pretrained(self, pretrained_embeddings: torch.Tensor, word2id: dict):
        # pretrained_embeddings: (vocab_size, embedding_dim)
        # word2id: dict, word to id mapping for pretrained embeddings (not necessarily the same as self.vocab)

        if self.embedding_dim < pretrained_embeddings.shape[1]:
            print("PCA: reducing dimensionality of pretrained embeddings")
            from sklearn.decomposition import PCA

            pca = PCA(n_components=self.embedding_dim)
            embeddings = pca.fit_transform(pretrained_embeddings.numpy())
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = torch.from_numpy(embeddings)
        else:
            embeddings = pretrained_embeddings

        n_suceess = 0
        n_fail = 0
        for word in self.vocab.word2idx:
            if word not in word2id:
                n_fail += 1
                continue
            n_suceess += 1
            pretraind_id = word2id[word]
            idx = self.vocab.word2idx[word]
            self.weight.data[idx].copy_(embeddings[pretraind_id])
        print(
            "Initialized {} words from pretrained embeddings, {} words not found".format(
                n_suceess, n_fail
            )
        )

    def view_embeddings(self, words: List[str]) -> torch.Tensor:
        # view for selected words
        idxs = [self.vocab.word2idx[word] for word in words]
        idxs = torch.tensor(idxs, dtype=torch.long, device=self.weight.device)
        return self.weight[idxs]

    def view_gradients(self, words: List[str]) -> torch.Tensor:
        idxs = [self.vocab.word2idx[word] for word in words]
        idxs = torch.tensor(idxs, dtype=torch.long, device=self.weight.device)
        return self.weight.grad[idxs]


class LocationEmbedding(WordEmbedding):
    def __init__(self, vocab_size, emb_dim, **kwargs):
        """Embedding class to represent locations on the sphere"""
        super().__init__(vocab_size, emb_dim, **kwargs)
        self.init()

    def init(self):
        # random vectors on the sphere
        nn.init.normal_(
            self.weight
        )  # + 10 * torch.ones_like(self.weight)  # initialize somewhere on the sphere
        self.weight.data = self.weight.data / self.weight.data.norm(dim=1, keepdim=True)
        # make self.weight ManifoldParameter
        self.weight = ManifoldParameter(
            self.weight.data.clone(), manifold=SphereExact(), requires_grad=True
        )


class LogScaleEmbedding(WordEmbedding):
    def __init__(self, vocab, emb_dim, **kwargs):
        """Embedding class to represent the log-scale parameters of probabilistic distributions"""
        self.scale_init = kwargs.get("scale_init", -1.0)
        if "scale_init" in kwargs:
            del kwargs["scale_init"]
        super().__init__(vocab, emb_dim, **kwargs)

        self.init()

    def init(self):
        # init log scale parameters to be small random values around 0
        nn.init.normal_(self.weight, mean=self.scale_init, std=1e-3)

    def norm(self, word) -> float:
        # return the norm of the scale parameter for a given word
        idx = self.vocab.word2idx[word]
        scale = torch.exp(self.weight[idx])
        scale_norm = torch.norm(scale).item()
        return scale_norm


class SourceDistribution(object):
    def __init__(self):
        raise NotImplementedError

    def get_similarity(self, target_embeddings: torch.Tensor) -> torch.Tensor:
        """Returns the similarity between the source and target embeddings"""
        raise NotImplementedError


class Similarity(nn.Module):
    def __init__(self):
        """Similarity function between words
        accepts the ids of the source and target words
        returns the similarity score between them
        """
        super().__init__()

        self.src_emb: LocationEmbedding = None
        self.tgt_emb: LocationEmbedding = None

    def get_source(self, source_ids: torch.Tensor) -> SourceDistribution:
        raise NotImplementedError

    def get_target(self, target_ids, tie_embeddings=False) -> torch.Tensor:
        """Returns target fixed embeddings by target word ids
        Optionally:
           - tie_embeddings: if true use source embeddings instead of target embeddings
           - fix_target: if true fix the location of target embedding (not trained)
        """
        target_emb = self.tgt_emb(target_ids)
        if tie_embeddings:
            print("Using only source embeddings")
            target_emb = self.src_emb(target_ids)

        return target_emb

    def forward(self, source_ids, target_ids, tie_embeddings=False):
        # source_ids: (batch_size)
        # target_ids: (batch_size, N) or (batch_size, N, K)
        # return: (batch_size) or (batch_size, N)
        source_emb = self.get_source(source_ids)
        target_emb = self.get_target(target_ids, tie_embeddings=tie_embeddings)

        if target_emb.dim() == 2:
            return source_emb.get_similarity(target_emb)
        elif target_emb.dim() == 3:
            return source_emb.get_similarity(target_emb.transpose(0, 1)).transpose(0, 1)
        else:
            raise ValueError("target_emb must be 2 or 3 dimensional")


WBG_TO_POINT_LOSS = ["kl_fy"]
WBG_TYPES = [
    "pole",  #  : (PoleWBGaussian, LogDiagScale),
    "embedded",  #  : (EmbeddedWBGaussian, LogDiagScaleProj)
]


class WrappedBetaGaussianWrapper(nn.Module):
    def __init__(self, wbg_type, wbg_loss):
        """Wrapper for wrapped beta gaussian
        supports two types of wrapped beta gaussian:
        - pole: pole wrapped beta gaussian
        - embedded: embedded wrapped beta gaussian
        supports two types of loss:
        - cross_fy: cross entropy loss
        - wrapped_cross_fy: wrapped cross entropy loss
        """
        super().__init__()
        self.wbg_type = wbg_type
        self.wbg_method = wbg_loss
        assert wbg_type in WBG_TYPES, f"Unknown WBG type {wbg_type}"
        assert wbg_loss in WBG_TO_POINT_LOSS, f"Unknown WBG loss {wbg_loss}"

        self.wbg_type = wbg_type

    def get_wbg(self, sources_emb, log_sources_scales, alpha):
        if self.wbg_type == "pole":
            return PoleWBGaussian(
                sources_emb, LogDiagScale(log_sources_scales, rcond=1e-50), alpha
            )
        elif self.wbg_type == "embedded":
            return EmbeddedWBGaussian(
                sources_emb,
                LogDiagScaleProj(log_sources_scales, sources_emb, rcond=1e-50),
                alpha,
            )
        else:
            raise ValueError(f"Unknown WBG type {self.wbg_type}")

    def get_similarity(self, wbg, target, **kwargs):
        return -getattr(wbg, self.wbg_method)(target)

    def get_distributional_similarity(self, wbg, target, **kwargs):
        return -getattr(wbg, self.wbg_method)(target)

    def dim(self):
        return len(self.wbg.loc.shape)


class WBGKLSimilarity(Similarity):
    def __init__(
        self,
        vocab: Vocab,
        embedding_dim,
        alpha=1.1,
        wbg_type=None,
        wbg_loss=None,
        fix_source=False,
        fix_target=True,
        scale_init=None,
        **kwargs,
    ):
        """Distribution-to-distribution similarity function between words
        based on wrapped beta gaussian distribution,
        cross fenchel young KL loss is used to compute similarity.
        Accepts the ids of the source and target words

        Parameters
        ----------
        vocab : Vocab object
        embedding_dim : int
            dimension of the embedding space
        alpha : float
            alpha parameter of wrapped beta gaussian (=2-beta)
        wbg_type : str, one of ["pole", "embedded"]
            type of wrapped beta gaussian
        wbg_loss : str, one of ["cross_fy", "wrapped_cross_fy"]
            type of loss (cross fenchel young / wrapped cross fenchel young))
        """
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.alpha = alpha

        scale_dim = embedding_dim - 1
        if wbg_type == "embedded":
            scale_dim = embedding_dim

        self._src_emb_init = LocationEmbedding(vocab, embedding_dim, sparse=SPARSE)
        self.src_emb = LocationEmbedding(vocab, embedding_dim, sparse=SPARSE)
        self.scale_src = LogScaleEmbedding(
            vocab, scale_dim, sparse=SPARSE, scale_init=scale_init
        )

        self.wbg_wrapper = WrappedBetaGaussianWrapper(wbg_type, wbg_loss)

    def get_source(self, source_ids) -> WBGBase:
        """Returns source probability distribtutiuon by source word ids
        Optionally:
          - fix_source: if true fix the location of source embedding (train only variance)
        """
        source_emb = self.src_emb(source_ids)
        assert torch.allclose(
            source_emb.norm(dim=-1), torch.ones_like(source_emb.norm(dim=-1)), atol=1e-3
        )

        scale = self.scale_src(source_ids)
        wbg = self.wbg_wrapper.get_wbg(source_emb, scale, self.alpha)
        return wbg

    def get_target(self, target_ids, tie_embeddings=True) -> WBGBase:
        return self.get_source(target_ids)  # always use source embeddings

    def forward(self, source_ids, target_ids, tie_embeddings=False):
        # source_ids: (batch_size)
        # target_ids: (batch_size, N) or (batch_size, N, K)
        # return: (batch_size) or (batch_size, N)
        source_emb: WrappedBetaGaussianWrapper = self.get_source(source_ids)
        target_emb: WrappedBetaGaussianWrapper = self.get_target(
            target_ids, tie_embeddings=tie_embeddings
        )

        if target_emb.dim() == 2:
            return source_emb.get_distributional_similarity(target_emb)
        elif target_emb.dim() == 3:
            sim = source_emb.get_distributional_similarity(target_emb)
            return sim
        else:
            raise ValueError("target_emb must be 2 or 3 dimensional")

    def get_covariance(self, word):
        # return covariance matrix for the word
        # using self.wbg.tangent_dist.cov
        idx = self.vocab.word2idx[word]
        return self.wbg_wrapper.wbg.tangent_dist.cov.shape

    def save(self, path):
        torch.save(self, path)

    def dist_from_init(self):
        # distance from initialization
        return sphere_distance(
            self.src_emb.weight.data, self._src_emb_init.weight.data
        ).mean()


SIMS = {
    "kl_fy": WBGKLSimilarity,
}


class ProbabilisticEmbeddings(nn.Module):
    def __init__(self, vocab, embedding_dim, similarity: Similarity):
        super().__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.similarity = similarity

    def save(self, path):
        print("saving model to", path)
        torch.save(self, path)

    @property
    def device(self):
        return next(self.parameters()).device
