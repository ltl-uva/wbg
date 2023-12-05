import itertools
import torch
import numpy as np
import pytest
from wbgauss.sphere import PoleWBGaussian, EmbeddedWBGaussian
from wbgauss.spcdist import BetaGaussian
from wbgauss.scale_operators import LogDiagScaleProj, LogDiagScale, DiagScale, EigScale

PI = np.pi
from test_scale_operators import SCALE_CLASSES
from test_beta_gaussian import _setup


def _setup(d, alpha, scale_cls, is_tilde=True, broadcast_batch=False):

    batch_dims = ()
    if broadcast_batch:
        batch_dims = (5, 6)

    torch.set_default_dtype(torch.float64)

    loc = torch.ones(*batch_dims, d)
    loc = loc / torch.norm(loc, dim=-1, keepdim=True)
    if issubclass(scale_cls, LogDiagScaleProj):
        scale_dim = d
    else:
        scale_dim=d-1
    scale, _ = INIT_FUNCS[scale_cls](batch_dims, d=scale_dim)

    if isinstance(scale, LogDiagScaleProj):
        wbg = EmbeddedWBGaussian(loc, scale, alpha)
    else:
        wbg = PoleWBGaussian(loc, scale, alpha)

    bg = wbg.tangent_dist

    torch.manual_seed(42)
    x = bg.rsample((25,))
    return bg, x


def random_init(*size):
    # size: batch_dims + (d,)
    # initialize as 2 ** arange(d) along the last dimention
    x = torch.arange(size[-1], dtype=torch.double)
    x = 2 ** x
    # repeat along batch dims
    for _ in range(len(size) - 1):
        x = x.unsqueeze(0)
    x = x.repeat(size[:-1] + (1,))
    x = x * torch.randn(*size, dtype=torch.double)
    return x


# data creation tests for each scale class
def _make_eigscale(batch_dims, d=5):
    torch.manual_seed(42)
    rank = max(d-1, 1)
    S = random_init(*(batch_dims + (d, rank)))
    S = S @ S.transpose(-2, -1)
    Sop = EigScale(S)
    return Sop, S


def _make_diagscale(batch_dims, d=5):
    torch.manual_seed(42)
    s = random_init(*(batch_dims +(d,)))
    s = .1 + s**2
    Sop = DiagScale(s)
    S = torch.diag_embed(s)
    return Sop, S


def _make_logdiagscale(batch_dims, d=5):
    torch.manual_seed(42)
    s = random_init(*(batch_dims +(d,)))
    Sop = LogDiagScale(s)
    S = torch.diag_embed(torch.exp(s))
    return Sop, S


def _make_logdiagscaleproj(batch_dims, d=5):
    torch.manual_seed(42)
    s = random_init(*(batch_dims +(d,)))
    x = torch.randn(*(batch_dims + (d,)))
    x = torch.nn.functional.normalize(x, dim=-1)
    Sop = LogDiagScaleProj(s, x)
    S = torch.diag_embed(torch.exp(s))
    for ix in itertools.product(*(range(m) for m in batch_dims)):
        P = torch.eye(d) - torch.outer(x[ix], x[ix])
        S[ix] = P @ S[ix] @ P
    return Sop, S


INIT_FUNCS = {
    EigScale: _make_eigscale,
    DiagScale: _make_diagscale,
    LogDiagScale: _make_logdiagscale,
    LogDiagScaleProj: _make_logdiagscaleproj
}


@pytest.mark.parametrize("scale_cls", SCALE_CLASSES)
@pytest.mark.parametrize("alpha", [1.1])
@pytest.mark.parametrize("d", [3])
@pytest.mark.parametrize("tilde", [True, False])
class TestBetaGaussian:
    def test_radius_constraint(self, scale_cls, alpha, d, tilde):
        if not tilde:
            pytest.skip("only for tilde=True")
        bg, x =  _setup(d, alpha, scale_cls, tilde)
        print("max_norm:",  torch.norm(x, dim=-1).max())
        assert torch.norm(x, dim=-1).max() < np.pi

    def test_covariance(self, scale_cls, alpha, d, tilde):
        TOL = 1e-6
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(42)

        batch_dims = ()
        if issubclass(scale_cls, LogDiagScaleProj):
            scale_dim = d
        else:
            scale_dim=d-1
        scale, _ = INIT_FUNCS[scale_cls](batch_dims, d=scale_dim)

        bg = BetaGaussian(loc=torch.zeros(scale_dim), scale=scale, alpha=alpha,
                          is_tilde=tilde)

        true_cov = bg.cov
        n_samples = 100000
        x = bg.rsample((n_samples,))  # n_samples x ... x d
        mean = x.mean(dim=0)

        emp_cov = ((x - mean).T @ (x - mean)) / n_samples

        assert torch.allclose(emp_cov, true_cov, rtol=1e-2, atol=1e-2)
