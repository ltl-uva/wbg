""" test factorized scale operator

testing strategy:
1. the *reconstruct test, separately written for each Scale class,
   ensure the semantics of the representation (cannot be generalized),
   and check that op.as_tensor() is correct.

2. the tests for trace, det, matvec etc are generalized in terms of the
   op.as_tensor(), so they depend on 1.

"""

import pytest
import torch
import itertools
from wbgauss.scale_operators import (
    EigScale,
    DiagScale,
    LogDiagScale,
    LogDiagScaleProj
)

torch.set_default_dtype(torch.float64)

BATCH_DIMS = [(), (1,), (3,), (2, 3)]


def _log_pdet(X: torch.Tensor):
    s, _ = torch.linalg.eigh(X)
    log_s = torch.log(s)
    log_s[s < 1e-10] = 0
    return torch.sum(log_s, dim=-1)


# data creation tests for each scale class
def _make_eigscale(batch_dims, d=5):
    torch.manual_seed(42)
    rank = max(d-1, 1)
    S = torch.randn(*(batch_dims + (d, rank)))
    S = S @ S.transpose(-2, -1)
    Sop = EigScale(S)
    return Sop, S


def _make_diagscale(batch_dims, d=5):
    torch.manual_seed(42)
    s = torch.randn(*(batch_dims +(d,)))
    s = .1 + s**2
    Sop = DiagScale(s)
    S = torch.diag_embed(s)
    return Sop, S


def _make_logdiagscale(batch_dims, d=5):
    torch.manual_seed(42)
    s = torch.randn(*(batch_dims +(d,)))
    Sop = LogDiagScale(s)
    S = torch.diag_embed(torch.exp(s))
    return Sop, S


def _make_logdiagscaleproj(batch_dims, d=5):
    torch.manual_seed(42)
    s = torch.randn(*(batch_dims +(d,)))
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


SCALE_CLASSES = INIT_FUNCS.keys()


# Reconstruction tests

@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("Cls", SCALE_CLASSES)
def test_scale_reconstruct(batch_dims, Cls):
    Sop, S = INIT_FUNCS[Cls](batch_dims)
    S_ = Sop.as_tensor()
    assert torch.allclose(Sop.as_tensor(), S)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("Cls", SCALE_CLASSES)
def test_scale_trace(batch_dims, Cls):
    Sop, _ = INIT_FUNCS[Cls](batch_dims)
    S = Sop.as_tensor()

    # need to use vmap to batch torch.trace
    d = S.shape[-1]
    S = S.reshape((-1, d, d))
    traces = torch.vmap(torch.trace)(S).reshape(batch_dims)
    assert torch.allclose(traces, Sop.trace)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("Cls", SCALE_CLASSES)
def test_scale_log_det(batch_dims, Cls):
    Sop, _ = INIT_FUNCS[Cls](batch_dims)
    S = Sop.as_tensor()
    # logdets = torch.linalg.slogdet(S).logabsdet
    logdets = _log_pdet(S)
    assert torch.allclose(logdets, Sop.log_det)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("Cls", SCALE_CLASSES)
def test_scale_inv_quad(batch_dims, Cls):

    Sop, _ = INIT_FUNCS[Cls](batch_dims)
    d = Sop.shape[-1]
    x = torch.randn(*((7,) + batch_dims + (d,)), dtype=torch.float64)

    S = Sop.as_tensor()
    xSinvx_solve = [
        (torch.linalg.lstsq(S, x_)[0] * x_).sum(-1)
        for x_ in x]

    xSinvx_op = Sop.inv_quad(x)

    for i in range(x.shape[0]):
        assert torch.allclose(xSinvx_solve[i], xSinvx_op[i])


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("Cls", SCALE_CLASSES)
def test_scale_sqrt_matvec(batch_dims, Cls):
    Sop, _ = INIT_FUNCS[Cls](batch_dims)
    d = Sop.shape[-1]

    S = Sop.as_tensor()

    # XXX this only holds in the full rank case.
    # There are many possible square roots.
    # I believe all of them should work here.
    # So, we check that y'y = x'Sx.

    # quad = torch.einsum("...ij,d...i,d...j->d...", S, x, x)
    # y = Sop.sqrt_matvec(x)
    # yssq = y.square().sum(-1)
    # assert torch.allclose(yssq, quad)

    # For low rank, we materialize the sqrt.

    # first, create a "batched" Identity matrix
    target_ones = (d,) + tuple(1 for _ in batch_dims) + (d,)
    target_shape = (d,) + batch_dims + (d,)
    Id = torch.eye(d, d).reshape(target_ones).expand(target_shape)

    # apply L@Id and permute axes accordingly
    permutation = tuple(1+x for x in range(len(batch_dims))) + (-1, 0)
    L = Sop.sqrt_matvec(Id).permute(permutation)

    assert torch.allclose(L @ L.transpose(-2, -1), S)
