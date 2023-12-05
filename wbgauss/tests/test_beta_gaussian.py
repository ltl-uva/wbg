import torch
import pytest
from wbgauss.spcdist import BetaGaussian
from wbgauss.scale_operators import LogDiagScaleProj

try:
    from spcdist.torch import MultivariateBetaGaussian
    SPCDIST_INSTALLED = True
except ImportError:
    SPCDIST_INSTALLED = False

from test_scale_operators import INIT_FUNCS, SCALE_CLASSES


def _setup(d, alpha, scale_cls, is_tilde=False, broadcast_batch=False):
    if not SPCDIST_INSTALLED:
        pytest.skip("spcdist not installed; cannot compare against it.")
    if d == 1 and scale_cls == LogDiagScaleProj:
        pytest.skip("LogDiagScaleProj not implemented for d=1")

    batch_dims = ()
    if broadcast_batch:
        batch_dims = (5, 6)

    torch.set_default_dtype(torch.float64)

    loc = torch.ones(*batch_dims, d)
    scale, _ = INIT_FUNCS[scale_cls](batch_dims, d=d)

    scale_tensor = scale.as_tensor()

    if is_tilde:
        # go from sigma_tilde to sigma
        log_det = scale.log_det
        scale_tensor = (scale_tensor *
                        (log_det[..., None, None] * ((alpha - 1) / 2)).exp())

    bg = BetaGaussian(loc, scale, alpha, is_tilde=is_tilde)
    bg_orig = MultivariateBetaGaussian(loc, scale_tensor, alpha)

    torch.manual_seed(42)
    x = bg_orig.rsample((25,))
    x[10:] *= 5
    x[20:] *= 5  # make some points that will certainly have zero prob

    if broadcast_batch:
        while x.dim() > 1:
            x = x[0]  # x: (d, )
        x = torch.unsqueeze(x, 0).unsqueeze(0).repeat((3, 4, 1))  # introduce new dims
        x[0, ...] *= 0.5
        x[2, ...] *= 2
        # x should be unsqueezed to match the batch dims of bg

    return bg, bg_orig, x


@pytest.mark.parametrize("scale_cls", SCALE_CLASSES)
@pytest.mark.parametrize("alpha", [1.01, 2])
@pytest.mark.parametrize("d", [1, 2, 3, 10, 20])
@pytest.mark.parametrize("tilde", (False, True))
@pytest.mark.parametrize("broadcast_batch", (False, True))
class TestBetaGaussian:
    def test_tau(self, scale_cls, alpha, d, tilde, broadcast_batch):
        bg, bg_orig, _ =  _setup(d, alpha, scale_cls, tilde)
        assert torch.allclose(bg._tau, bg_orig._tau)  # impl via log_negtau

    def test_tsallis_entropy(self, scale_cls, alpha, d, tilde, broadcast_batch):
        bg, bg_orig, _ = _setup(d, alpha, scale_cls, tilde)
        assert torch.allclose(bg.tsallis_entropy, bg_orig.tsallis_entropy)

    def test_log_prob(self, scale_cls, alpha, d, tilde, broadcast_batch):
        bg, bg_orig, x = _setup(d, alpha, scale_cls, tilde, broadcast_batch=broadcast_batch)
        assert torch.allclose(bg.log_prob(x, broadcast_batch=broadcast_batch), bg_orig.log_prob(x, broadcast_batch=broadcast_batch))

    def test_cross_fy(self, scale_cls, alpha, d, tilde, broadcast_batch):
        bg, bg_orig, x = _setup(d, alpha, scale_cls, tilde, broadcast_batch=broadcast_batch)
        c = bg.scale.rank / (2 * alpha)  # equal up to a constant
        assert torch.allclose(bg.cross_fy(x, broadcast_batch=broadcast_batch), bg_orig.cross_fy(x, broadcast_batch=broadcast_batch) - c)

    def test_pdf(self, scale_cls, alpha, d, tilde, broadcast_batch):
        bg, bg_orig, x = _setup(d, alpha, scale_cls, tilde, broadcast_batch=broadcast_batch)
        assert torch.allclose(bg.pdf(x, broadcast_batch=broadcast_batch), bg_orig.pdf(x, broadcast_batch=broadcast_batch))
