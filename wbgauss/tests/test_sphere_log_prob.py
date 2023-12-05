import torch
import numpy as np
import pytest
from wbgauss.sphere import PoleWBGaussian
from wbgauss.scale_operators import LogDiagScale

# q: uniform on the sphere, p: wrapped beta gaussian
# 1 = int_S p(u) du = int_S p(u) * q(u) / q(u) du = E_q [p(u) / q(u)]

try:
    from power_spherical import PowerSpherical
    PS_INSTALLED = True
except ImportError:
    PS_INSTALLED = False

@pytest.mark.parametrize("d", [3, 4])
@pytest.mark.parametrize("alpha", [1.01, 1.25, 2.0, 3.0])
def test_integral_pole_wbg(d, alpha):
    if not PS_INSTALLED:
        pytest.skip("power_spherical not installed; "
                    "cannot test the log_prob calculation.")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)
    loc = torch.randn(d)
    loc = loc / torch.norm(loc)
    scale = torch.randn(d-1)
    Sop = LogDiagScale(scale)
    wbg = PoleWBGaussian(loc, Sop, alpha)

    dist = PowerSpherical(loc=loc, scale = torch.tensor([1]))
    u = dist.rsample(torch.Size([1000000]))
    res = ((wbg.log_prob(u) - dist.log_prob(u)).exp()).mean()
    assert torch.allclose(res, torch.ones_like(res), atol=1e-2), res
