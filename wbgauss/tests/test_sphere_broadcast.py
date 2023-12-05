import torch
import pytest
from wbgauss.scale_operators import DiagScale
from wbgauss.sphere import PoleWBGaussian


@pytest.mark.parametrize("method", ["log_prob", "cross_fy", "wrapped_cross_fy"])
@pytest.mark.parametrize("alpha", [1.01, 1.5, 2])
def test_broadcasting(alpha, method):
    torch.set_anomaly_enabled(True)
    torch.set_default_dtype(torch.float64)
    if alpha == 1:
        pytest.skip("alpha=1 not implemented")
    mean = torch.tensor([[1, 2, 3], [4, 5, 6.]]).unsqueeze(0)  # 1 x 2 x 3
    mean = mean / mean.norm(dim=-1, keepdim=True)
    scale_diag = torch.tensor([[1, 2], [4, 5.]]).unsqueeze(0)
    mbg = PoleWBGaussian(mean, DiagScale(scale_diag), alpha)

    x = torch.tensor([[-1, -2, -3],
                      [1, 1, 2.],
                      [1, 1, 2.],
                      [1, 1, 2.]]).unsqueeze(1) # 4 x 1 x 3
    ce = getattr(mbg, method)(x)  # should return 4 x 2

    ce_test = torch.zeros(4, 2, dtype=torch.double)
    for i, (mean, scale_diag) in enumerate(zip(
            torch.tensor([[1, 2, 3], [4, 5, 6.]]),
            torch.tensor([[1, 2], [4, 5.]])
    )):
        mean = mean / mean.norm(dim=-1, keepdim=True)
        for j, x in enumerate(torch.tensor([[-1, -2, -3],
                                            [1, 1, 2.],
                                            [1, 1, 2.],
                                            [1, 1, 2.]])):
            mbg_test = PoleWBGaussian(mean, DiagScale(scale_diag), alpha)
            print("getattr", getattr(mbg_test, method)(x))
            ce_test[j][i] += getattr(mbg_test, method)(x)
    print("ce/ce_test", ce.shape, ce_test.shape)
    print("ce", ce)
    print("ce_test", ce_test)
    assert torch.allclose(ce, ce_test, atol=1e-2, rtol=1e-2)

def test_broadcasting_rsample():
    mean = torch.tensor([[1, 2, 3], [4, 5, 6.]]).unsqueeze(0)  # 1 x 2 x 3
    mean = mean / mean.norm(dim=-1, keepdim=True)
    scale_diag = torch.tensor([[1, 2], [4, 5.]]).unsqueeze(0)
    alpha = torch.tensor([1.5, 2])
    mbg = PoleWBGaussian(mean, DiagScale(scale_diag), alpha)

    # batch shape, event shape
    assert mbg.rsample().shape == (1, 2, 3)

    # sample shape, batch shape, event shape
    assert mbg.rsample(torch.Size([4])).shape == (4, 1, 2, 3)


    mean = (torch.tensor([[1, 2, 3, 4], [4, 5, 6., 7]])
            .unsqueeze(0).repeat(10, 1, 1))  # 10 x 2 x 4
    mean = mean / mean.norm(dim=-1, keepdim=True)
    scale_diag = (torch.tensor([[1, 2, 3], [4, 5., 6]])
                  .unsqueeze(0).repeat(10, 1, 1))  # 10 x 2 x 3
    alpha = torch.tensor([1.5, 2])  # 2
    mbg = PoleWBGaussian(mean, DiagScale(scale_diag), alpha)

    # batch shape, event shape
    assert mbg.rsample().shape == (10, 2, 4)

    # sample shape, batch shape, event shape
    assert mbg.rsample(torch.Size([4])).shape == (4, 10, 2, 4)
