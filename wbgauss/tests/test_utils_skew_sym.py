import math
import torch
import pytest

from wbgauss.utils.skew_sym import log_jacobian_from_thetas, skew_symmetric_exp_and_angles


n_trials = 100

def _so_n_exp(n):
    torch.manual_seed(42)

    for _ in range(n_trials):
        L = torch.randn(n, n, dtype=torch.double)
        A = L - L.T
        expA, _ = skew_symmetric_exp_and_angles(A)
        true_expA = torch.matrix_exp(A)

        assert torch.allclose(expA, true_expA)


@pytest.mark.parametrize('n', [6, 5])
def test_so_n_exp(n):
    _so_n_exp(n=n)


def _so_n_thetas(n, average_pairs):
    torch.manual_seed(42)
    for _ in range(n_trials):
        L = torch.randn(n, n, dtype=torch.double)
        A = L - L.T
        _, thetas = skew_symmetric_exp_and_angles(A, average_pairs=average_pairs)

        # check shape first
        assert thetas.shape == (math.ceil(n/2),)

        # check values
        true_thetas = torch.linalg.eigvals(A).imag[::2].abs()
        # annoyingly, there's no sort order for complex eigvals so we need to
        # sort ourselves
        true_thetas, _ = torch.sort(true_thetas, descending=True)
        assert torch.allclose(true_thetas, thetas)


@pytest.mark.parametrize('n', [6, 5])
def test_so_n_thetas(n):
    _so_n_thetas(n=n, average_pairs=True)


def test_finite_gradient():

    n = 4

    X = torch.randn(n, n, dtype=torch.double)
    Y = torch.randn(n, n, dtype=torch.double)

    torch.manual_seed(42)
    for _ in range(n_trials):

        L = torch.randn(n, n, dtype=torch.double, requires_grad=True)
        A = L - L.T
        expA, thetas = skew_symmetric_exp_and_angles(A, average_pairs=True)

        loss = (Y - expA @ X).square().sum() + thetas.sum()

        loss.backward()
        assert torch.all(torch.isfinite(L.grad))
        # print(torch.all(torch.isfinite(L.grad)))


@pytest.mark.parametrize('n', (6, 5))
def test_jacdet_numeric(n):

    torch.manual_seed(42)
    torch.set_default_dtype(torch.double)

    for _ in range(n_trials):

        L = torch.randn(n, n)
        A = L - L.T
        a = A.ravel()

        # doing this raveling to ensure jacobian is a matrix.
        # if we write jacobian(matrix_exp, A), we get [n,n,n,n]
        # and probably we need to ravel along (01) and (23), but what if i am
        # misinterpreting the axes? This way I'm sure.
        def f(a):
            A = a.reshape(n, n)
            eA = torch.linalg.matrix_exp(A)
            ea = eA.ravel()
            return ea

        J = torch.autograd.functional.jacobian(f, a)
        sld = torch.linalg.slogdet(J)
        expected = sld.logabsdet

        _, thetas = skew_symmetric_exp_and_angles(A)
        obtained = log_jacobian_from_thetas(thetas, n)
        assert torch.allclose(expected, obtained)


@pytest.mark.parametrize('n', (6, 5))
def test_jacdet_finite_grad(n):
    torch.manual_seed(42)
    torch.set_default_dtype(torch.double)

    for _ in range(n_trials):

        L = torch.randn(n, n, requires_grad=True)
        A = L - L.T
        _, thetas = skew_symmetric_exp_and_angles(A)
        logjacdet = log_jacobian_from_thetas(thetas, n)
        logjacdet.backward()
        assert torch.all(torch.isfinite(L.grad))
