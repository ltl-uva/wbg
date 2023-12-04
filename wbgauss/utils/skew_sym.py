import math

import torch

LOG2 = math.log(2)
LOGPI = math.log(math.pi)


def log_sinc(x):
    # return torch.log(torch.sinc(x / math.pi))
    xnrm = x / math.pi
    return -torch.lgamma(1 + xnrm) - torch.lgamma(1 - xnrm)


def skew_symmetric_exp_and_angles(A, average_pairs=False, eps=1e-8):
    """Compute exp(A) and the theta_j of skew-symmetric A.

    If A is a n-by-n skew-symmetric matrix, its spectrum is

        { \\pm i*theta_j } for j=1,...,floor(n/2).

    Instead of computing a costly complex eigendecomposition,
    we use an eigendecomposition of A^2 which is mappable to the
    Youla canonical form decomposition of A, or to an SVD of A.

    Parameters
    ----------

    A: tensor, shape=(n,n)
        input skew-symmetric matrix.

    average_pairs: bool, default=False:
        whether to account for eigendecomposition numerical instability
        by averaging the supposed equal eigenvalues.

    Returns
    -------

    expA: tensor, shape=(n,n)
        matrix exponential exp(A)

    thetas: tensor, shape=(ceil(n/2),)
        tensor of thetas. In case A is odd, one of them will be an unpaired
        zero. If any other eigenvalues of A are zero, they will come in pairs.
    """

    ## NOTE to self: eigendecomposition aprpoach might be wrong up to a sign so
    ## so lead to transpose approach. Using SVD hopefully will be fine
    ## Todo figure out why.

    n = A.shape[-1]
    U, S, Vh = torch.linalg.svd(A)

    # clip tiny singular values:
    if eps > 0:
        S = torch.where(S.abs() < eps, 0, S)

    # XXX
    #
    # the following is correct, but gradients don't work due to identical sv.
    # one fix is to implement custom backward pass just for this expA.
    # either using the standard recipe (requiring computing an exp)
    # or (cheaper, but needs pen&paper work) using U, S, Vh.

    # symm_part = U @ torch.diag_embed(torch.cos(S)) @ U.T
    # skew_part = U @ torch.diag_embed(torch.sin(S)) @  Vh
    # expA = symm_part + skew_part

    # For now, being lazy:

    expA = torch.matrix_exp(A)

    if average_pairs:
        thetas_odd = S[:-1:2]
        thetas_even = S[1::2]
        thetas = 0.5 * (thetas_odd + thetas_even)

        if n % 2 == 1:
            thetas = torch.nn.functional.pad(thetas, (0, 1), mode="constant", value=0)
    else:
        thetas = S[::2]  # skip duplicates, keep final zero

    return expA, thetas


def log_jacobian_from_thetas(thetas, n):
    # thetas includes final zero in case n is odd

    thetas_p = thetas[:-1] if n % 2 else thetas
    pdif = (thetas.unsqueeze(1) - thetas_p.unsqueeze(0)) / 2
    psum = (thetas.unsqueeze(1) + thetas_p.unsqueeze(0)) / 2
    return 2 * (log_sinc(pdif).sum() + log_sinc(psum).sum())


def log_volume_so_n(n):
    """calculate integral over SO(n) under Haar measure.

    This is the (log) product of sphere surface integrals from dim 1 to n-1.

    Cf. Thm 2.24, https://arxiv.org/abs/1509.00537 for O(n) case which is 2x.
    """

    return (
        (n - 1) * LOG2
        + (n * (n + 1) / 4) * LOGPI
        - torch.sum(torch.lgamma(torch.arange(1, n + 1) / 2))
    )
