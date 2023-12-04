import math
import numbers

import numpy as np
import torch
from torch import Tensor

from wbgauss.scale_operators import LogDiagScale
from wbgauss.spcdist import BetaGaussian, _log_radius
from wbgauss.utils.skew_sym import (log_jacobian_from_thetas,
                                    skew_symmetric_exp_and_angles)

PI = math.pi
SQRT_2 = math.sqrt(2)
PI_SQRT_2 = PI * SQRT_2


class PoleWBGaussian:
    def __init__(
        self,
        loc: Tensor,
        scale: LogDiagScale,
        alpha=1.05,
        inj_radius=PI_SQRT_2,
        validate_args=None,
    ):
        if isinstance(alpha, numbers.Number):
            alpha: torch.Tensor = loc.new_tensor(alpha)

        # ensure loc is an BxNxN matrix, shape BxNxN
        self._n = loc.shape[-1]

        # the SO(n) manifold dimension.
        # scale should be a vector of this size.
        self._d = (self._n * (self._n - 1)) // 2

        self._batch_dims = loc.shape[:-2]
        self._ix = torch.tril_indices(self._n, self._n, offset=-1)

        if validate_args:
            if loc.shape[-2] != self._n:
                raise ValueError(
                    "`loc` should be a (broadcasted) square "
                    "matrix, instead got shape",
                    loc.shape,
                )

            if scale.shape[-1] != self._d:
                raise ValueError(
                    f"scale should be a (broadcasted) psd matrix,"
                    f"shape n(n-1)/2 when loc is n-by-n. Expected "
                    f"{self._d}, but got {scale.shape[-1]}."
                )

            if scale.shape[:-2] != self._batch_dims:
                raise ValueError(f"scale, loc should have the same batch dims.")

            dets = torch.det(loc)
            if not torch.allclose(dets, torch.ones(self._batch_dims)):
                raise ValueError(
                    f"loc should be orthogonal matrices. Got " f"determinant {dets}."
                )

        # constraint eigenvalues of sigma_tilde
        rank = scale.rank
        log_radius = _log_radius(alpha, rank)
        max_norm = torch.exp(2 * (np.log(inj_radius) - log_radius))
        scale.reparametrize_(max_norm)  # max_norm

        self.loc = loc
        self.scale = scale

        self.tangent_dist = BetaGaussian(
            loc.new_zeros(self._batch_dims + (self._d,)),
            scale=scale,
            alpha=alpha,
            is_tilde=True,
            validate_args=validate_args,
        )

    def _rsample_lie(self, shape) -> tuple[Tensor, Tensor, Tensor]:
        # sample lower triangle from tangent distribution
        lt = self.tangent_dist.rsample(sample_shape=shape)

        # build lie algebra elements (skew-symmetric matrices)

        # note: the grad is not a problem here
        # pytorch flows gradient into lt after the assignment.
        v = lt.new_zeros(shape + (self._n, self._n), requires_grad=False)
        v[..., self._ix[0], self._ix[1]] = lt / SQRT_2

        v += -v.transpose(-2, -1)

        # map to manifold
        x_id, thetas = skew_symmetric_exp_and_angles(v)

        x = self.loc @ x_id

        return x, lt, thetas

    def rsample(self, shape):
        x, _, _ = self._rsample_lie(shape)
        return x

    def sample_and_log_prob(self, shape=torch.Size([])):
        x, lt, thetas = self._rsample_lie(shape)
        logdet = log_jacobian_from_thetas(thetas, self._n)
        log_prob = self.tangent_dist.log_prob(lt) - logdet

        return x, log_prob

    def log_prob(self, x: Tensor, broadcast_batch: bool = False):
        raise NotImplementedError

    def cross_fy(self, x: Tensor, broadcast_batch: bool = False):
        raise NotImplementedError

    def wrapped_cross_fy(self, x: Tensor, broadcast_batch: bool = False):
        raise NotImplementedError
