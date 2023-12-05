from __future__ import annotations

import numbers
from typing import Union

import numpy as np
import torch

from wbgauss.scale_operators import EigScale, LogDiagScale, LogDiagScaleProj
from wbgauss.spcdist import BetaGaussian, _log_radius
from wbgauss.utils import (parallel_transport, sphere_distance, sphere_exp,
                           sphere_log)

LOG_2 = np.log(2)
LOG_PI = np.log(np.pi)
PI = np.pi
EPS = {torch.float32: 1e-4, torch.float64: 1e-7}
INF = {torch.float32: 1e38, torch.float64: 1e308}


class WBGBase(torch.distributions.Distribution):
    def __init__(self, loc: torch.Tensor, validate_args=None):
        self.loc = loc

        super().__init__(
            batch_shape=loc.shape[:-1],
            event_shape=loc.shape[-1:],
            validate_args=validate_args,
        )

    def log_prob(self, x):
        raise NotImplementedError

    def rsample(self, shape=torch.Size()):
        raise NotImplementedError

    def cross_fy(self, x):
        raise NotImplementedError

    ####### utils ###########
    @property
    def tangent_dim(self):
        # tangent space dimention
        return self.loc.shape[-1] - 1

    @property
    def amb_dim(self):
        # ambient space dimention
        return self.loc.shape[-1]

    @property
    def dtype(self):
        return self.loc.dtype

    def _too_far(self, x):
        # utility function to check if x is too far from the pole
        return sphere_distance(-self.loc, x) < EPS[self.dtype]

    def _get_log_det_Jacobian(self, rs):
        """Log J(v)
        Params:
            rs: the distances ||u||_2 on the tangnet plane
        """
        return torch.log(torch.abs(torch.sinc(rs / np.pi))) * (self.amb_dim - 2)

    def _to_tangent_space(self, x):
        raise NotImplementedError

    def wrapped_cross_fy(self, x, broadcast_batch=False):
        """Wrapped cross-FY loss including jacobian."""
        n = x.shape[-1]
        v = self._to_tangent_space(x)

        bg = self.tangent_dist  # assuming this is a beta gaussian...
        a_m1 = bg.alpha - 1

        # log_beta p_0(v), aka the Tsallis log-probability, but not clipped at 0
        logb_p0_v = -bg._tau - bg._mahalanobis(v, broadcast_batch) - 1 / a_m1

        # computing the jacobian directly for numerical reasons
        rs = torch.norm(v, dim=-1)
        logb_jac = (torch.sinc(rs / torch.pi) ** ((2 - n) * a_m1) - 1) / a_m1

        tangent_fy = a_m1 * (-bg.tsallis_entropy) - logb_p0_v
        wrapped_fy = tangent_fy - logb_jac - (a_m1 * logb_jac * logb_p0_v)

        return torch.where(
            self._too_far(x), torch.full_like(wrapped_fy, INF[self.dtype]), wrapped_fy
        )


class PoleWBGaussian(WBGBase):
    """Wrapped Beta-Gaussian distribution on the sphere with pole parametrization."""

    def __init__(
        self,
        loc: torch.Tensor,
        scale: Union[EigScale, LogDiagScale],
        alpha=1.05,
        inj_radius=PI,
        validate_args=None,
    ):
        if isinstance(alpha, numbers.Number):
            alpha: torch.Tensor = loc.new_tensor(alpha)
        assert (
            scale.shape[-1] == loc.shape[-1] - 1
        ), f"scale should be of dimention d-1, got scale:{scale.shape} and loc:{loc.shape}"

        batch_shape = torch.broadcast_shapes(
            scale.shape[:-2], loc.shape[:-1], alpha.shape
        )
        loc = loc.expand(batch_shape + (-1,))

        # assert loc is on the sphere
        assert torch.allclose(
            torch.norm(loc, dim=-1), torch.ones_like(loc[..., 0]), atol=1e-4
        ), "loc should be on the sphere"

        # can we also broadcase the scale?
        alpha = alpha.expand(batch_shape)

        self.loc = loc
        self.alpha = alpha
        self.inj_radius = inj_radius

        self.pole = torch.eye(self.amb_dim, device=loc.device, dtype=loc.dtype)[-1]

        # constraint eigenvalues of scale tilde
        rank = scale.rank
        log_radius = _log_radius(alpha, rank)
        max_norm = torch.exp(2 * (np.log(inj_radius) - log_radius))
        scale.reparametrize_(max_norm)  # max_norm
        assert torch.allclose(
            scale.rank, rank
        ), "rank should not change after reparametrization"

        self.tangent_dist = BetaGaussian(
            loc.new_zeros(batch_shape + (self.tangent_dim,)),
            scale,
            alpha=alpha,
            is_tilde=True,  # constrains are implemented only for the scale_tilde
            validate_args=validate_args,
        )
        self._scale = scale

        super().__init__(self.loc, validate_args=False)

    def log_prob(self, x, broadcast_batch=False):
        """log probability of x on the sphere"""
        v = self._to_tangent_space(x)
        rs = torch.norm(v, dim=-1)
        log_det_Jacobians = self._get_log_det_Jacobian(rs)
        lprob = self.tangent_dist.log_prob(v, broadcast_batch) - log_det_Jacobians
        return torch.where(self._too_far(x), torch.full_like(lprob, -torch.inf), lprob)

    def rsample(self, shape=torch.Size()):
        """returns: samples on the sphere"""
        v = self.tangent_dist.rsample(shape)
        # assert that the constraint is satisfied
        assert torch.norm(v, dim=-1).max() < self.inj_radius + 1e-8, torch.norm(
            v, dim=-1
        ).max()
        x = self._from_tangent_space(v)  # map to the sphere
        return x

    def cross_fy(self, x, broadcast_batch=False):
        """Tangent cross-FY loss."""
        mask = self._too_far(x)
        v = self._to_tangent_space(x)
        cross_fy = self.tangent_dist.cross_fy(v, broadcast_batch)
        return torch.where(mask, torch.full_like(cross_fy, INF[self.dtype]), cross_fy)

    def kl_fy(self, other: PoleWBGaussian, n_samples=1):
        """Tangent Fenchel-Young KL divergence.
        return KL(q(x) || p(x))
        """

        def log_beta(x, beta):
            # beta-logarithm
            return (torch.pow(x, 1 - beta) - 1) / (1 - beta)

        x = self.rsample(
            torch.Size(
                [
                    n_samples,
                ]
            )
        )  # sample from q
        v = self._to_tangent_space(x)
        rs = torch.norm(v, dim=-1)
        log_det_Jacobians = self._get_log_det_Jacobian(rs)

        beta = 2 - self.alpha
        omega_conj = (1 / (2 - beta)) * log_beta(
            torch.exp(self.tangent_dist.log_prob(v) - log_det_Jacobians), beta
        ).mean(0)
        cross_fy = other.wrapped_cross_fy(x).mean(0)
        return cross_fy + omega_conj

    def wasserstein_distance(self, other: PoleWBGaussian):
        assert torch.allclose(self.alpha, other.alpha), "alpha should be the same"
        dmu = sphere_distance(self.loc, other.loc)
        dscale = self.tangent_dist.scale.bures_distance(other.tangent_dist.scale)
        log_radius = _log_radius(self.alpha, self.tangent_dist.scale.rank)
        N = self.tangent_dist.scale.rank
        coef = log_radius.exp() ** 2 / (N + 2 * self.alpha / (self.alpha - 1))
        return dmu + dscale * coef

    ####### utils ###########
    def _to_tangent_space(self, x):
        """PT_{loc->pole} Log_loc(x)
        input: a point x on the sphere
        output: a vector v in tangent space at the north pole.
        """
        # x: on the sphere
        x = sphere_log(x, self.loc)  # project the points to the tangent plane
        x = self._parallel_transport(
            x, self.loc, self.pole
        )  # parallel transport the points to the pole
        x = self._remove_extra_dim(x)  # remove extra dim (d) -> (d - 1)
        return x

    def _from_tangent_space(self, v):
        v = self._add_missing_dim(v)  # add zeros for missing dim (d - 1) -> d
        y = self._parallel_transport(
            v, self.pole, self.loc
        )  # parallel transport from the pole to the location on the sphere
        x = sphere_exp(y, self.loc)
        return x

    def _add_missing_dim(self, x):
        return torch.cat([x, torch.zeros_like(x[..., :1])], dim=-1)

    def _remove_extra_dim(self, x):
        return x[..., :-1]

    def _parallel_transport(self, x, from_point, to_point):
        return parallel_transport(from_point, to_point, x)


class EmbeddedWBGaussian(WBGBase):
    """Wrapped Beta-Gaussian distribution on the sphere with embedded parametrization."""

    def __init__(
        self,
        loc: torch.Tensor,
        scale: LogDiagScaleProj,
        alpha=1.05,
        inj_radius=PI,
        validate_args=None,
    ):
        if isinstance(alpha, numbers.Number):
            alpha: torch.Tensor = loc.new_tensor(alpha)
        assert scale.shape[-1] == loc.shape[-1], "scale should be of dimention d"

        batch_shape = torch.broadcast_shapes(
            scale.shape[:-2], loc.shape[:-1], alpha.shape
        )
        loc = loc.expand(batch_shape + (-1,))
        # assert loc is on the sphere
        assert torch.allclose(
            torch.norm(loc, dim=-1), torch.ones_like(loc[..., 0])
        ), "loc should be on the sphere"

        # can we also broadcase the scale?
        alpha = alpha.expand(batch_shape)

        self.loc = loc
        self.alpha = alpha
        self.inj_radius = inj_radius

        self.pole = torch.eye(self.amb_dim, device=loc.device, dtype=loc.dtype)[-1]

        # constraint eigenvalues of scale tilde
        rank = scale.rank
        log_radius = _log_radius(alpha, rank)
        max_norm = torch.exp(2 * (np.log(inj_radius) - log_radius))
        scale.reparametrize_(max_norm)  # max_norm
        assert torch.allclose(
            scale.rank, rank
        ), "rank should not change after reparametrization"

        self.tangent_dist = BetaGaussian(
            loc.new_zeros(batch_shape + (self.amb_dim,)),
            scale,
            alpha=alpha,
            is_tilde=True,  # constrains are implemented only for the scale_tilde
            validate_args=validate_args,
        )

        super().__init__(self.loc, validate_args=False)

    def log_prob(self, x, broadcast_batch=False):
        y = self._to_tangent_space(x)
        rs = torch.norm(y, dim=-1)
        log_det_Jacobians = self._get_log_det_Jacobian(rs)
        lprob = self.tangent_dist.log_prob(x, broadcast_batch) - log_det_Jacobians
        return torch.where(self._too_far(x), torch.full_like(lprob, -torch.inf), lprob)

    def rsample(self, shape=torch.Size()):
        """returns: samples on the sphere"""
        v = self.tangent_dist.rsample(shape)
        # assert that the constraint is satisfied
        assert torch.norm(v, dim=-1).max() < self.inj_radius + 1e-8, torch.norm(
            v, dim=-1
        ).max()
        return sphere_exp(v, self.loc)

    def cross_fy(self, x, broadcast_batch=False):
        y = self._to_tangent_space(x)
        cross_fy = self.tangent_dist.cross_fy(y, broadcast_batch)
        return torch.where(
            self._too_far(x), torch.full_like(cross_fy, INF[self.dtype]), cross_fy
        )

    def kl_fy(self, other: EmbeddedWBGaussian, n_samples=1):
        """Tangent Fenchel-Young KL divergence.
        return KL(q(x) || p(x))
        """

        def log_beta(x, beta):
            # beta-logarithm
            return (torch.pow(x, 1 - beta) - 1) / (1 - beta)

        x = self.rsample(
            torch.Size(
                [
                    n_samples,
                ]
            )
        )  # sample from q
        y = self._to_tangent_space(x)
        rs = torch.norm(y, dim=-1)
        log_det_Jacobians = self._get_log_det_Jacobian(rs)

        beta = 2 - self.alpha
        omega_conj = (1 / (2 - beta)) * log_beta(
            torch.exp(self.tangent_dist.log_prob(y) - log_det_Jacobians), beta
        ).mean(0)
        cross_fy = other.wrapped_cross_fy(x).mean(0)
        return cross_fy + omega_conj

    def _to_tangent_space(self, x):
        x = sphere_log(x, self.loc)
        return x

    def _from_tangent_space(self, v):
        x = sphere_exp(v, self.loc)
        return x
