"""Sparse continuous distributions: beta-Gaussians

A refactoring of deep-spin/sparse-contiuous-distributions
"""
from __future__ import annotations

import math
import numbers
from typing import Union

import numpy as np
import torch
from torch.distributions import Beta, constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property

from wbgauss.scale_operators import BaseScale, EigScale

LOG_2 = np.log(2)
LOG_PI = np.log(np.pi)


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.

    https://github.com/pytorch/pytorch/issues/39242

    If x == 0 this returns -inf
    if x > 0  this returns nan
    """
    mask = -math.log(2) < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )


def _log_radius(alpha, rank):
    """Logarithm of the max-radius R of the distribution."""

    alpha = alpha
    alpha_m1 = alpha - 1
    alpha_ratio = alpha / alpha_m1

    # n = self.scale.rank
    n = rank
    half_n = n / 2

    lg_n_a = torch.lgamma(half_n + alpha_ratio)
    lg_a = torch.lgamma(alpha_ratio)

    log_first = lg_n_a - lg_a - half_n * LOG_PI
    log_second = (LOG_2 - torch.log(alpha_m1)) / alpha_m1
    log_inner = log_first + log_second
    log_radius = (alpha_m1 / (2 + alpha_m1 * n)) * log_inner
    return log_radius


class BetaGaussian(Distribution):
    arg_constraints = {
        "loc": constraints.real_vector,
        "alpha": constraints.greater_than(1),
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(
        self,
        loc: torch.Tensor,
        scale: Union[BaseScale, torch.Tensor],
        alpha=2,
        is_tilde=True,
        validate_args=None,
    ):
        """Batched multivariate beta-Gaussian random variable.

        The r.v. is parametrized in terms of a location (mean), scale
        (proportional to covariance) matrix, and scalar alpha>1.

        The pdf takes the form

        p(x) = [(alpha-1) * -.5 (x-u)' inv(Sigma) (x-u) - tau]_+ ** (alpha-1)

        where (u, Sigma) are the location and scale parameters.

        The distribution is parametrized via Sigma_Tilde if is_tilde=True.

        Parameters
        ----------
        loc: tensor, shape (broadcastable to) (*batch_dims, D)
            mean of the the distribution.

        scale: BaseScale or Tensor, shape (broadcast to) (*batch_dims, D, D)
            A representation of the scale parameter operator.
            If a raw tensor is passed, it is eigendecomposed.

        alpha: scalar or tensor broadcastable to (*batch_dims)
            The exponent parameter of the distribution.
            For alpha -> 1, the distribution converges to a Gaussian.
            For alpha = 2, the distribution is a Truncated Paraboloid
                (n-d generalization of the Epanechnikov kernel.)
            For alpha -> infty, the distribution converges to a
            uniform on an ellipsoid.
        """

        if isinstance(alpha, numbers.Number):
            alpha = loc.new_tensor(alpha)
        if isinstance(scale, torch.Tensor):
            scale = EigScale(scale)

        self.is_tilde = is_tilde

        # dimensions must be compatible to:
        # mean: [B x D]
        # scale: [B x D x D]
        # alpha: [B x 1]

        batch_shape = torch.broadcast_shapes(
            scale.shape[:-2], loc.shape[:-1], alpha.shape
        )

        event_shape = loc.shape[-1:]

        loc = loc.expand(batch_shape + (-1,))
        alpha = alpha.expand(batch_shape)

        self.loc = loc
        self.scale = scale
        self.alpha = alpha

        super().__init__(batch_shape, event_shape, validate_args)

    @lazy_property
    def sigma(self):
        """The scale matrix Sigma."""
        if self.is_tilde:
            sigma_tilde = self.scale.as_tensor()
            pdet = self.scale.log_det.exp()
            return sigma_tilde * pdet[..., None] ** (
                (self.alpha[..., None] - 1) / 2
            )  # sigma from sigma_tilde
        else:
            return self.scale.as_tensor()

    @lazy_property
    def sigma_tilde(self):
        """The scale matrix Sigma_Tilde."""
        if self.is_tilde:
            return self.scale.as_tensor()
        else:
            sigma = self.scale.as_tensor()
            N = self.scale.rank
            sigma_tilde = sigma * self.scale.log_det.exp() ** (
                -1 / (N + 2 / (self.alpha - 1))
            )
            return sigma_tilde

    @lazy_property
    def log_radius(self):
        """Logarithm of the max-radius R of the distribution."""
        return _log_radius(self.alpha, self.scale.rank)

    @lazy_property
    def _log_negtau(self):
        n = self.scale.rank
        c = 2 / (self.alpha - 1) if self.is_tilde else n + (2 / (self.alpha - 1))
        scaled_log_det = self.scale.log_det / c
        return 2 * self.log_radius - LOG_2 - scaled_log_det

    @lazy_property
    def _tau(self):
        return -torch.exp(self._log_negtau)

    @lazy_property
    def tsallis_entropy(self):
        """The Tsallis entropy -Omega_alpha of the distribution"""
        n = self.scale.rank
        alpha_m1 = self.alpha - 1
        alpha_term = 1 / (self.alpha * alpha_m1)
        denom = 2 * self.alpha + n * alpha_m1
        tau_term = 2 * self._tau / denom
        return alpha_term + tau_term

    @lazy_property
    def cov(self):
        """True covariance"""
        N = self.scale.rank
        alpha = self.alpha
        R = torch.exp(self.log_radius)
        return (R**2 / (N + 2 * alpha / (alpha - 1)))[
            ..., None, None
        ] * self.sigma_tilde

    def _inv_quad_diff(self, x, broadcast_batch=False):
        # x: shape [B', D] -- possibly different B'
        # loc: shape [B, D].

        d = x.shape[-1]

        if broadcast_batch:
            # assume loc is [B, D] and manually insert ones
            # to make it [B, 1,...1, D]

            x_batch_shape = x.shape[:-1]
            x = x.reshape(x_batch_shape + tuple(1 for _ in self.batch_shape) + (d,))

        # [B', B, D]
        diff = x - self.loc
        return self.scale.inv_quad(diff)

    def _mahalanobis(self, x, broadcast_batch=False):
        maha = self._inv_quad_diff(x, broadcast_batch) / 2
        if self.is_tilde:
            log_scale = ((1 - self.alpha) / 2) * self.scale.log_det
            maha = torch.exp(log_scale) * maha

        return maha

    def _log_mahalanobis(self, x, broadcast_batch=False):
        log_maha = torch.log(self._inv_quad_diff(x, broadcast_batch) / 2)
        if self.is_tilde:
            log_scale = ((1 - self.alpha) / 2) * self.scale.log_det
            log_maha = log_maha + log_scale
        return log_maha

    def log_prob(self, x, broadcast_batch=False):
        if not self.is_tilde:
            return self.pdf(x, broadcast_batch).log()

        # the more stable version of log_pdf
        a_m1 = self.alpha - 1
        log_am1 = torch.log(a_m1)
        log_negtau = self._log_negtau
        log_quad = torch.log(self._inv_quad_diff(x, broadcast_batch))
        log_Rsq = 2 * self.log_radius

        # log_ratio = log1p(maha/tau)
        # on support, log_quad < log_Rsq
        # outside of support, log_ratio = log(1-exp(0)) = -inf.
        log_ratio = log1mexp(torch.clip(log_quad - log_Rsq, max=0))

        return (log_am1 + log_negtau + log_ratio) / a_m1

    def pdf(self, x, broadcast_batch=False):
        """Probability of an broadcastable observation x (could be zero)"""
        f = -self._tau - self._mahalanobis(x, broadcast_batch)
        return torch.clip((self.alpha - 1) * f, min=0) ** (1 / (self.alpha - 1))

    def cross_fy(self, x, broadcast_batch=False):
        """The cross-Omega Fenchel-Young loss w.r.t. a Dirac observation x"""

        #  the more stable version of \ell^times loss

        if self.is_tilde:
            scaled_entr = (1 - self.alpha) * self.tsallis_entropy
            Rsq = torch.exp(2 * self.log_radius)
            inv_quad_diff = self._inv_quad_diff(x, broadcast_batch)
            # below, use fact that maha/tau = -inv_quad_diff/Rsq and factor tau
            tau_plus_maha = self._tau * (1 - inv_quad_diff / Rsq)
            return 1 / (self.alpha - 1) + scaled_entr + tau_plus_maha
        else:
            maha = self._mahalanobis(x, broadcast_batch)
            n = self.scale.rank
            scaled_entropy = (1 + (n * (self.alpha - 1)) / 2) * self.tsallis_entropy
            return maha + scaled_entropy - (n / (2 * self.alpha))

    def kl_fy(self, other: BetaGaussian, n_samples=1000):
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
        )
        beta = 2 - self.alpha
        omega_conj = (1 / (2 - beta)) * log_beta(self.pdf(x), beta).mean(0)
        cross_fy = other.cross_fy(x).mean(0)
        return cross_fy + omega_conj

    def rsample(self, sample_shape):
        """Draw samples from the distribution."""
        shape = self._extended_shape(sample_shape)

        radius = torch.exp(self.log_radius)
        radius = radius.expand(sample_shape + radius.shape)

        U = torch.randn(shape)
        # project U onto the correct sphere)
        U = self.scale.project_noise(U)

        norm = U.norm(dim=-1).unsqueeze(dim=-1)
        U = U / norm

        n = self.scale.rank
        half_n = n / 2
        alpha_m1 = self.alpha - 1
        alpha_ratio = self.alpha / alpha_m1

        ratio_dist = Beta(half_n, alpha_ratio).expand(shape[:-1])
        ratio = ratio_dist.rsample()
        r = radius * torch.sqrt(ratio)

        Z = r.unsqueeze(dim=-1) * U
        # Z = Z.unsqueeze(dim=-1)

        LZ = self.scale.sqrt_matvec(Z)

        if not self.is_tilde:
            c = torch.exp(-self.scale.log_det / (2 * n + 4 / (self.alpha - 1)))
            c = c.expand(sample_shape + c.shape).unsqueeze(-1)
            LZ = c * LZ

        return LZ
