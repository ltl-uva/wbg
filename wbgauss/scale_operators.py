"""Classes for scale SPD/SPSD operators.

Inspiration from scipy.stats _PSD class and cornelliusgp/linear_operator.

One day we could merge this with linear_operator and support modularity.
"""

from typing import Optional

import numpy as np
import torch
from torch.distributions.utils import lazy_property


def _parametrize_sigma_logsigmoid(logs, log_max):
    reparam_logs_sigma = torch.nn.functional.logsigmoid(logs)
    return log_max[..., None] + reparam_logs_sigma  # from -inf to log_max


def _parametrize_sigma_sigmoid(s, s_max):
    reparam_sigma = torch.sigmoid(s)
    return s_max[..., None] * reparam_sigma  # from 0 to s_max


def _constrain_log_diag(log_diag, log_max_value):
    # s: [batch_dims x d]
    # s_max: [batch_dims]
    # return: [batch_dims x d], s with constrained values
    return _parametrize_sigma_logsigmoid(log_diag, log_max_value)


def _constrain_diag(diag, max_value):
    # s: [batch_dims x d]
    # s_max: [batch_dims]
    # return: [batch_dims x d], s with constrained values
    return _parametrize_sigma_sigmoid(diag, max_value)


# reimplement scipy's cutoff for eigenvalues
def _eigvalsh_to_eps(
    spectra: torch.Tensor, rcond: Optional[float] = None, log=False
) -> torch.Tensor:
    """
    For each spectrum, compute the appropriate cutoff.

    Parameters
    ----------
    spectra: tensor, [batch_dims x d]

    Returns
    -------
    eps: tensor, [batch_dims]
    """
    spectra = spectra.detach()
    if rcond in [None, -1]:
        t = str(spectra.dtype)[-2:]
        factor = {"32": 1e3, "64": 1e6}
        rcond = factor[t] * torch.finfo(spectra.dtype).eps

    rcond = spectra.new_full(size=spectra.shape[:-1], fill_value=rcond)
    if log:
        return torch.log(rcond) + torch.max(spectra, dim=-1).values
    else:
        return rcond * torch.max(torch.abs(spectra), dim=-1).values


class BaseScale:
    """API for scale operators"""

    @property
    def rank(self) -> torch.Tensor:
        return NotImplementedError

    @property
    def trace(self) -> torch.Tensor:
        return NotImplementedError

    @property
    def log_det(self) -> torch.Tensor:
        return NotImplementedError

    @property
    def shape(self):
        return NotImplementedError

    def inv_quad(self, x) -> torch.Tensor:
        """compute quadratic form x' S^-1 x with appropriate batching"""
        return NotImplementedError

    def sqrt_matvec(self, x) -> torch.Tensor:
        """compute S^{1/2} x with appropriate batching."""
        return NotImplementedError

    def as_tensor(self) -> torch.Tensor:
        return NotImplementedError

    def reparametrize_(self, max_value):
        """constrain the maximal eigenvalue of the scale operator"""
        return NotImplementedError

    def bures_distance(self, other):
        """compute the Bures distance between two scale operators"""
        return NotImplementedError


class EigScale(BaseScale):
    def __init__(self, scales, rcond=None):
        """Factorized representation of a batch of scale PSD matrices."""

        self._shape = scales.shape
        self._zero = scales.new_zeros(size=(1,))
        self.rcond = rcond

        # scales: [B x D x D]

        # s, u = torch.symeig(scales, eigenvectors=True, upper=True)
        s, u = torch.linalg.eigh(scales)
        self.u = u

        self.s_mask, self.s, self.s_inv = self._mask_s(s, rcond)

    def _mask_s(self, s, rcond):
        eps = _eigvalsh_to_eps(s, rcond)

        if torch.any(torch.min(s, dim=-1).values < -eps):
            raise ValueError("scale is not positive semidefinite")

        # probably could use searchsorted
        s_mask = s > eps.unsqueeze(dim=-1)

        # P = u @ s @ u.T
        s = torch.where(s_mask, s, self._zero)
        s_inv = torch.where(s_mask, 1 / s, self._zero)
        return s_mask, s, s_inv

    @property
    def shape(self):
        return self._shape

    @lazy_property
    def rank(self) -> torch.Tensor:
        return self.s_mask.sum(dim=-1)

    @lazy_property
    def trace(self) -> torch.Tensor:
        return self.s.sum(dim=-1)

    @lazy_property
    def log_det(self) -> torch.Tensor:
        log_s = torch.where(self.s_mask, torch.log(self.s), self._zero)
        return torch.sum(log_s, dim=-1)

    def inv_quad(self, x: torch.Tensor) -> torch.Tensor:
        """compute quadratic form x' S^-1 x with appropriate batching

        If scale has shape [B, D, D], x should have shape [B', B, D]
        where B, B' may be tuples (possibly zero).

        Return will have shape [B', B].
        """
        # this is suboptimal if x = y-u and y, u must broadcast.
        # in that case we should use:
        # (y-u)' Q (y-u) = y'Qy + u'Qu - 2yQu
        # but u=0 always for wrapped distributions.

        Ux = self.u.transpose(-2, -1) @ x.unsqueeze(-1)
        Lx = torch.sqrt(self.s_inv) * Ux.squeeze(-1)
        return Lx.square().sum(dim=-1)

    def sqrt_matvec(self, x) -> torch.Tensor:
        """compute Lx, where LL'=S, with appropriate batching."""
        S_sqrt_x = torch.mul(x, torch.sqrt(self.s))  # shape=x
        # Lx = self.u.transpose(-2, -1) @ S_sqrt_x.unsqueeze(-1)
        Lx = self.u @ S_sqrt_x.unsqueeze(-1)
        return Lx.squeeze(-1)

    def as_tensor(self) -> torch.Tensor:
        return self.u @ (self.s.unsqueeze(dim=-1) * self.u.transpose(-2, -1))

    def project_noise(self, u) -> torch.Tensor:
        """When sampling from a low-rank dist, initial noise is projected"""
        mask = self.s_mask.expand(u.shape)
        return torch.where(mask, u, u.new_zeros(1))

    def reparametrize_(self, max_value):
        """constrain the maximal eigenvalue of the scale operator"""
        s = _constrain_diag(self.s, max_value)
        self.s_mask, self.s, self.s_inv = self._mask_s(s, self.rcond)


class DiagScale(EigScale):
    def __init__(self, diag_scales, rcond=None):
        """Compact representation of a batch of diagonal scale matrices."""
        self._diag_scales = diag_scales

        self._zero = diag_scales.new_zeros(size=(1,))

        shape = diag_scales.shape
        self._shape = shape + (shape[-1],)
        self.rcond = rcond

        self.s_mask, self.s, self.s_inv = self._mask_s(diag_scales, rcond)

    def _mask_s(self, diag_scales, rcond):
        eps = _eigvalsh_to_eps(diag_scales, rcond)

        if torch.any(torch.min(diag_scales, dim=-1).values < -eps):
            raise ValueError("scale is not positive semidefinite")

        s_mask = diag_scales > eps.unsqueeze(dim=-1)
        s = torch.where(s_mask, diag_scales, self._zero)
        s_inv = torch.where(s_mask, 1 / diag_scales, self._zero)
        return s_mask, s, s_inv

    def as_tensor(self) -> torch.Tensor:
        return torch.diag_embed(self.s)

    def inv_quad(self, x: torch.Tensor) -> torch.Tensor:
        """compute quadratic form x' S^-1 x with appropriate batching

        If scale has shape [B, D, D], x should have shape [B', B, D]
        where B, B' may be tuples (possibly zero).

        Return will have shape [B', B].
        """

        Lx = torch.sqrt(self.s_inv) * x
        return Lx.square().sum(dim=-1)

    def sqrt_matvec(self, x) -> torch.Tensor:
        """compute Lx, where LL'=S, with appropriate batching."""
        Bx = torch.sqrt(self.s) * x
        return Bx

    def reparametrize_(self, max_value):
        """constrain the maximal eigenvalue of the scale operator"""
        diag_scales = _constrain_diag(self._diag_scales, max_value)
        self.s_mask, self.s, self.s_inv = self._mask_s(diag_scales, self.rcond)


class LogDiagScale(BaseScale):
    def __init__(self, log_diag_scales, rcond=None):
        """Compact representation of a batch of diagonal scale matrices."""

        self._zero = torch.log(log_diag_scales.new_zeros(size=(1,)))
        self._log_diag_scales = log_diag_scales

        shape = log_diag_scales.shape
        self._shape = shape + (shape[-1],)

        self.rcond = rcond
        self.s_mask, self.s, self.s_inv = self._mask_s(log_diag_scales, rcond)

    def _mask_s(self, log_diag_scales, rcond):
        log_eps = _eigvalsh_to_eps(log_diag_scales, rcond, log=True)

        if torch.any(torch.min(log_diag_scales, dim=-1).values < log_eps):
            print(
                "scale is not positive semidefinite, min(log_diag_scales):",
                torch.min(log_diag_scales),
                "max log_eps:",
                torch.max(log_eps),
            )
            # raise ValueError('scale is not positive semidefinite')

        s_mask = log_diag_scales > log_eps.unsqueeze(dim=-1)
        s = torch.where(s_mask, log_diag_scales, self._zero)
        s_inv = torch.where(s_mask, -log_diag_scales, self._zero)
        return s_mask, s, s_inv

    def as_tensor(self) -> torch.Tensor:
        return torch.diag_embed(torch.exp(self.s))

    @property
    def shape(self):
        return self._shape

    @lazy_property
    def rank(self) -> torch.Tensor:
        return self.s_mask.sum(dim=-1)

    @lazy_property
    def trace(self) -> torch.Tensor:
        return torch.exp(torch.logsumexp(self.s, dim=-1))

    @lazy_property
    def trace_inv(self) -> torch.Tensor:
        return torch.exp(torch.logsumexp(-self.s, dim=-1))

    @lazy_property
    def log_det(self) -> torch.Tensor:
        log_s = torch.where(self.s_mask, self.s, 0)
        return torch.sum(log_s, dim=-1)

    def inv_quad(self, x: torch.Tensor) -> torch.Tensor:
        """compute quadratic form x' S^-1 x with appropriate batching

        If scale has shape [B, D, D], x should have shape [B', B, D]
        where B, B' may be tuples (possibly zero).

        Return will have shape [B', B].
        """
        Lx = torch.exp(self.s_inv / 2) * x
        return Lx.square().sum(dim=-1)

    def sqrt_matvec(self, x) -> torch.Tensor:
        """compute Lx, where LL'=S, with appropriate batching."""
        Bx = torch.exp(self.s / 2) * x
        return Bx

    def project_noise(self, u) -> torch.Tensor:
        """When sampling from a low-rank dist, initial noise is projected"""
        mask = self.s_mask.expand(u.shape)
        return torch.where(mask, u, u.new_zeros(1))

    def reparametrize_(self, max_value):
        """constrain the maximal eigenvalue of the scale operator"""
        log_diag_scales = _constrain_log_diag(
            self._log_diag_scales, torch.log(max_value)
        )
        self.s_mask, self.s, self.s_inv = self._mask_s(log_diag_scales, self.rcond)

    def bures_distance(self, other):
        sqrt_s1 = torch.exp(self.s / 2)
        sqrt_s2 = torch.exp(other.s / 2)
        return (sqrt_s1 - sqrt_s2).square().sum(dim=-1)


class LogDiagScaleProj(LogDiagScale):
    def __init__(self, scales, x, rcond=None):
        """Compact representation of projected log-diagonal scale matrices.

        Where S = diag(exp(log_scales)), this class
        implements the linear operator (1-xx') S (1-xx')
        """
        if x.shape[-1] == 1:
            raise ValueError("x should be at least 2 dimentional")

        if torch.any(torch.min(scales, dim=-1).values < np.log(1e-10)):
            # crop scales to be at least 1e-10
            scales = torch.where(scales < np.log(1e-10), np.log(1e-10), scales)

        self.x = x
        super().__init__(scales, rcond=1e-30)

        # low rank is not supported
        self.s_mask = torch.ones_like(self.s, dtype=torch.bool)

    def as_tensor(self) -> torch.Tensor:
        S = super().as_tensor()
        # P = I - xx'
        P = torch.diag_embed(torch.ones_like(self.s)) - self.x.unsqueeze(
            -1
        ) @ self.x.unsqueeze(-2)
        return P @ S @ P

    @property
    def rank(self):
        # this class does not support low-rank matrices at the moment.
        return self.s_mask.sum(-1) - 1

    @lazy_property
    def _log_xsx(self):
        return torch.logsumexp(2 * torch.log(torch.abs(self.x)) + self.s, dim=-1)

    @lazy_property
    def _log_xsinvx(self):
        return torch.logsumexp(2 * torch.log(torch.abs(self.x)) - self.s, dim=-1)

    @lazy_property
    def trace(self):
        return torch.exp(torch.logsumexp(self.s, dim=-1)) - torch.exp(self._log_xsx)

    @lazy_property
    def log_det(self):
        # warning: this expression is only correct for full rank s.
        return torch.sum(self.s, dim=-1) + self._log_xsinvx

    def inv_quad(self, X):
        esinv = torch.exp(self.s_inv)
        tmp = torch.sum(esinv * X**2, dim=-1)
        corr_num = torch.sum(X * self.x * esinv, dim=-1) ** 2
        corr_den = torch.sum(self.x**2 * esinv, dim=-1)
        return tmp - corr_num / corr_den

    def sqrt_matvec(self, X) -> torch.Tensor:
        # [...Batch, n]
        tmp = torch.exp(self.s / 2) * X

        # project to tangent plane at self.x
        # aka (I-xx') tmp
        tmp = tmp - (tmp * self.x).sum(dim=-1, keepdim=True) * self.x
        return tmp

    def project_noise(self, u):
        u = u - (u * self.x).sum(dim=-1, keepdim=True) * self.x
        return u

    def reparametrize_(self, max_value):
        """constrain the maximal eigenvalue of the scale operator"""
        log_diag_scales = _constrain_log_diag(
            self._log_diag_scales, torch.log(max_value)
        )
        _, self.s, self.s_inv = self._mask_s(log_diag_scales, self.rcond)

        # low rank is not supported
        self.s_mask = torch.ones_like(self.s, dtype=torch.bool)
