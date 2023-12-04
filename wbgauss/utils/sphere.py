from math import pi

import geoopt
import torch as th
from geoopt.manifolds.sphere import EPS


def sphere_exp(u, loc):
    # point, loc
    sphere = geoopt.manifolds.sphere.SphereExact()
    z = sphere.expmap(loc, u)
    return z


def sphere_log(z, loc):
    # point, loc
    sphere = geoopt.manifolds.sphere.SphereExact()
    x = loc
    y = z
    u = sphere.proju(x, y - x)
    dist = sphere.dist(x, y, keepdim=True)
    return u * dist / u.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])


def sphere_distance_acos(a, b, inner=None, keepdim=False):
    norm = th.norm(a + b, dim=-1, keepdim=keepdim) / 2
    norm = th.clamp(norm, max=1 - EPS[norm.dtype])
    return pi - 2 * th.arcsin(norm)


def sphere_distance_asin(a, b, inner=None, keepdim=False):
    norm = th.norm(a - b, dim=-1, keepdim=keepdim) / 2
    norm = th.clamp(norm, max=1 - EPS[norm.dtype])
    return 2 * th.arcsin(norm)


def sphere_distance(a, b, keepdim=False):
    inner = th.sum(a * b, dim=-1, keepdim=keepdim)
    dist = th.where(
        inner > 0,
        sphere_distance_asin(a, b, keepdim=keepdim),
        sphere_distance_acos(a, b, inner=inner, keepdim=keepdim),
    )
    return dist
