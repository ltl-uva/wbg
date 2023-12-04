import torch as th

from .sphere import sphere_distance, sphere_log


def parallel_transport_1(x, y, ksi):
    """
    x, y: [d] prev, next locations of tangent plane
    ksi: [batch, d] points on T_loc1 to be mapped
    """
    log_x_y = sphere_log(y, x)
    log_y_x = sphere_log(x, y)
    proj = th.sum(log_x_y * ksi, dim=-1, keepdim=True)
    dist = sphere_distance(x, y) ** 2
    return ksi - (log_x_y + log_y_x) * proj / dist.unsqueeze(-1)


def parallel_transport_2(x, y, ksi):
    """
    Householder reflection
    x, y: [d] prev, next locations of tangent plane
    ksi: [batch, d] points on T_loc1 to be mapped
    """
    z = (x + y) / 2
    z = z / th.norm(z, dim=-1, keepdim=True)
    return ksi - 2 * z * th.sum(z * ksi, dim=-1, keepdim=True)


def parallel_transport(x, y, ksi):
    """
    x, y: [d] prev, next locations of tangent plane
    ksi: [batch, d] points on T_loc1 to be mapped
    """
    normx = th.norm(x, dim=-1, keepdim=True)
    normy = th.norm(y, dim=-1, keepdim=True)
    assert th.allclose(normx, th.ones_like(normx), atol=1e-5)
    assert th.allclose(normy, th.ones_like(normy), atol=1e-5)

    z = parallel_transport_2(x, y, ksi)

    assert th.allclose(
        th.sum(z * y, dim=-1, keepdim=True),
        th.sum(z * y, dim=-1, keepdim=True),
        atol=1e-5,
    )
    return z
