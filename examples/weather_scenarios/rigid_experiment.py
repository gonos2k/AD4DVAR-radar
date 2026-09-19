"""CPU synthetic rigid-transport experiment; not a P0/P1 production path.

Center scatter preserves total echo but can perturb uniform fields under
rotation. It is not yet a consistent general divergence-free PDE remapper.
"""
from __future__ import annotations

import torch
from torch import Tensor


def rigid_remap(echo: Tensor, translation_yx: Tensor, angle: Tensor, *, steps: int = 1) -> Tensor:
    """Apply the h-fold rigid map once, distributing source echo to four cells.

    Equal-area Cartesian cells are assumed. Nonnegative scatter weights preserve
    echo integral except for explicit outflow. This is a cell-center quadrature,
    not exact polygon-overlap integration. Derivatives are piecewise within cells.
    """
    if echo.ndim != 2 or min(echo.shape) == 0 or translation_yx.shape != (2,) or angle.ndim != 0:
        raise ValueError('expected a 2-D field, two translation components, and scalar angle')
    if type(steps) is not int or not 1 <= steps <= 18:
        raise ValueError('steps must be an integer from 1 through 18')
    if echo.device.type != 'cpu' or echo.dtype not in (torch.float32, torch.float64):
        raise ValueError('this experiment supports CPU float32/float64')
    if translation_yx.dtype != echo.dtype or angle.dtype != echo.dtype:
        raise ValueError('field and transform must use the same dtype')
    if translation_yx.device != echo.device or angle.device != echo.device:
        raise ValueError('field and transform must use the same device')
    if not bool(torch.isfinite(echo).all() & torch.isfinite(translation_yx).all() & torch.isfinite(angle)):
        raise ValueError('all inputs must be finite')
    if bool((echo < 0).any()):
        raise ValueError('echo must be nonnegative')
    height, width = echo.shape
    y, x = torch.meshgrid(
        torch.arange(height, dtype=echo.dtype),
        torch.arange(width, dtype=echo.dtype), indexing='ij',
    )
    y, x = y - (height - 1)/2, x - (width - 1)/2
    # Compose the transform, not repeated interpolations of the field.
    cosine, sine = torch.cos(angle), torch.sin(angle)
    for _ in range(steps):
        x, y = cosine*x - sine*y + translation_yx[1], sine*x + cosine*y + translation_yx[0]
    y, x = y + (height - 1)/2, x + (width - 1)/2
    if not bool(torch.isfinite(y).all() & torch.isfinite(x).all()):
        raise ValueError('transform exceeds the supported finite coordinate range')
    # Avoid integer overflow for finite but far-outside transformed centers.
    active = (y > -1) & (y < height) & (x > -1) & (x < width)
    y = torch.where(active, y, torch.zeros_like(y))
    x = torch.where(active, x, torch.zeros_like(x))
    iy, ix = torch.floor(y), torch.floor(x)
    fy, fx = y-iy, x-ix
    result = torch.zeros_like(echo).flatten()
    for oy, wy in ((0, 1-fy), (1, fy)):
        for ox, wx in ((0, 1-fx), (1, fx)):
            target_y, target_x = iy.long()+oy, ix.long()+ox
            valid = active & (target_y >= 0) & (target_y < height) & (target_x >= 0) & (target_x < width)
            index = (target_y.clamp(0, height-1)*width + target_x.clamp(0, width-1)).flatten()
            contribution = torch.where(valid, echo*wy*wx, torch.zeros_like(echo))
            result = result.scatter_add(0, index, contribution.flatten())
    return result.reshape_as(echo)


def _rotate_yx(position_yx: Tensor, angle: Tensor) -> Tensor:
    """Rotate a centered ``(y, x)`` position using rigid_remap's angle sign."""

    cosine, sine = torch.cos(angle), torch.sin(angle)
    y, x = position_yx.unbind()
    return torch.stack((cosine * y + sine * x, -sine * y + cosine * x))


def _check_position_triplet(positions_yx: Tensor) -> None:
    if (
        not isinstance(positions_yx, Tensor)
        or positions_yx.shape != (3, 2)
        or positions_yx.device.type != 'cpu'
        or positions_yx.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError('positions_yx must be a CPU float32/float64 tensor with shape (3, 2)')
    if not bool(torch.isfinite(positions_yx).all()):
        raise ValueError('positions_yx must be finite')


def fit_rigid_transform_from_positions(positions_yx: Tensor) -> tuple[Tensor, Tensor]:
    """Fit ``(translation_yx, angle)`` from three centered feature positions.

    The positions are successive observations of one identifiable feature in
    grid-cell coordinates relative to the grid center.  For a rigid map,
    ``d_2 = R(angle) d_1``; hence the signed angle comes from the two observed
    displacement vectors and translation follows from the first position.
    A stationary (or unresolved) feature has no angular information and is
    rejected.  This deliberately does not claim to identify transforms from
    isotropic fields without a trackable feature.
    """

    _check_position_triplet(positions_yx)
    displacement_1 = positions_yx[1] - positions_yx[0]
    displacement_2 = positions_yx[2] - positions_yx[1]
    scale = max(1.0, float(torch.linalg.vector_norm(positions_yx).detach()))
    resolution = 64.0 * torch.finfo(positions_yx.dtype).eps * scale
    if (
        float(torch.linalg.vector_norm(displacement_1).detach()) <= resolution
        or float(torch.linalg.vector_norm(displacement_2).detach()) <= resolution
    ):
        raise ValueError('omega is unidentifiable from constant or unresolved positions')

    # Convert yx vectors to xy before using the conventional signed cross
    # product.  This preserves the angle convention used by rigid_remap.
    first_xy = displacement_1.flip(0)
    second_xy = displacement_2.flip(0)
    cross = first_xy[0] * second_xy[1] - first_xy[1] * second_xy[0]
    dot = torch.dot(first_xy, second_xy)
    angle = torch.atan2(cross, dot)
    translation_yx = positions_yx[1] - _rotate_yx(positions_yx[0], angle)
    return translation_yx, angle


def forecast_rigid_positions(
    initial_position_yx: Tensor,
    translation_yx: Tensor,
    angle: Tensor,
    *,
    lead_count: int = 18,
) -> Tensor:
    """Apply one fitted rigid transform directly for each future lead."""

    if (
        not isinstance(initial_position_yx, Tensor)
        or not isinstance(translation_yx, Tensor)
        or not isinstance(angle, Tensor)
        or initial_position_yx.shape != (2,)
        or translation_yx.shape != (2,)
        or angle.ndim != 0
        or initial_position_yx.device.type != 'cpu'
        or initial_position_yx.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError('expected CPU float32/float64 positions and scalar angle')
    if translation_yx.dtype != initial_position_yx.dtype or angle.dtype != initial_position_yx.dtype:
        raise ValueError('position and transform must use the same dtype')
    if not bool(torch.isfinite(initial_position_yx).all() & torch.isfinite(translation_yx).all() & torch.isfinite(angle)):
        raise ValueError('all positions and transform values must be finite')
    if type(lead_count) is not int or not 1 <= lead_count <= 18:
        raise ValueError('lead_count must be an integer from 1 through 18')

    position = initial_position_yx
    future = []
    for _ in range(lead_count):
        position = _rotate_yx(position, angle) + translation_yx
        if not bool(torch.isfinite(position).all()):
            raise ValueError('transform exceeds the supported finite position range')
        future.append(position)
    return torch.stack(future)
