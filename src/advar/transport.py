"""Small finite-volume transport baseline for the spatial-flow candidate.

The module intentionally contains only local Cartesian-grid operations.  The
caller owns the gauge used to construct ``psi`` and any time-step schedule.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Callable, Iterator, Optional, Sequence, Tuple, cast

import torch
from torch import Tensor

from .matrix_free import recompute


@dataclass(frozen=True)
class FVStepResult:
    """Result and mass-budget terms for one SSPRK2 finite-volume step."""

    echo: Tensor
    support: Tensor
    transport_inflow: Tensor
    transport_outflow: Tensor
    physical_inflow: Tensor
    physical_outflow: Tensor
    source_quadrature: Tensor


BoundaryEdges = Tuple[Tensor, Tensor, Tensor, Tensor]
BoundaryStages = Tuple[BoundaryEdges, BoundaryEdges]
BoundarySchedule = Tuple[BoundaryStages, ...]

_Edges = BoundaryEdges
_BoundaryStages = BoundaryStages
_SMALL_LOG_GROWTH = 0.125
_MinmodStageObserver = Callable[[Tensor, Tensor, Tensor], None]
_minmod_stage_observer: ContextVar[_MinmodStageObserver | None] = ContextVar(
    "advar_minmod_stage_observer", default=None,
)


@contextmanager
def observe_minmod_stages(observer: _MinmodStageObserver) -> Iterator[None]:
    """Observe this call context's minmod stages without replacing model code.

    Diagnostics receive detached copies and should run outside automatic
    differentiation and checkpoint replay.
    Nested observers run from innermost to outermost, matching the old serial
    wrapper order while keeping each thread/task's collector separate.
    """
    if not callable(observer):
        raise TypeError("minmod stage observer must be callable")
    previous = _minmod_stage_observer.get()

    def notify(q: Tensor, qx: Tensor, qy: Tensor) -> None:
        # Each observer gets its own diagnostic copy; neither model state nor
        # an enclosing observer can be changed by an inner callback.
        observer(q.detach().clone(), qx.detach().clone(), qy.detach().clone())
        if previous is not None:
            previous(q, qx, qy)

    token = _minmod_stage_observer.set(notify)
    try:
        yield
    finally:
        _minmod_stage_observer.reset(token)


def _check_tensor(name: str, value: object, *, dtype: Optional[torch.dtype] = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have float32 or float64 dtype")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _check_nonnegative(name: str, value: Tensor) -> None:
    if bool((value < 0).any()):
        raise ValueError(f"{name} must be nonnegative")


def _check_support(name: str, value: Tensor) -> None:
    _check_nonnegative(name, value)
    # Shared-face cancellation and convex sums can round a constant one just
    # above one. Preserve those values and their derivatives rather than clip.
    tolerance = 16 * torch.finfo(value.dtype).eps
    if bool((value > 1 + tolerance).any()):
        raise ValueError(f"{name} must lie in [0, 1] within dtype roundoff")


def _validate_edges(
    name: str,
    stages: object,
    *,
    height: int,
    width: int,
    dtype: torch.dtype,
) -> _BoundaryStages:
    if not isinstance(stages, (tuple, list)) or len(stages) != 2:
        raise ValueError(f"{name} must contain two stage edge tuples")
    expected = ((height,), (height,), (width,), (width,))
    checked = []
    for stage_index, stage in enumerate(stages):
        if not isinstance(stage, (tuple, list)) or len(stage) != 4:
            raise ValueError(f"{name}[{stage_index}] must be (left, right, bottom, top)")
        edges = []
        for edge_index, (edge, shape) in enumerate(zip(stage, expected)):
            edge_name = ("left", "right", "bottom", "top")[edge_index]
            edge = _check_tensor(f"{name}[{stage_index}].{edge_name}", edge, dtype=dtype)
            if edge.shape != shape:
                raise ValueError(
                    f"{name}[{stage_index}].{edge_name} must have shape {shape}"
                )
            edges.append(edge)
        checked.append(tuple(edges))
    return checked[0], checked[1]  # type: ignore[return-value]


def _zero_edges(reference: Tensor, height: int, width: int) -> _Edges:
    # Keep a zero diagnostic/output connected to the reference graph.  This
    # matters for callers that differentiate a complete no-inflow step.
    zero = reference.reshape(-1)[0] * 0
    return (
        zero.expand(height),
        zero.expand(height),
        zero.expand(width),
        zero.expand(width),
    )


def _volume_parts(volume_flux: Tensor) -> Tuple[Tensor, Tensor]:
    # At a tie, maximum/minimum split the derivative equally. Direct selection
    # also preserves subnormal fluxes that would vanish when multiplied by 0.5.
    zero = torch.zeros_like(volume_flux)
    return torch.maximum(volume_flux, zero), torch.minimum(volume_flux, zero)


def _scale_by_growth(value: Tensor, log_growth: Tensor) -> Tensor:
    """Scale a transported value without rounding a small integrating factor."""
    if bool(torch.abs(log_growth) < _SMALL_LOG_GROWTH):
        return torch.addcmul(value, value, torch.expm1(log_growth))
    return torch.exp(log_growth) * value


def _euler_donorcell(
    q: Tensor,
    qx: Tensor,
    qy: Tensor,
    edges: _Edges,
    dt: Tensor,
    area: Tensor,
) -> Tensor:
    """Positive-coefficient Euler update for one stage."""

    left, right, bottom, top = edges
    qx_plus, qx_minus = _volume_parts(qx)
    qy_plus, qy_minus = _volume_parts(qy)
    outgoing = (
        qx_plus[:, 1:]
        - qx_minus[:, :-1]
        + qy_plus[1:, :]
        - qy_minus[:-1, :]
    )
    left_source = torch.cat((left[:, None], q[:, :-1]), dim=1)
    right_source = torch.cat((q[:, 1:], right[:, None]), dim=1)
    bottom_source = torch.cat((bottom[None, :], q[:-1, :]), dim=0)
    top_source = torch.cat((q[1:, :], top[None, :]), dim=0)
    incoming = (
        qx_plus[:, :-1] * left_source
        - qx_minus[:, 1:] * right_source
        + qy_plus[:-1, :] * bottom_source
        - qy_minus[1:, :] * top_source
    )
    # Use the same division/multiplication order as the CFL acceptance check.
    return (1 - dt * (outgoing / area)) * q + dt * (incoming / area)


def _minmod(left: Tensor, right: Tensor) -> Tensor:
    # Avoid left*right: finite inputs can overflow in that sign test.
    positive = (left > 0) & (right > 0)
    negative = (left < 0) & (right < 0)
    positive_value = torch.minimum(left, right)
    negative_value = torch.maximum(left, right)
    return torch.where(
        positive,
        positive_value,
        torch.where(negative, negative_value, torch.zeros_like(left)),
    )


def _muscl_slopes(q: Tensor) -> Tuple[Tensor, Tensor]:
    """Return minmod slopes with zero reconstruction on the perimeter."""

    height, width = q.shape
    if width <= 2:
        sx = torch.zeros_like(q)
    else:
        interior_x = _minmod(
            q[:, 1:-1] - q[:, :-2],
            q[:, 2:] - q[:, 1:-1],
        )
        sx = torch.cat(
            (torch.zeros_like(q[:, :1]), interior_x, torch.zeros_like(q[:, :1])),
            dim=1,
        )
        if height > 2:
            sx = torch.cat(
                (torch.zeros_like(q[:1, :]), sx[1:-1, :], torch.zeros_like(q[:1, :])),
                dim=0,
            )
        else:
            sx = torch.zeros_like(q)

    if height <= 2:
        sy = torch.zeros_like(q)
    else:
        interior_y = _minmod(
            q[1:-1, :] - q[:-2, :],
            q[2:, :] - q[1:-1, :],
        )
        sy = torch.cat(
            (torch.zeros_like(q[:1, :]), interior_y, torch.zeros_like(q[:1, :])),
            dim=0,
        )
        if width > 2:
            sy = torch.cat(
                (torch.zeros_like(q[:, :1]), sy[:, 1:-1], torch.zeros_like(q[:, :1])),
                dim=1,
            )
        else:
            sy = torch.zeros_like(q)
    return sx, sy


def _euler_minmod(
    q: Tensor,
    qx: Tensor,
    qy: Tensor,
    edges: _Edges,
    dt: Tensor,
    area: Tensor,
) -> Tensor:
    """Advance one minmod-MUSCL Euler stage with supplied exterior traces."""

    observer = _minmod_stage_observer.get()
    if observer is not None:
        observer(q, qx, qy)

    sx, sy = _muscl_slopes(q)
    left_face = q - 0.5 * sx
    right_face = q + 0.5 * sx
    bottom_face = q - 0.5 * sy
    top_face = q + 0.5 * sy
    for name, value in (
        ("left MUSCL face", left_face),
        ("right MUSCL face", right_face),
        ("bottom MUSCL face", bottom_face),
        ("top MUSCL face", top_face),
    ):
        _check_tensor(name, value, dtype=q.dtype)

    left, right, bottom, top = edges
    qx_plus, qx_minus = _volume_parts(qx)
    qy_plus, qy_minus = _volume_parts(qy)
    left_source = torch.cat((left[:, None], right_face[:, :-1]), dim=1)
    right_source = torch.cat((left_face[:, 1:], right[:, None]), dim=1)
    bottom_source = torch.cat((bottom[None, :], top_face[:-1, :]), dim=0)
    top_source = torch.cat((bottom_face[1:, :], top[None, :]), dim=0)
    incoming = (
        qx_plus[:, :-1] * left_source
        - qx_minus[:, 1:] * right_source
        + qy_plus[:-1, :] * bottom_source
        - qy_minus[1:, :] * top_source
    )
    outgoing = (
        qx_plus[:, 1:] * right_face
        - qx_minus[:, :-1] * left_face
        + qy_plus[1:, :] * top_face
        - qy_minus[:-1, :] * bottom_face
    )
    return q - dt * (outgoing / area) + dt * (incoming / area)


def _check_minmod_support(
    support: Tensor,
    qx: Tensor,
    qy: Tensor,
    support_edges: _BoundaryStages,
) -> None:
    """Require full-known support wherever an exterior value can enter."""

    tolerance = 16 * torch.finfo(support.dtype).eps
    if bool((support < 1 - tolerance).any()):
        raise ValueError("minmod reconstruction requires fully known initial support")
    incoming_masks = (
        (qx[:, 0] > 0, qx[:, -1] < 0, qy[0, :] > 0, qy[-1, :] < 0),
        (qx[:, 0] > 0, qx[:, -1] < 0, qy[0, :] > 0, qy[-1, :] < 0),
    )
    edge_names = ("left", "right", "bottom", "top")
    for stage_index, (edges, masks) in enumerate(zip(support_edges, incoming_masks)):
        for edge_name, edge, incoming in zip(edge_names, edges, masks):
            if bool(torch.any(incoming & (edge < 1 - tolerance))):
                raise ValueError(
                    "minmod reconstruction requires fully known incoming "
                    f"boundary_support[{stage_index}].{edge_name}"
                )


def _boundary_budget(
    qx: Tensor,
    qy: Tensor,
    q: Tensor,
    edges: _Edges,
) -> Tuple[Tensor, Tensor]:
    """Return incoming/outgoing boundary mass rates for one stage.

    Directional positive parts are the actual stage flux weights.  Thus an
    absent boundary (represented by graph-connected zeros) contributes no
    known inflow, while outflow always uses the adjacent interior value.
    """

    left, right, bottom, top = edges
    qx_plus, qx_minus = _volume_parts(qx)
    qy_plus, qy_minus = _volume_parts(qy)
    incoming = (
        (qx_plus[:, 0] * left).sum()
        + (-qx_minus[:, -1] * right).sum()
        + (qy_plus[0, :] * bottom).sum()
        + (-qy_minus[-1, :] * top).sum()
    )
    outgoing = (
        ((-qx_minus[:, 0]) * q[:, 0]).sum()
        + (qx_plus[:, -1] * q[:, -1]).sum()
        + ((-qy_minus[0, :]) * q[0, :]).sum()
        + (qy_plus[-1, :] * q[-1, :]).sum()
    )
    return incoming, outgoing


def _validate_divergence(qx: Tensor, qy: Tensor) -> None:
    divergence = (qx[:, 1:] - qx[:, :-1]) + (qy[1:, :] - qy[:-1, :])
    scale = (
        torch.abs(qx[:, 1:])
        + torch.abs(qx[:, :-1])
        + torch.abs(qy[1:, :])
        + torch.abs(qy[:-1, :])
    )
    if not bool(torch.isfinite(divergence).all()) or not bool(torch.isfinite(scale).all()):
        raise ValueError("discrete divergence is not finite")
    tolerance = 256 * torch.finfo(qx.dtype).eps * scale
    if bool((torch.abs(divergence) > tolerance).any()):
        raise ValueError("qx and qy must satisfy discrete divergence zero")


def face_volume_fluxes(psi: Tensor) -> Tuple[Tensor, Tensor]:
    """Construct shared Cartesian face volume fluxes from vertex ``psi``.

    ``psi`` has shape ``[H+1, W+1]``.  Rows increase physical +y, so
    ``Qx = psi[1:] - psi[:-1]`` and ``Qy = -(psi[:, 1:] - psi[:, :-1])``.
    """

    psi = _check_tensor("psi", psi)
    if psi.ndim != 2 or psi.shape[0] < 2 or psi.shape[1] < 2:
        raise ValueError("psi must have shape [H+1, W+1] for a non-empty grid")
    return psi[1:, :] - psi[:-1, :], -(psi[:, 1:] - psi[:, :-1])


def finite_volume_step(
    echo: Tensor,
    support: Tensor,
    qx: Tensor,
    qy: Tensor,
    *,
    dt_seconds: float,
    spacing_yx: Tuple[float, float],
    log_growth: Tensor | float = 0,
    boundary_echo: Optional[Sequence[Sequence[Tensor]]] = None,
    boundary_support: Optional[Sequence[Sequence[Tensor]]] = None,
    max_courant: float = 0.9,
    reconstruction: str = "donorcell",
) -> FVStepResult:
    """Advance one constant-growth Cartesian FV SSPRK2 substep.

    ``log_growth`` is the integrated growth over this substep.  Boundary
    traces are physical ``q``/support values at stage 0 and stage 1. Positive
    homogeneity lets the second RK stage use physical coordinates without
    first multiplying its boundary trace by inverse growth.
    Echo traces are already whole-face known contributions: support is not
    multiplied into them. At a jump, supply the start's right trace and the
    end's left trace; the caller owns time splitting and the fixed schedule.
    The source quadrature approximates physical mass change, whereas the
    transformed transport budget is the discrete SSPRK2 identity.  The
    default donor-cell reconstruction supports partial/unknown contributions.
    ``reconstruction="minmod"`` is an opt-in full-known-field MUSCL path:
    incoming boundary support must be one (within dtype roundoff), and the
    reconstruction is not a separable known-contribution operator.
    Both growth exponentials, the evaluated stages, and all returned budget
    terms must be representable in the input dtype. A finite physical answer
    alone does not guarantee a representable transformed transport budget.
    """

    if not isinstance(reconstruction, str):
        raise TypeError("reconstruction must be 'donorcell' or 'minmod'")
    if reconstruction not in ("donorcell", "minmod"):
        raise ValueError("reconstruction must be 'donorcell' or 'minmod'")
    echo = _check_tensor("echo", echo)
    support = _check_tensor("support", support, dtype=echo.dtype)
    qx = _check_tensor("qx", qx, dtype=echo.dtype)
    qy = _check_tensor("qy", qy, dtype=echo.dtype)
    if echo.ndim != 2 or echo.shape[0] == 0 or echo.shape[1] == 0:
        raise ValueError("echo must be a non-empty [H, W] grid")
    if support.shape != echo.shape:
        raise ValueError("support must have the same shape as echo")
    height, width = echo.shape
    if qx.shape != (height, width + 1):
        raise ValueError(f"qx must have shape {(height, width + 1)}")
    if qy.shape != (height + 1, width):
        raise ValueError(f"qy must have shape {(height + 1, width)}")
    _check_nonnegative("echo", echo)
    _check_support("support", support)
    _validate_divergence(qx, qy)

    if not isinstance(dt_seconds, Real) or isinstance(dt_seconds, bool):
        raise TypeError("dt_seconds must be a positive finite float")
    dt_value = float(dt_seconds)
    if not math.isfinite(dt_value) or dt_value <= 0:
        raise ValueError("dt_seconds must be a positive finite float")
    if not isinstance(spacing_yx, (tuple, list)) or len(spacing_yx) != 2:
        raise ValueError("spacing_yx must be (dy, dx)")
    try:
        dy, dx = float(spacing_yx[0]), float(spacing_yx[1])
    except (TypeError, ValueError):
        raise TypeError("spacing_yx must contain finite positive floats") from None
    if not math.isfinite(dy) or not math.isfinite(dx) or dy <= 0 or dx <= 0:
        raise ValueError("spacing_yx must contain finite positive floats")
    area_value = dy * dx
    if not math.isfinite(area_value) or area_value <= 0:
        raise ValueError("physical cell area must be finite and positive")
    area = echo.new_tensor(area_value)
    if not bool(torch.isfinite(area)) or not bool(area > 0):
        raise ValueError("physical cell area is not representable in the input dtype")

    if not isinstance(max_courant, Real) or isinstance(max_courant, bool):
        raise TypeError("max_courant must be in (0, 1]")
    courant_limit = float(max_courant)
    if not math.isfinite(courant_limit) or not 0 < courant_limit <= 1:
        raise ValueError("max_courant must be in (0, 1]")

    if (boundary_echo is None) != (boundary_support is None):
        raise ValueError("boundary_echo and boundary_support must be provided together")
    if boundary_echo is None:
        zero_echo_edges = _zero_edges(echo, height, width)
        zero_support_edges = _zero_edges(support, height, width)
        echo_edges: _BoundaryStages = (zero_echo_edges, zero_echo_edges)
        support_edges: _BoundaryStages = (zero_support_edges, zero_support_edges)
    else:
        echo_edges = _validate_edges(
            "boundary_echo", boundary_echo, height=height, width=width, dtype=echo.dtype
        )
        support_edges = _validate_edges(
            "boundary_support",
            boundary_support,
            height=height,
            width=width,
            dtype=echo.dtype,
        )
        for stage_index, edges in enumerate(support_edges):
            _check_support(f"boundary_support[{stage_index}]", torch.cat(edges))
        for stage_index, edges in enumerate(echo_edges):
            _check_nonnegative(f"boundary_echo[{stage_index}]", torch.cat(edges))
    if reconstruction == "minmod":
        _check_minmod_support(support, qx, qy, support_edges)

    # The volume flux is shared by both RK stages.  Its outward positive part
    # is the positivity/CFL weight, independent of the transported field.
    qx_plus, qx_minus = _volume_parts(qx)
    qy_plus, qy_minus = _volume_parts(qy)
    outgoing_rate = (
        qx_plus[:, 1:]
        - qx_minus[:, :-1]
        + qy_plus[1:, :]
        - qy_minus[:-1, :]
    ) / area
    cfl_tensor = dt_value * outgoing_rate.max()
    if not bool(torch.isfinite(cfl_tensor)):
        raise ValueError("outgoing CFL is not finite")
    cfl_value = cfl_tensor.detach().item()
    effective_courant_limit = (
        min(courant_limit, 0.5) if reconstruction == "minmod" else courant_limit
    )
    if cfl_value > effective_courant_limit:
        raise ValueError(
            f"outgoing CFL {cfl_value} exceeds max_courant "
            f"{effective_courant_limit} for {reconstruction} reconstruction"
        )

    if isinstance(log_growth, Tensor):
        growth_log = _check_tensor("log_growth", log_growth)
        if growth_log.ndim != 0:
            raise ValueError("log_growth tensor must be scalar")
        if growth_log.dtype != echo.dtype:
            growth_log = growth_log.to(dtype=echo.dtype)
    elif isinstance(log_growth, Real) and not isinstance(log_growth, bool):
        growth_value = float(log_growth)
        if not math.isfinite(growth_value):
            raise ValueError("log_growth must be finite")
        growth_log = echo.new_tensor(growth_value)
    else:
        raise TypeError("log_growth must be a finite scalar tensor or float")
    if not bool(torch.isfinite(growth_log)):
        raise ValueError("log_growth must be finite")

    growth_inverse = torch.exp(-growth_log)
    growth = torch.exp(growth_log)
    if not bool(torch.isfinite(growth_inverse)) or not bool(torch.isfinite(growth)):
        raise FloatingPointError("growth exponential is not finite")

    dt = echo.new_tensor(dt_value)
    if not bool(torch.isfinite(dt)) or not bool(dt > 0):
        raise ValueError("dt_seconds is not representable in the input dtype")
    euler = _euler_donorcell if reconstruction == "donorcell" else _euler_minmod
    grown_echo = _scale_by_growth(echo, growth_log)
    _check_tensor("grown echo", grown_echo, dtype=echo.dtype)
    # T(a*q, a*b) = a*T(q, b) for a > 0 for both reconstructions.
    # Amplify before transport to retain tiny incoming contributions; apply
    # decay afterwards so transport can collect them before scaling down.
    if bool(growth_log > 0):
        left, right, bottom, top = echo_edges[0]
        scalable = (
            left <= torch.finfo(echo.dtype).max / growth,
            right <= torch.finfo(echo.dtype).max / growth,
            bottom <= torch.finfo(echo.dtype).max / growth,
            top <= torch.finfo(echo.dtype).max / growth,
        )
        small_edges = (
            torch.where(scalable[0], left, torch.zeros_like(left)),
            torch.where(scalable[1], right, torch.zeros_like(right)),
            torch.where(scalable[2], bottom, torch.zeros_like(bottom)),
            torch.where(scalable[3], top, torch.zeros_like(top)),
        )
        grown_edges = (
            _scale_by_growth(small_edges[0], growth_log),
            _scale_by_growth(small_edges[1], growth_log),
            _scale_by_growth(small_edges[2], growth_log),
            _scale_by_growth(small_edges[3], growth_log),
        )
        stage1 = euler(grown_echo, qx, qy, grown_edges, dt, area)
        if not all(bool(mask.all()) for mask in scalable):
            # Exterior traces enter additively; minmod slopes depend only on
            # interior cells. Transport large traces separately before growth
            # so they cannot force tiny traces to use an underflowing order.
            large_edges = (
                torch.where(scalable[0], torch.zeros_like(left), left),
                torch.where(scalable[1], torch.zeros_like(right), right),
                torch.where(scalable[2], torch.zeros_like(bottom), bottom),
                torch.where(scalable[3], torch.zeros_like(top), top),
            )
            stage1 = stage1 + _scale_by_growth(euler(
                torch.zeros_like(echo), qx, qy, large_edges, dt, area
            ), growth_log)
        transformed_stage1 = stage1
    else:
        transformed_stage1 = euler(echo, qx, qy, echo_edges[0], dt, area)
        stage1 = _scale_by_growth(transformed_stage1, growth_log)
    _check_tensor("stage-1 echo", stage1, dtype=echo.dtype)
    _check_nonnegative("stage-1 echo", stage1)
    stage2 = euler(stage1, qx, qy, echo_edges[1], dt, area)
    _check_tensor("stage-2 echo", stage2, dtype=echo.dtype)
    _check_nonnegative("stage-2 echo", stage2)
    # Both states are nonnegative: their difference cannot overflow, and this
    # form preserves an unchanged subnormal instead of halving it twice.
    echo_new = grown_echo + 0.5 * (stage2 - grown_echo)

    s1 = _euler_donorcell(support, qx, qy, support_edges[0], dt, area)
    _check_tensor("stage-1 support", s1, dtype=echo.dtype)
    _check_support("stage-1 support", s1)
    s2 = _euler_donorcell(s1, qx, qy, support_edges[1], dt, area)
    _check_tensor("stage-2 support", s2, dtype=echo.dtype)
    _check_support("stage-2 support", s2)
    support_new = 0.5 * (support + s2)

    _check_tensor("transport echo result", echo_new, dtype=echo.dtype)
    _check_tensor("transport support result", support_new, dtype=echo.dtype)
    _check_nonnegative("transport echo result", echo_new)
    _check_support("transport support result", support_new)

    in0, out0 = _boundary_budget(qx, qy, echo, echo_edges[0])
    in1, out1 = _boundary_budget(qx, qy, stage1, echo_edges[1])
    # Weight the physical flux before inverse scaling. An inverse-scaled
    # boundary value can overflow even when this integrated budget is finite.
    transport_in = (dt * 0.5) * in0 + ((dt * 0.5) * in1) * growth_inverse
    transport_out = (dt * 0.5) * out0 + ((dt * 0.5) * out1) * growth_inverse
    if bool(growth_log < 0):
        # For decay, inverse-scale tiny traces before flux multiplication.
        # Keep the raw first stage: scaling down then up may have lost it.
        left, right, bottom, top = echo_edges[1]
        scalable = (
            left <= torch.finfo(echo.dtype).max / growth_inverse,
            right <= torch.finfo(echo.dtype).max / growth_inverse,
            bottom <= torch.finfo(echo.dtype).max / growth_inverse,
            top <= torch.finfo(echo.dtype).max / growth_inverse,
        )
        transformed_edges = (
            torch.where(scalable[0], left, torch.zeros_like(left)) * growth_inverse,
            torch.where(scalable[1], right, torch.zeros_like(right)) * growth_inverse,
            torch.where(scalable[2], bottom, torch.zeros_like(bottom)) * growth_inverse,
            torch.where(scalable[3], top, torch.zeros_like(top)) * growth_inverse,
        )
        transformed_in1, transformed_out1 = _boundary_budget(
            qx, qy, transformed_stage1, transformed_edges
        )
        if bool(torch.isfinite(transformed_in1)):
            transport_in = dt * (in0 + 0.5 * (transformed_in1 - in0))
            if not all(bool(mask.all()) for mask in scalable):
                large_edges = (
                    torch.where(scalable[0], torch.zeros_like(left), left),
                    torch.where(scalable[1], torch.zeros_like(right), right),
                    torch.where(scalable[2], torch.zeros_like(bottom), bottom),
                    torch.where(scalable[3], torch.zeros_like(top), top),
                )
                large_in1, _ = _boundary_budget(qx, qy, torch.zeros_like(echo), large_edges)
                transport_in = transport_in + ((dt * 0.5) * large_in1) * growth_inverse
        if bool(torch.isfinite(transformed_out1)):
            transport_out = dt * (out0 + 0.5 * (transformed_out1 - out0))
    physical_in = dt * (in0 + 0.5 * (in1 - in0))
    physical_out = dt * (out0 + 0.5 * (out1 - out0))
    mass0 = echo.sum() * area
    mass1_physical = stage1.sum() * area
    source = growth_log * 0.5 * (mass0 + mass1_physical)
    for name, value in (
        ("transport_inflow", transport_in),
        ("transport_outflow", transport_out),
        ("physical_inflow", physical_in),
        ("physical_outflow", physical_out),
        ("source_quadrature", source),
    ):
        if not bool(torch.isfinite(value)):
            raise FloatingPointError(f"{name} is not finite")

    return FVStepResult(
        echo=echo_new,
        support=support_new,
        transport_inflow=transport_in,
        transport_outflow=transport_out,
        physical_inflow=physical_in,
        physical_outflow=physical_out,
        source_quadrature=source,
    )


def _validate_psi_basis(
    psi_coefficients: Tensor,
    psi_basis: Tensor,
    *,
    height: int,
    width: int,
) -> None:
    """Validate the fixed, gauge-anchored basis outside the AD path."""

    expected = (psi_coefficients.shape[0], height + 1, width + 1)
    if psi_basis.shape != expected:
        raise ValueError(f"psi_basis must have shape {expected}")
    if psi_basis.requires_grad:
        raise ValueError("psi_basis is fixed and must not require gradients")
    if bool(torch.any(psi_basis[:, 0, 0] != 0)):
        raise ValueError("every psi_basis slice must be anchored at [0, 0] == 0")

    # Normalize each basis slice before the rank test so a valid basis is not
    # rejected merely because its coordinates have different physical scales.
    # The basis is fixed data, so this diagnostic must not become part of the
    # differentiable trajectory.
    count = psi_coefficients.shape[0]
    if count == 0:
        return
    with torch.no_grad():
        flattened = psi_basis.reshape(count, -1).mT
        scales = flattened.abs().amax(dim=0)
        if bool(torch.any(~torch.isfinite(scales)) or torch.any(scales <= 0)):
            raise ValueError("psi_basis slices must be nonzero and finite")
        normalized = flattened / scales
        singular_values = torch.linalg.svdvals(normalized)
        largest = singular_values[0]
        tolerance = max(normalized.shape) * torch.finfo(psi_basis.dtype).eps * largest
        rank = int(torch.count_nonzero(singular_values > tolerance))
        if rank != count:
            raise ValueError("psi_basis slices must have full column rank")


def bounded_fv_coefficients(
    control: Tensor,
    *,
    psi_basis: Tensor,
    coefficient_limits: Tensor,
    dt_seconds: float,
    spacing_yx: tuple[float, float],
    reconstruction: str = "minmod",
    max_courant: float = 0.5,
) -> Tensor:
    """Map unconstrained controls to a fixed CFL-safe coefficient box.

    For divergence-free fluxes, outward flow is half the sum of absolute
    face fluxes. The triangle inequality bounds it over |alpha_j| <= L_j.
    A vertex-magnitude allowance also covers cancellation when forming fluxes
    from the combined streamfunction. The step's actual CFL check still applies.
    Geometry and limits are fixed; only ``control`` is differentiated.
    """
    control = _check_tensor("control", control)
    psi_basis = _check_tensor("psi_basis", psi_basis, dtype=control.dtype)
    limits = _check_tensor("coefficient_limits", coefficient_limits, dtype=control.dtype)
    if control.ndim != 1 or limits.shape != control.shape:
        raise ValueError("control and coefficient_limits must have the same vector shape")
    if limits.requires_grad or bool((limits <= 0).any()):
        raise ValueError("coefficient_limits must be fixed and strictly positive")
    if psi_basis.ndim != 3 or min(psi_basis.shape[1:]) < 2:
        raise ValueError("psi_basis must have shape [K-1, H+1, W+1]")
    _validate_psi_basis(control, psi_basis, height=psi_basis.shape[1]-1, width=psi_basis.shape[2]-1)
    if isinstance(dt_seconds, bool) or not isinstance(dt_seconds, Real) or not math.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("dt_seconds must be finite and positive")
    if not isinstance(spacing_yx, (tuple, list)) or len(spacing_yx) != 2:
        raise ValueError("spacing_yx must be (dy, dx)")
    if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) or x <= 0 for x in spacing_yx):
        raise ValueError("spacing_yx must contain finite positive floats")
    if reconstruction not in ("donorcell", "minmod"):
        raise ValueError("reconstruction must be 'donorcell' or 'minmod'")
    if isinstance(max_courant, bool) or not isinstance(max_courant, Real) or not 0 < max_courant <= 1:
        raise ValueError("max_courant must be in (0, 1]")
    cap = min(max_courant, .5) if reconstruction == "minmod" else max_courant
    with torch.no_grad():
        area = control.new_tensor(spacing_yx[0] * spacing_yx[1])
        dt = control.new_tensor(dt_seconds)
        if not bool(torch.isfinite(area) & (area > 0) & torch.isfinite(dt) & (dt > 0)):
            raise ValueError("cell area and timestep must be representable")
        weight = limits[:, None, None]
        fx = ((psi_basis[:, 1:, :] - psi_basis[:, :-1, :]).abs() * weight).sum(0)
        fy = ((psi_basis[:, :, 1:] - psi_basis[:, :, :-1]).abs() * weight).sum(0)
        vertices = (psi_basis.abs() * weight).sum(0)
        rounding = 8 * (control.numel() + 1) * torch.finfo(control.dtype).eps
        ex = rounding * (vertices[1:, :] + vertices[:-1, :])
        ey = rounding * (vertices[:, 1:] + vertices[:, :-1])
        outward = .5 * (fx[:, 1:] + fx[:, :-1] + fy[1:, :] + fy[:-1, :])
        outward = outward + ex[:, 1:] + ex[:, :-1] + ey[1:, :] + ey[:-1, :]
        bound = dt * (outward / area).amax()
        margin = 64 * max(1, control.numel()) * torch.finfo(control.dtype).eps
        if not bool(torch.isfinite(bound)) or margin >= 1 or bool(bound > cap * (1-margin)):
            raise ValueError("coefficient_limits exceed the fixed FV CFL domain")
    return limits * torch.tanh(control)


def _validate_boundary_schedule(
    name: str,
    schedule: object,
    *,
    steps: int,
    height: int,
    width: int,
    dtype: torch.dtype,
) -> BoundarySchedule:
    if not isinstance(schedule, (tuple, list)):
        raise TypeError(f"{name} must be a sequence of stage edge tuples")
    if len(schedule) != steps:
        raise ValueError(f"{name} must have length {steps}")
    return tuple(
        _validate_edges(
            f"{name}[{step}]",
            stages,
            height=height,
            width=width,
            dtype=dtype,
        )
        for step, stages in enumerate(schedule)
    )


def _flatten_step_boundaries(
    echo_schedule: BoundarySchedule,
    support_schedule: BoundarySchedule,
    *,
    start: int,
    stop: int,
) -> tuple[Tensor, ...]:
    """Pack one interval's stage edges in the replay block's tensor order."""

    packed = []
    for step in range(start, stop):
        for stages in (echo_schedule[step], support_schedule[step]):
            for edges in stages:
                packed.extend(edges)
    return tuple(packed)


def finite_volume_trajectory(
    initial_echo: Tensor,
    initial_support: Tensor,
    psi_coefficients: Tensor,
    log_growth_per_interval: Tensor,
    *,
    psi_basis: Tensor,
    leads: int,
    substeps_per_interval: int,
    interval_seconds: float,
    spacing_yx: tuple[float, float],
    boundary_echo: BoundarySchedule,
    boundary_support: BoundarySchedule,
    reconstruction: str = "minmod",
    max_courant: float = 0.5,
    replay: bool = False,
) -> tuple[Tensor, Tensor]:
    """Advance a stationary-ψ FV trajectory and return lead-boundary frames.

    ψ and its face fluxes are constructed once.  Each interval is then a
    tensor-only block so replay mode can retain the complete AD dependency on
    the initial echo, ψ controls, growth, and every supplied boundary trace.
    Boundary schedules contain one pair of SSPRK2 stage traces per substep.
    """

    initial_echo = _check_tensor("initial_echo", initial_echo)
    initial_support = _check_tensor(
        "initial_support", initial_support, dtype=initial_echo.dtype
    )
    psi_coefficients = _check_tensor(
        "psi_coefficients", psi_coefficients, dtype=initial_echo.dtype
    )
    log_growth_per_interval = _check_tensor(
        "log_growth_per_interval",
        log_growth_per_interval,
        dtype=initial_echo.dtype,
    )
    if initial_echo.ndim != 2 or initial_echo.shape[0] == 0 or initial_echo.shape[1] == 0:
        raise ValueError("initial_echo must be a non-empty [H, W] grid")
    if initial_support.shape != initial_echo.shape:
        raise ValueError("initial_support must have the same shape as initial_echo")
    if psi_coefficients.ndim != 1:
        raise ValueError("psi_coefficients must have shape [K-1]")
    if log_growth_per_interval.ndim != 0:
        raise ValueError("log_growth_per_interval must be a scalar tensor")
    _check_nonnegative("initial_echo", initial_echo)
    _check_support("initial_support", initial_support)

    if isinstance(leads, bool) or not isinstance(leads, Integral):
        raise TypeError("leads must be a positive integer")
    if isinstance(substeps_per_interval, bool) or not isinstance(
        substeps_per_interval, Integral
    ):
        raise TypeError("substeps_per_interval must be a positive integer")
    leads = int(leads)
    substeps_per_interval = int(substeps_per_interval)
    if leads <= 0:
        raise ValueError("leads must be a positive integer")
    if substeps_per_interval <= 0:
        raise ValueError("substeps_per_interval must be a positive integer")
    if not isinstance(replay, bool):
        raise TypeError("replay must be a bool")
    if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, Real):
        raise TypeError("interval_seconds must be a positive finite float")
    interval_value = float(interval_seconds)
    if not math.isfinite(interval_value) or interval_value <= 0:
        raise ValueError("interval_seconds must be a positive finite float")
    if not isinstance(spacing_yx, (tuple, list)) or len(spacing_yx) != 2:
        raise ValueError("spacing_yx must be (dy, dx)")
    spacing_values = []
    for spacing in spacing_yx:
        if isinstance(spacing, bool) or not isinstance(spacing, Real):
            raise TypeError("spacing_yx must contain finite positive floats")
        spacing_value = float(spacing)
        if not math.isfinite(spacing_value) or spacing_value <= 0:
            raise ValueError("spacing_yx must contain finite positive floats")
        spacing_values.append(spacing_value)
    # Freeze caller-provided mutable schedule metadata before any replay.
    spacing_yx = (spacing_values[0], spacing_values[1])

    height, width = initial_echo.shape
    psi_basis = _check_tensor("psi_basis", psi_basis, dtype=initial_echo.dtype)
    _validate_psi_basis(
        psi_coefficients,
        psi_basis,
        height=height,
        width=width,
    )
    echo_schedule = _validate_boundary_schedule(
        "boundary_echo",
        boundary_echo,
        steps=leads * substeps_per_interval,
        height=height,
        width=width,
        dtype=initial_echo.dtype,
    )
    support_schedule = _validate_boundary_schedule(
        "boundary_support",
        boundary_support,
        steps=leads * substeps_per_interval,
        height=height,
        width=width,
        dtype=initial_echo.dtype,
    )

    psi = torch.einsum("k,kij->ij", psi_coefficients, psi_basis)
    qx, qy = face_volume_fluxes(psi)
    dt_seconds = interval_value / substeps_per_interval

    def interval_block(
        state: Tensor,
        block_qx: Tensor,
        block_qy: Tensor,
        block_growth: Tensor,
        *boundary_tensors: Tensor,
    ) -> Tensor:
        if not boundary_tensors or len(boundary_tensors) % 16:
            raise RuntimeError("trajectory interval boundary tensor count mismatch")
        block_steps = len(boundary_tensors) // 16
        echo, support = state.unbind(0)
        growth_per_step = block_growth / substeps_per_interval
        for substep in range(block_steps):
            offset = 16 * substep
            echo_edges = (
                tuple(boundary_tensors[offset : offset + 4]),
                tuple(boundary_tensors[offset + 4 : offset + 8]),
            )
            support_offset = offset + 8
            support_edges = (
                tuple(boundary_tensors[support_offset : support_offset + 4]),
                tuple(boundary_tensors[support_offset + 4 : support_offset + 8]),
            )
            result = finite_volume_step(
                echo,
                support,
                block_qx,
                block_qy,
                dt_seconds=dt_seconds,
                spacing_yx=spacing_yx,
                log_growth=growth_per_step,
                boundary_echo=echo_edges,
                boundary_support=support_edges,
                max_courant=max_courant,
                reconstruction=reconstruction,
            )
            echo, support = result.echo, result.support
        return torch.stack((echo, support))

    state = torch.stack((initial_echo, initial_support))
    echo_frames = [initial_echo]
    support_frames = [initial_support]
    for lead in range(leads):
        start = lead * substeps_per_interval
        stop = start + substeps_per_interval
        # Bound the differentiable replay tape independently of interval length.
        # Splitting does not alter a substep, its stage traces, or growth/dt.
        block_size = min(8, substeps_per_interval) if replay else substeps_per_interval
        for block_start in range(start, stop, block_size):
            boundary_tensors = _flatten_step_boundaries(
                echo_schedule, support_schedule, start=block_start,
                stop=min(block_start + block_size, stop),
            )
            if replay:
                state = cast(Tensor, recompute(
                    interval_block, state, qx, qy, log_growth_per_interval,
                    *boundary_tensors,
                ))
            else:
                state = interval_block(
                    state, qx, qy, log_growth_per_interval, *boundary_tensors,
                )
        echo_frames.append(state[0])
        support_frames.append(state[1])

    return torch.stack(echo_frames), torch.stack(support_frames)
