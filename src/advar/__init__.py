"""Core three-frame radar echo nowcasting API.

Specialized research, ledger, promotion, intervention, audit, and deployment
types remain available from their owning ``advar.<module>`` namespaces. They
are deliberately not imported into the package startup path.
"""

from .nowcast import NowcastConfig, RadarGridTimeContract, nowcast
from .variational import AnalysisConfig, variational_nowcast

__all__ = [
    "AnalysisConfig",
    "NowcastConfig",
    "RadarGridTimeContract",
    "nowcast",
    "variational_nowcast",
]
