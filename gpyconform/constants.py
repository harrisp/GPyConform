#!/usr/bin/env python3

"""
Fixed-point scaling and numeric thresholds used across gpyconform.
"""

from typing import Final

# Number of decimal places supported for confidence levels and corresponding scale
# for avoiding precision issues
_CONF_DP: Final[int]   = 6
_CONF_SCALE: Final[int] = 10 ** _CONF_DP

# Treat gamma >= 1e8 as "effectively infinite"
_GAMMA_INF_THRESHOLD = 1e8