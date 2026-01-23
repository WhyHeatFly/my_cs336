"""
Windows shim for the Unix-only stdlib `resource` module.

Some course tests import `resource` unconditionally.
On Windows, stdlib has no `resource`, so we provide a minimal subset.

This aims to be good enough for tests that only read ru_maxrss or call
(get/set)rlimit without requiring real OS-enforced limits.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

# --- constants commonly used ---
RUSAGE_SELF = 0

# RLIMIT_* constants don't exist on Windows; provide placeholders
RLIMIT_AS = 0
RLIMIT_DATA = 1
RLIMIT_STACK = 2
RLIMIT_RSS = 3
RLIMIT_NOFILE = 4

# "infinite" limit placeholder
RLIM_INFINITY = (1 << 63) - 1


@dataclass
class _RUsage:
    # On Linux: ru_maxrss is in kilobytes.
    # We'll return KB here to mimic Linux behavior.
    ru_maxrss: int = 0


def _get_rss_bytes() -> int:
    """Best-effort RSS bytes for current process."""
    # Try psutil if available (most accurate/cross-platform)
    try:
        import psutil  # type: ignore
        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass

    # Fallback: not great, but prevents crashing.
    # Use Python's memory allocator stats as rough proxy (NOT RSS).
    try:
        import tracemalloc
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            return int(peak)
    except Exception:
        pass

    return 0


def getrusage(who: int) -> _RUsage:
    # Only support RUSAGE_SELF in this shim
    rss_bytes = _get_rss_bytes()
    rss_kb = rss_bytes // 1024
    return _RUsage(ru_maxrss=rss_kb)


# --- rlimit API (no-op placeholders) ---
# Some tests set limits to prevent runaway memory usage on Unix.
# Windows can't enforce these the same way, but tests often just need the import to work.

_limits = {
    RLIMIT_AS: (RLIM_INFINITY, RLIM_INFINITY),
    RLIMIT_DATA: (RLIM_INFINITY, RLIM_INFINITY),
    RLIMIT_STACK: (RLIM_INFINITY, RLIM_INFINITY),
    RLIMIT_RSS: (RLIM_INFINITY, RLIM_INFINITY),
    RLIMIT_NOFILE: (RLIM_INFINITY, RLIM_INFINITY),
}


def getrlimit(resource: int):
    return _limits.get(resource, (RLIM_INFINITY, RLIM_INFINITY))


def setrlimit(resource: int, limits):
    # Store but do not enforce.
    # limits should be (soft, hard)
    try:
        soft, hard = limits
        _limits[resource] = (soft, hard)
    except Exception:
        # mimic resource.error style
        raise ValueError("invalid limits for setrlimit")
