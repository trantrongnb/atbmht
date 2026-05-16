"""Fallback shim for environments without the external `ipdb` package."""

from __future__ import annotations


def set_trace(*args, **kwargs):
    """Degrade gracefully to the built-in breakpoint when ipdb is unavailable."""
    breakpoint()
