"""Path helpers for FOHO-local third_party dependencies."""

from __future__ import annotations

import os


def foho_root() -> str:
    # src/foho/configs -> src/foho -> src -> FOHO root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def third_party_root() -> str:
    return os.path.join(foho_root(), "third_party")

