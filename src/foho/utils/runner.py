"""Helpers for running steps in conda envs."""

from __future__ import annotations

import os
import subprocess
from typing import Dict, Iterable, Optional


def run_in_conda(
    conda_sh: str,
    env_name: str,
    cmd: str,
    cwd: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    full_cmd = f"source {conda_sh} && conda activate {env_name} && {cmd}"
    subprocess.run(["bash", "-lc", full_cmd], check=True, cwd=cwd, env=env)


def join_args(args: Iterable[str]) -> str:
    return " ".join(args)
