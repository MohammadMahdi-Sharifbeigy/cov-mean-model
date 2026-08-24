"""Command-line parsing and dispatch for the experiment workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config import RunPaths

COMMANDS = ("step1", "step2", "step3", "step4", "step5", "step6", "step7", "all")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one pipeline command and its required artifact directory."""
    parser = argparse.ArgumentParser(
        description="Run one Mean--Covariance experiment step or all steps."
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing this run's artifacts and checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML experiment configuration (default: RUN_DIR/config.yaml).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments then invoke the workflow without executing on import."""
    args = parse_args(argv)
    from .workflow import dispatch, load_run_config

    paths = RunPaths.from_root(args.run_dir)
    dispatch(args.command, paths, load_run_config(paths, args.config))
    return 0
