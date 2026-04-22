"""PySide GUI entry point."""

from __future__ import annotations

from src.ui import run_gui


def main() -> None:
    """Run the Qt application and exit with its status code."""
    raise SystemExit(run_gui())


if __name__ == "__main__":
    main()
