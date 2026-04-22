"""Application launcher for the PySide-based user interface."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .main_window import PlasmaCalculatorWindow


def run_gui() -> int:
    """Create the Qt application and show the main window."""
    app = QApplication(sys.argv)
    window = PlasmaCalculatorWindow()
    window.show()
    return app.exec()
