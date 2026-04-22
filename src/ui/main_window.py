"""Main Qt window for running a single plasma simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
)

from src.app import FixedInputs, format_simulation_result, run_single_simulation

UI_FILENAME = "plasma_calculator.ui"

INPUT_FIELD_NAMES = {
    "chamber_height_mm": "chamberHeightMmEdit",
    "chamber_radius_mm": "chamberRadiusMmEdit",
    "pressure_torr": "pressureTorrEdit",
    "temperature_k": "temperatureKEdit",
    "electrode_radius_mm": "electrodeRadiusMmEdit",
    "electron_temperature_ev": "initialElectronTemperatureEvEdit",
    "sheath_voltage": "initialSheathVoltageEdit",
    "sheath_length_electrode_mm": "initialSheathLengthElectrodeMmEdit",
    "sheath_length_grounded_mm": "initialSheathLengthGroundedMmEdit",
    "rf_power": "rfPowerEdit",
    "rf_frequency": "rfFrequencyEdit",
}


class PlasmaCalculatorWindow:
    """Qt main window loaded from Designer UI."""

    def __init__(self) -> None:
        self._input_fields: dict[str, QLineEdit] = {}
        self._result_view: QPlainTextEdit
        self._window: QMainWindow
        self._build_ui()
        self._wire_events()
        self._populate_defaults()

    def _build_ui(self) -> None:
        self._window = self._load_ui()

        self._result_view = self._require_child(QPlainTextEdit, "resultTextEdit")
        self._set_label_text("inputsLabel", "Inputs")
        self._set_label_text("resultsTitleLabel", "Results")

        for field_name, object_name in INPUT_FIELD_NAMES.items():
            self._input_fields[field_name] = self._require_child(QLineEdit, object_name)

    def _load_ui(self) -> QMainWindow:
        ui_path = Path(__file__).with_name(UI_FILENAME)
        ui_file = QFile(str(ui_path))
        if not ui_file.open(QFile.ReadOnly):
            raise RuntimeError(f"Unable to open UI file: {ui_path}")

        try:
            loaded = QUiLoader().load(ui_file)
        finally:
            ui_file.close()

        if loaded is None or not isinstance(loaded, QMainWindow):
            raise RuntimeError(f"Unable to load main window UI: {ui_path}")
        return loaded

    def show(self) -> None:
        """Show the loaded Qt main window."""
        self._window.show()

    def _require_child(self, widget_type: type[Any], object_name: str) -> Any:
        widget = self._window.findChild(widget_type, object_name)
        if widget is None:
            raise RuntimeError(f"Missing widget '{object_name}' in {UI_FILENAME}.")
        return widget

    def _set_label_text(self, object_name: str, text: str) -> None:
        label = self._window.findChild(QLabel, object_name)
        if label is not None:
            label.setText(text)

    def _wire_events(self) -> None:
        run_button = self._require_child(QPushButton, "runCalculationButton")
        reset_button = self._require_child(QPushButton, "resetDefaultsButton")
        run_button.clicked.connect(self._run_simulation)
        reset_button.clicked.connect(self._populate_defaults)

    def _populate_defaults(self) -> None:
        defaults = FixedInputs()
        for field_name, value in defaults.__dict__.items():
            self._input_fields[field_name].setText(str(value))
        self._result_view.clear()

    def _collect_inputs(self) -> FixedInputs:
        values: dict[str, Any] = {}
        for field_name, widget in self._input_fields.items():
            text = widget.text().strip()
            if not text:
                raise ValueError(f"{field_name} cannot be empty.")
            values[field_name] = float(text)
        return FixedInputs(**values)

    def _run_simulation(self) -> None:
        try:
            inputs = self._collect_inputs()
            result = run_single_simulation(inputs)
        except Exception as exc:
            QMessageBox.critical(self._window, "Simulation failed", str(exc))
            return

        self._result_view.setPlainText(format_simulation_result(result))
