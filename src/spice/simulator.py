"""간단한 PySpice 래퍼 클래스

이 클래스는 회로를 구성하고 스파이스 분석을 수행해서
전력, 전류 같은 값을 반환하는 간단한 API를 제공한다.
나중에 GUI나 플라즈마 계산 쪽에서 이 객체를 호출하게 된다.
"""

# Base.py를 import 하면 환경변수가 셋업된다
# (PySpice가 동작하는 데 필요한 ngspice 경로 설정)
# 현재 패키지 구조에서는 src.Base를 불러오면 된다.
from .. import Base  # noqa: F401

from dataclasses import dataclass
from math import pi

import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

logger = Logging.setup_logging()


@dataclass(frozen=True)
class PlasmaCircuitParameters:
    """플라즈마 등가회로를 구성하는 lumped parameters."""

    plasma_resistance: float
    plasma_coil_henry: float
    plasma_cap_farad: float
    plasma_sheath_capacitance_electrode: float
    plasma_sheath_capacitance_grounded: float
    plasma_sheath_resistance_electrode: float
    plasma_sheath_resistance_grounded: float
    rf_frequency_hz: float


@dataclass(frozen=True)
class PlasmaCircuitResult:
    """주파수 한 점에서 계산한 플라즈마 등가회로 결과."""

    angular_frequency: float
    target_power_w: float
    source_voltage_rms: complex
    source_voltage_peak: complex
    source_current_rms: complex
    src_node_current_rms: complex
    src_node_resistor_current_rms: complex
    src_node_capacitor_current_rms: complex
    electrode_sheath_impedance: complex
    bulk_plasma_impedance: complex
    grounded_sheath_impedance: complex
    total_impedance: complex
    electrode_sheath_resistor_power_w: float
    plasma_resistance_power_w: float
    grounded_sheath_resistor_power_w: float
    total_resistor_power_w: float
    average_power_w: float


class SpiceSimulator:
    def __init__(self):
        # 초기 상태의 회로 객체
        self.circuit = None
        self._plasma_params = None
        self._target_power_w = None
        self._source_voltage_peak_v = None

    def build_plasma_equivalent_circuit(
        self,
        params: PlasmaCircuitParameters,
        target_power_w: float,
    ):
        """플라즈마 bulk + sheath 등가회로를 생성한다.

        토폴로지:
        source -> electrode sheath(R||C) -> bulk plasma(R-L-C 직렬)
        -> grounded sheath(R||C) -> ground
        """
        self._plasma_params = params
        self._target_power_w = target_power_w
        equivalent = self._compute_equivalent_impedances(params)
        source_voltage_rms_v = self._solve_source_voltage_rms(
            target_power_w,
            equivalent["total_impedance"],
        )
        source_voltage_peak_v = source_voltage_rms_v * (2 ** 0.5)
        self._source_voltage_peak_v = source_voltage_peak_v
        self.circuit = Circuit("PlasmaEquivalent")
        self.circuit.SinusoidalVoltageSource(
            "input",
            "src",
            self.circuit.gnd,
            amplitude=source_voltage_peak_v @ u_V,
            frequency=params.rf_frequency_hz @ u_Hz,
        )

        self.circuit.R(
            "sheath_e",
            "src",
            "node_e",
            params.plasma_sheath_resistance_electrode @ u_Ohm,
        )
        self.circuit.C(
            "sheath_e",
            "src",
            "node_e",
            params.plasma_sheath_capacitance_electrode @ u_F,
        )
        self.circuit.R(
            "bulk_r",
            "node_e",
            "node_l",
            params.plasma_resistance @ u_Ohm,
        )
        self.circuit.L(
            "bulk_l",
            "node_l",
            "node_g",
            abs(params.plasma_coil_henry) @ u_H,
        )
        self.circuit.C(
            "bulk_c",
            "node_e",
            "node_g",
            params.plasma_cap_farad @ u_F,
        )
        self.circuit.R(
            "sheath_g",
            "node_g",
            self.circuit.gnd,
            params.plasma_sheath_resistance_grounded @ u_Ohm,
        )
        self.circuit.C(
            "sheath_g",
            "node_g",
            self.circuit.gnd,
            params.plasma_sheath_capacitance_grounded @ u_F,
        )

    def compute_plasma_circuit_response(self) -> PlasmaCircuitResult:
        """구성된 플라즈마 등가회로에서 전압, 전류, 전력을 계산한다."""
        if self.circuit is None or self._plasma_params is None:
            raise ValueError("Plasma equivalent circuit must be built first.")
        if self._source_voltage_peak_v is None:
            raise ValueError("Source voltage must be set before analysis.")
        if self._target_power_w is None:
            raise ValueError("Target power must be set before analysis.")

        params = self._plasma_params
        equivalent = self._compute_equivalent_impedances(params)
        source_voltage_peak = complex(self._source_voltage_peak_v, 0.0)
        source_voltage_rms = source_voltage_peak / (2 ** 0.5)
        total_impedance = equivalent["total_impedance"]

        if total_impedance == 0:
            raise ValueError("Total plasma impedance must be non-zero.")

        source_current_rms = source_voltage_rms / total_impedance
        bulk_voltage_rms = source_current_rms * equivalent["bulk_plasma_impedance"]
        grounded_sheath_voltage_rms = (
            source_current_rms * equivalent["grounded_sheath_impedance"]
        )
        bulk_series_impedance = (
            complex(params.plasma_resistance, 0.0)
            + self._inductive_impedance(
                params.plasma_coil_henry,
                equivalent["angular_frequency"],
            )
        )
        electrode_sheath_voltage_rms = (
            source_current_rms * equivalent["electrode_sheath_impedance"]
        )
        src_node_resistor_current_rms = electrode_sheath_voltage_rms / complex(
            params.plasma_sheath_resistance_electrode,
            0.0,
        )
        src_node_capacitor_current_rms = electrode_sheath_voltage_rms / (
            self._capacitive_impedance(
                params.plasma_sheath_capacitance_electrode,
                equivalent["angular_frequency"],
            )
        )
        src_node_current_rms = (
            src_node_resistor_current_rms + src_node_capacitor_current_rms
        )
        bulk_series_current_rms = bulk_voltage_rms / bulk_series_impedance
        grounded_sheath_resistor_current_rms = grounded_sheath_voltage_rms / complex(
            params.plasma_sheath_resistance_grounded,
            0.0,
        )
        electrode_sheath_resistor_power_w = (
            electrode_sheath_voltage_rms * src_node_resistor_current_rms.conjugate()
        ).real
        plasma_resistance_power_w = (
            (
                bulk_series_current_rms * complex(params.plasma_resistance, 0.0)
            ) * bulk_series_current_rms.conjugate()
        ).real
        grounded_sheath_resistor_power_w = (
            grounded_sheath_voltage_rms
            * grounded_sheath_resistor_current_rms.conjugate()
        ).real
        total_resistor_power_w = (
            electrode_sheath_resistor_power_w
            + plasma_resistance_power_w
            + grounded_sheath_resistor_power_w
        )
        average_power = (source_voltage_rms * source_current_rms.conjugate()).real

        return PlasmaCircuitResult(
            angular_frequency=equivalent["angular_frequency"],
            target_power_w=self._target_power_w,
            source_voltage_rms=source_voltage_rms,
            source_voltage_peak=source_voltage_peak,
            source_current_rms=source_current_rms,
            src_node_current_rms=src_node_current_rms,
            src_node_resistor_current_rms=src_node_resistor_current_rms,
            src_node_capacitor_current_rms=src_node_capacitor_current_rms,
            electrode_sheath_impedance=equivalent["electrode_sheath_impedance"],
            bulk_plasma_impedance=equivalent["bulk_plasma_impedance"],
            grounded_sheath_impedance=equivalent["grounded_sheath_impedance"],
            total_impedance=total_impedance,
            electrode_sheath_resistor_power_w=electrode_sheath_resistor_power_w,
            plasma_resistance_power_w=plasma_resistance_power_w,
            grounded_sheath_resistor_power_w=grounded_sheath_resistor_power_w,
            total_resistor_power_w=total_resistor_power_w,
            average_power_w=average_power,
        )

    def _compute_equivalent_impedances(
        self,
        params: PlasmaCircuitParameters,
    ) -> dict[str, float | complex]:
        omega = 2 * pi * params.rf_frequency_hz
        electrode_sheath_impedance = self._parallel_impedance(
            params.plasma_sheath_resistance_electrode,
            self._capacitive_impedance(
                params.plasma_sheath_capacitance_electrode,
                omega,
            ),
        )
        bulk_series_impedance = (
            complex(params.plasma_resistance, 0.0)
            + self._inductive_impedance(params.plasma_coil_henry, omega)
        )
        bulk_capacitive_impedance = self._capacitive_impedance(
            params.plasma_cap_farad,
            omega,
        )
        bulk_plasma_impedance = self._parallel_impedance(
            bulk_series_impedance,
            bulk_capacitive_impedance,
        )
        grounded_sheath_impedance = self._parallel_impedance(
            params.plasma_sheath_resistance_grounded,
            self._capacitive_impedance(
                params.plasma_sheath_capacitance_grounded,
                omega,
            ),
        )
        total_impedance = (
            electrode_sheath_impedance
            + bulk_plasma_impedance
            + grounded_sheath_impedance
        )
        return {
            "angular_frequency": omega,
            "electrode_sheath_impedance": electrode_sheath_impedance,
            "bulk_plasma_impedance": bulk_plasma_impedance,
            "grounded_sheath_impedance": grounded_sheath_impedance,
            "total_impedance": total_impedance,
        }

    def _solve_source_voltage_rms(
        self,
        target_power_w: float,
        total_impedance: complex,
    ) -> float:
        if target_power_w <= 0:
            raise ValueError("Target power must be positive.")
        if total_impedance == 0:
            raise ValueError("Total plasma impedance must be non-zero.")

        admittance = 1 / complex(total_impedance)
        conductance = admittance.real
        if conductance <= 0:
            raise ValueError("Total plasma circuit must dissipate positive real power.")
        return (target_power_w / conductance) ** 0.5

    def _parallel_impedance(self, z1: float | complex, z2: float | complex) -> complex:
        z1_complex = complex(z1)
        z2_complex = complex(z2)
        if z1_complex == 0 or z2_complex == 0:
            return complex(0.0, 0.0)
        return 1 / ((1 / z1_complex) + (1 / z2_complex))

    def _capacitive_impedance(self, capacitance_f: float, omega: float) -> complex:
        if capacitance_f <= 0:
            raise ValueError("Capacitance must be positive.")
        if omega <= 0:
            raise ValueError("Angular frequency must be positive.")
        return -1j / (omega * capacitance_f)

    def _inductive_impedance(self, inductance_h: float, omega: float) -> complex:
        if omega <= 0:
            raise ValueError("Angular frequency must be positive.")
        return 1j * omega * inductance_h
