"""간단한 PySpice 래퍼 클래스

이 클래스는 회로를 구성하고 스파이스 분석을 수행해서
전력, 전류 같은 값을 반환하는 간단한 API를 제공한다.
나중에 GUI나 플라즈마 계산 쪽에서 이 객체를 호출하게 된다.
"""

# Base.py를 import 하면 환경변수가 셋업된다
# (PySpice가 동작하는 데 필요한 ngspice 경로 설정)
# 현재 패키지 구조에서는 src.Base를 불러오면 된다.
from .. import Base  # noqa: F401

import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

logger = Logging.setup_logging()


class SpiceSimulator:
    def __init__(self):
        # 초기 상태의 회로 객체
        self.circuit = None

    def build_rc_lowpass(self, R_value, C_value):
        """저역 필터 회로를 간단히 생성한다.

        Parameters
        ----------
        R_value : ~PySpice.Unit.Quantity
            저항값 (예: 1@u_kOhm)
        C_value : ~PySpice.Unit.Quantity
            커패시터값 (예: 1@u_uF)
        """
        self.circuit = Circuit("Lowpass")
        self.circuit.SinusoidalVoltageSource('input', 'n1', self.circuit.gnd,
                                             amplitude=1@u_V, frequency=100@u_Hz)
        self.R = self.circuit.R(1, 'n1', 'n2', R_value)
        self.C = self.circuit.C(1, 'n2', self.circuit.gnd, C_value)

    def run_ac(self, start_freq, stop_freq, points=100):
        """AC 분석을 수행하고 결과를 반환한다."""
        if self.circuit is None:
            raise ValueError("회로가 정의되어 있지 않습니다.")

        simulator = self.circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.ac(start_frequency=start_freq,
                                stop_frequency=stop_freq,
                                variation='dec',
                                number_of_points=points)
        return analysis

    # 추후 전력/전류 계산 도우미 메서드를 추가할 수 있다.
    def compute_power(self, node_voltage, node_current):
        """전압과 전류를 입력받아 전력을 계산."""
        return node_voltage * node_current