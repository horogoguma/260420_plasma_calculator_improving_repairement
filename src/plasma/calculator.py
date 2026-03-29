"""Plasma calculation primitives and default reactor conditions."""

from dataclasses import dataclass
from math import pi
import math
from pydoc import resolve

TORR_TO_PA = 133.32236842105263
MM_TO_M = 1e-3


@dataclass(frozen=True)
class BasicConstants:
    """Physical constants used by the plasma model."""

    boltzmann_constant: float = 1.38065e-23
    vacuum_permittivity: float = 8.85e-12
    electron_charge: float = 1.6e-19
    electron_mass: float = 9.1095e-31
    argon_mass: float = 6.6335e-26
    excitation_energy_ev: float = 12.14
    ionization_energy_ev: float = 15.76


@dataclass(frozen=True)
class ChamberConditions:
    """Geometry and neutral-gas conditions for the reactor."""

    chamber_height_m: float = 5.679328897 * MM_TO_M
    chamber_radius_m: float = 238.438997 * MM_TO_M
    pressure_torr: float = 3.5
    temperature_k: float = 423.0

    @property
    def chamber_height_mm(self) -> float:
        return self.chamber_height_m / MM_TO_M

    @property
    def chamber_radius_mm(self) -> float:
        return self.chamber_radius_m / MM_TO_M

    @property
    def chamber_volume_m3(self) -> float:
        return pi * (self.chamber_radius_m**2) * self.chamber_height_m

    @property
    def pressure_pa(self) -> float:
        return self.pressure_torr * TORR_TO_PA


@dataclass
class PlasmaConditions:
    """Plasma-side operating conditions."""

    electron_temperature_ev: float = 2.0
    sheath_voltage: float = 100.0
    RF_power: float = 1000.0
    RF_frequency: int = 12_900_000


class PlasmaCalculator:
    """Container for plasma-side calculations."""

    def __init__(
        self,
        gas: str = "argon",
        constants: BasicConstants | None = None,
        chamber: ChamberConditions | None = None,
        plasma_conditions: PlasmaConditions | None = None,
    ) -> None:
        self.gas = gas
        self.constants = constants if constants is not None else BasicConstants()
        self.chamber = chamber if chamber is not None else ChamberConditions()
        self.plasma_conditions = (
            plasma_conditions if plasma_conditions is not None else PlasmaConditions()
        )

    @property
    def electron_temperature_ev(self) -> float:
        return self.plasma_conditions.electron_temperature_ev

    @electron_temperature_ev.setter
    def electron_temperature_ev(self, value: float) -> None:
        self.plasma_conditions.electron_temperature_ev = value

    @property
    def sheath_voltage(self) -> float:
        return self.plasma_conditions.sheath_voltage

    @sheath_voltage.setter
    def sheath_voltage(self, value: float) -> None:
        self.plasma_conditions.sheath_voltage = value

    @property
    def RF_power(self) -> float:
        return self.plasma_conditions.RF_power

    @RF_power.setter
    def RF_power(self, value: float) -> None:
        self.plasma_conditions.RF_power = value

    @property
    def RF_frequency(self) -> int:
        return self.plasma_conditions.RF_frequency

    @RF_frequency.setter
    def RF_frequency(self, value: int) -> None:
        self.plasma_conditions.RF_frequency = value

    def _resolved_electron_temperature_ev(
        self,
        electron_temperature_ev: float | None,
    ) -> float:
        return (
            self.electron_temperature_ev
            if electron_temperature_ev is None
            else electron_temperature_ev
        )

    def _resolved_sheath_voltage(self, sheath_voltage: float | None) -> float:
        return self.sheath_voltage if sheath_voltage is None else sheath_voltage

    def _resolved_rf_power(self, rf_power: float | None) -> float:
        return self.RF_power if rf_power is None else rf_power
    
    def _resolved_rf_frequency(self, rf_frequency: int | None) -> int:
        return self.RF_frequency if rf_frequency is None else rf_frequency

    def _resolved_volume_m3(self, volume_m3: float | None) -> float:
        return self.chamber.chamber_volume_m3 if volume_m3 is None else volume_m3

    def _resolved_pressure_torr(self, pressure_torr: float | None) -> float:
        return self.chamber.pressure_torr if pressure_torr is None else pressure_torr

    def _resolved_pressure_pa(self, pressure_pa: float | None) -> float:
        return self.chamber.pressure_pa if pressure_pa is None else pressure_pa

    def _resolved_temperature_k(self, temperature_k: float | None) -> float:
        return self.chamber.temperature_k if temperature_k is None else temperature_k

    def _resolved_chamber_radius_m(self, chamber_radius_m: float | None) -> float:
        return (
            self.chamber.chamber_radius_m
            if chamber_radius_m is None
            else chamber_radius_m
        )

    def _resolved_chamber_height_m(self, chamber_height_m: float | None) -> float:
        return (
            self.chamber.chamber_height_m
            if chamber_height_m is None
            else chamber_height_m
        )

    def compute_impedance(self, voltage: complex, current: complex) -> complex:
        """Return impedance from voltage and current."""
        if current == 0:
            raise ValueError("Current must be non-zero.")
        return voltage / current

    def compute_power_density(self, power: float, volume: float | None = None) -> float:
        """Return power density using the supplied or default chamber volume."""
        resolved_volume = self._resolved_volume_m3(volume)
        if resolved_volume == 0:
            raise ValueError("Volume must be non-zero.")
        return power / resolved_volume

    def compute_ion_mean_free_path(self, pressure_torr: float | None = None) -> float:
        """Return ion mean free path from chamber pressure in torr."""
        resolved_pressure = self._resolved_pressure_torr(pressure_torr)
        if resolved_pressure == 0:
            raise ValueError("Chamber pressure must be non-zero.")
        return resolved_pressure / 330

    def compute_gas_number_density(
        self,
        pressure_pa: float | None = None,
        temperature_k: float | None = None,
    ) -> float:
        """Return gas number density from pressure in Pa and temperature in K."""
        resolved_pressure_pa = self._resolved_pressure_pa(pressure_pa)
        resolved_temperature_k = self._resolved_temperature_k(temperature_k)
        return resolved_pressure_pa / (
            self.constants.boltzmann_constant * resolved_temperature_k
        )

    def compute_effective_area(
        self,
        chamber_radius_m: float | None = None,
        chamber_height_m: float | None = None,
    ) -> float:
        """Return effective area from chamber radius in m."""
        resolved_radius = self._resolved_chamber_radius_m(chamber_radius_m)
        resolved_height = self._resolved_chamber_height_m(chamber_height_m)
        if resolved_radius == 0:
            raise ValueError("Chamber radius must be non-zero.")
        return (
            2 * pi * resolved_radius * resolved_radius * 0.61
            + 2 * pi * resolved_radius * resolved_height * 0.61
        )

    def compute_effective_length(
        self,
        chamber_radius_m: float | None = None,
        chamber_height_m: float | None = None,
    ) -> float:
        """Return effective length from chamber height in m."""
        resolved_radius = self._resolved_chamber_radius_m(chamber_radius_m)
        resolved_height = self._resolved_chamber_height_m(chamber_height_m)
        if resolved_height == 0:
            raise ValueError("Chamber height must be non-zero.")
        return (
            pi * resolved_radius * resolved_radius * resolved_height
        ) / self.compute_effective_area(resolved_radius, resolved_height)

    def compute_elastic_collision_constant(
        self,
        electron_temperature_ev: float | None = None,
    ) -> float:
        """Return elastic collision constant."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        return (
            0.00000000000002336
            * (resolved_electron_temperature_ev**1.609)
            * math.exp(
                0.0618 * (math.log(resolved_electron_temperature_ev) ** 2)
                - 0.117 * (math.log(resolved_electron_temperature_ev) ** 3)
            )
        )

    def compute_exitation_constant(
        self,
        electron_temperature_ev: float | None = None,
    ) -> float:
        """Return excitation collision constant."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        return (
            0.0000000000000248
            * (resolved_electron_temperature_ev**0.33)
            * math.exp(-12.78 / resolved_electron_temperature_ev)
        )

    def compute_ionization_constant(
        self,
        electron_temperature_ev: float | None = None,
    ) -> float:
        """Return ionization collision constant."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        return (
            0.0000000000000234
            * (resolved_electron_temperature_ev**0.59)
            * math.exp(-17.44 / resolved_electron_temperature_ev)
        )

    def compute_number_need_to_be_one(
        self,
        electron_temperature_ev: float | None = None,
        pressure_pa: float | None = None,
        temperature_k: float | None = None,
        chamber_radius_m: float | None = None,
        chamber_height_m: float | None = None,
    ) -> float:
        """Return the target ratio that should approach one."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        resolved_pressure_pa = self._resolved_pressure_pa(pressure_pa)
        resolved_temperature_k = self._resolved_temperature_k(temperature_k)
        resolved_chamber_radius_m = self._resolved_chamber_radius_m(chamber_radius_m)
        resolved_chamber_height_m = self._resolved_chamber_height_m(chamber_height_m)

        if resolved_electron_temperature_ev <= 0:
            raise ValueError("electron_temperature_ev must be positive.")
        if (
            resolved_electron_temperature_ev < 0.1
            or resolved_electron_temperature_ev > 100
        ):
            raise ValueError("electron_temperature_ev is out of allowed range.")

        return (
            self.compute_ionization_constant(resolved_electron_temperature_ev)
            * self.compute_gas_number_density(
                resolved_pressure_pa,
                resolved_temperature_k,
            )
            * self.compute_effective_length(
                resolved_chamber_radius_m,
                resolved_chamber_height_m,
            )
        ) / self.compute_bohm_velocity(resolved_electron_temperature_ev)

    def solve_electron_temperature(
        self,
        target: float = 1.0,
        start_ev: float | None = None,
        step_ev: float = 0.01,
        tolerance: float = 0.01,
        max_iterations: int = 5000,
        pressure_pa: float | None = None,
        temperature_k: float | None = None,
        chamber_radius_m: float | None = None,
        chamber_height_m: float | None = None,
    ) -> tuple[float, int, float]:
        """Iteratively solve for the electron temperature that matches the target."""
        electron_temperature_ev = self._resolved_electron_temperature_ev(start_ev)
        last_value = 0.0

        for iteration in range(max_iterations):
            last_value = self.compute_number_need_to_be_one(
                electron_temperature_ev=electron_temperature_ev,
                pressure_pa=pressure_pa,
                temperature_k=temperature_k,
                chamber_radius_m=chamber_radius_m,
                chamber_height_m=chamber_height_m,
            )
            error = last_value - target
            if abs(error) < tolerance:
                return electron_temperature_ev, iteration, last_value
            if error > 0:
                electron_temperature_ev -= step_ev
            else:
                electron_temperature_ev += step_ev

        return electron_temperature_ev, max_iterations, last_value

    def compute_bohm_velocity(
        self,
        electron_temperature_ev: float | None = None,
    ) -> float:
        """Return Bohm velocity from electron temperature in eV."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        return (
            self.constants.electron_charge
            * resolved_electron_temperature_ev
            / self.constants.argon_mass
        ) ** 0.5

    def compute_collision_energy_loss(
        self,
        electron_temperature_ev: float | None = None,
    ) -> float:
        """Return electron collision energy loss from electron temperature in eV."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        return (
            (
                self.compute_elastic_collision_constant(resolved_electron_temperature_ev)
                * (3 * self.constants.electron_mass / self.constants.argon_mass)
                * resolved_electron_temperature_ev
                + self.compute_exitation_constant(resolved_electron_temperature_ev)
                * self.constants.excitation_energy_ev
                + self.compute_ionization_constant(resolved_electron_temperature_ev)
                * self.constants.ionization_energy_ev
            )
            / self.compute_ionization_constant(resolved_electron_temperature_ev)
        )

    def compute_electron_ion_energy_loss(
        self,
        electron_temperature_ev: float | None = None,
        sheath_voltage: float | None = None,
    ) -> float:
        """Return electron-ion energy loss from electron temperature in eV."""
        resolved_sheath_voltage = self._resolved_sheath_voltage(sheath_voltage)
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        return (
            resolved_sheath_voltage
            + (0.5 * resolved_electron_temperature_ev)
            + (2 * resolved_electron_temperature_ev)
        )

    def compute_total_energy_loss(
        self,
        electron_temperature_ev: float | None = None,
        sheath_voltage: float | None = None,
    ) -> float:
        """Return total energy loss from electron temperature in eV."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        resolved_sheath_voltage = self._resolved_sheath_voltage(sheath_voltage)
        return self.compute_collision_energy_loss(
            resolved_electron_temperature_ev
        ) + self.compute_electron_ion_energy_loss(
            resolved_electron_temperature_ev,
            resolved_sheath_voltage,
        )

    def compute_plasma_density(
        self,
        electron_temperature_ev: float | None = None,
        RF_power: float | None = None,
        sheath_voltage: float | None = None,
        chamber_radius_m: float | None = None,
        chamber_height_m: float | None = None,
    ) -> float:
        """Return plasma density from excitation and ionization energies."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        resolved_rf_power = self._resolved_rf_power(RF_power)
        resolved_sheath_voltage = self._resolved_sheath_voltage(sheath_voltage)
        resolved_chamber_radius_m = self._resolved_chamber_radius_m(chamber_radius_m)
        resolved_chamber_height_m = self._resolved_chamber_height_m(chamber_height_m)
        return resolved_rf_power / (
            self.constants.electron_charge
            * self.compute_bohm_velocity(resolved_electron_temperature_ev)
            * self.compute_effective_area(
                resolved_chamber_radius_m,
                resolved_chamber_height_m,
            )
            * self.compute_total_energy_loss(
                resolved_electron_temperature_ev,
                resolved_sheath_voltage,
            )
        )

    def compute_collisional_frequency(
        self,
        electron_temperature_ev: float | None = None,
        pressure_pa: float | None = None,
    ) -> float:
        """Return collisional frequency from electron temperature in eV and pressure in Pa."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        resolved_pressure_pa = self._resolved_pressure_pa(pressure_pa)
        return self.compute_gas_number_density(
            resolved_pressure_pa,
            self.chamber.temperature_k,
        ) * self.compute_elastic_collision_constant(resolved_electron_temperature_ev)

    def compute_plasma_angular_frequency(
        self,
        electron_temperature_ev: float | None = None,
        RF_power: float | None = None,
        sheath_voltage: float | None = None,
        chamber_radius_m: float | None = None,
        chamber_height_m: float | None = None,
    ) -> float:
        """Return plasma angular frequency in rad/s."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        resolved_rf_power = self._resolved_rf_power(RF_power)
        resolved_sheath_voltage = self._resolved_sheath_voltage(sheath_voltage)
        resolved_chamber_radius_m = self._resolved_chamber_radius_m(chamber_radius_m)
        resolved_chamber_height_m = self._resolved_chamber_height_m(chamber_height_m)
        return (
            (self.compute_plasma_density(
                resolved_electron_temperature_ev,
                resolved_rf_power,
                resolved_sheath_voltage,
                resolved_chamber_radius_m,
                resolved_chamber_height_m,
            )
            * self.constants.electron_charge
            * self.constants.electron_charge
        ) / (self.constants.electron_mass * self.constants.vacuum_permittivity)
        ) ** 0.5
    
    def compute_plasma_conductivity(
        self,
        electron_temperature_ev: float | None = None,
        RF_power: float | None = None,
        RF_frequency: int | None = None,
        sheath_voltage: float | None = None,
        chamber_radius_m: float | None = None,
        chamber_height_m: float | None = None,
        pressure_pa: float | None = None,
    ) -> float:
        """Return plasma angular frequency in rad/s."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        resolved_rf_power = self._resolved_rf_power(RF_power)
        resolved_rf_frequency = self._resolved_rf_frequency(RF_frequency)
        resolved_sheath_voltage = self._resolved_sheath_voltage(sheath_voltage)
        resolved_chamber_radius_m = self._resolved_chamber_radius_m(chamber_radius_m)
        resolved_chamber_height_m = self._resolved_chamber_height_m(chamber_height_m)
        resolved_pressure_pa = self._resolved_pressure_pa(pressure_pa)
        return (
            (   self.constants.vacuum_permittivity * self.compute_collisional_frequency(
                resolved_electron_temperature_ev,
                resolved_pressure_pa)
                * (self.compute_plasma_angular_frequency(
                    resolved_electron_temperature_ev,
                    resolved_rf_power,
                    resolved_sheath_voltage,
                    resolved_chamber_radius_m,
                    resolved_chamber_height_m,
                    )
                ) ** (2)                
            )
            /
            (  
                (
                    resolved_rf_frequency ** 2
                ) 
            +   (
                    (self.compute_collisional_frequency(
                    resolved_electron_temperature_ev,
                    resolved_pressure_pa
                    )) ** (2)
                )
            )
        )
    
    
    def compute_plasma_relative_permittivity(
        self,
        electron_temperature_ev: float | None = None,
        RF_power: float | None = None,
        RF_frequency: int | None = None,
        sheath_voltage: float | None = None,
        chamber_radius_m: float | None = None,
        chamber_height_m: float | None = None,
        pressure_pa: float | None = None,
    ) -> float:
        """Return plasma relative permittivity."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        resolved_rf_power = self._resolved_rf_power(RF_power)
        resolved_rf_frequency = self._resolved_rf_frequency(RF_frequency)
        resolved_sheath_voltage = self._resolved_sheath_voltage(sheath_voltage)
        resolved_chamber_radius_m = self._resolved_chamber_radius_m(chamber_radius_m)
        resolved_chamber_height_m = self._resolved_chamber_height_m(chamber_height_m)
        resolved_pressure_pa = self._resolved_pressure_pa(pressure_pa)
        plasma_angular_frequency = self.compute_plasma_angular_frequency(
            resolved_electron_temperature_ev,
            resolved_rf_power,
            resolved_sheath_voltage,
            resolved_chamber_radius_m,
            resolved_chamber_height_m,
        )
        collisional_frequency = self.compute_collisional_frequency(
            resolved_electron_temperature_ev,
            resolved_pressure_pa,
        )
        return (
                    - (
                        (   
                            (self.compute_plasma_angular_frequency(
                                resolved_electron_temperature_ev,
                                resolved_rf_power,
                                resolved_sheath_voltage,
                                resolved_chamber_radius_m,
                                resolved_chamber_height_m,
                                )
                            ) ** (2)  
                        )
                        /
                        (  
                            (
                                (resolved_rf_frequency * 2 * pi) ** 2
                            ) 
                            + (
                                (self.compute_collisional_frequency(
                                resolved_electron_temperature_ev,
                                resolved_pressure_pa
                                )) ** 2
                            )
                        )
                    )
                )
    
    def compute_debye_length_m(
        self,
        electron_temperature_ev: float | None = None,
        RF_power: float | None = None,
        sheath_voltage: float | None = None,
        chamber_radius_m: float | None = None,
        chamber_height_m: float | None = None,
    ) -> float:
        """Return Debye length from plasma density and electron temperature."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        resolved_rf_power = self._resolved_rf_power(RF_power)
        resolved_sheath_voltage = self._resolved_sheath_voltage(sheath_voltage)
        resolved_chamber_radius_m = self._resolved_chamber_radius_m(chamber_radius_m)
        resolved_chamber_height_m = self._resolved_chamber_height_m(chamber_height_m)
        plasma_density = self.compute_plasma_density(
            resolved_electron_temperature_ev,
            resolved_rf_power,
            resolved_sheath_voltage,
            resolved_chamber_radius_m,
            resolved_chamber_height_m,
        )
        return (
            self.constants.vacuum_permittivity
            * self.constants.boltzmann_constant
            * 11600
            * resolved_electron_temperature_ev
            / (plasma_density * self.constants.electron_charge * self.constants.electron_charge)
        ) ** 0.5

    def compute_plasma_resistance(
        self,
        electron_temperature_ev: float | None = None,
        RF_power: float | None = None,
        sheath_voltage: float | None = None,
        chamber_radius_m: float | None = None,
        chamber_height_m: float | None = None,
        pressure_pa: float | None = None,
    ) -> float:
        """Return plasma resistance from plasma conductivity and effective length."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        resolved_rf_power = self._resolved_rf_power(RF_power)
        resolved_sheath_voltage = self._resolved_sheath_voltage(sheath_voltage)
        resolved_chamber_radius_m = self._resolved_chamber_radius_m(chamber_radius_m)
        resolved_chamber_height_m = self._resolved_chamber_height_m(chamber_height_m)
        resolved_pressure_pa = self._resolved_pressure_pa(pressure_pa)
        plasma_conductivity = self.compute_plasma_conductivity(
            resolved_electron_temperature_ev,
            resolved_rf_power,
            self.RF_frequency,
            resolved_sheath_voltage,
            resolved_chamber_radius_m,
            resolved_chamber_height_m,
            resolved_pressure_pa,
        )
        return resolved_chamber_radius_m / (
            plasma_conductivity * pi * resolved_chamber_radius_m * resolved_chamber_radius_m
        )
    
    def compute_plasma_coil_reactance(
        self,
        electron_temperature_ev: float | None = None,
        RF_power: float | None = None,
        RF_frequency: int | None = None,
        sheath_voltage: float | None = None,
        chamber_radius_m: float | None = None,
        chamber_height_m: float | None = None,
        pressure_pa: float | None = None,
    ) -> float:
        """Return plasma resistance from plasma conductivity and effective length."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        resolved_rf_power = self._resolved_rf_power(RF_power)
        resolved_rf_frequency = self._resolved_rf_frequency(RF_frequency)
        resolved_sheath_voltage = self._resolved_sheath_voltage(sheath_voltage)
        resolved_chamber_radius_m = self._resolved_chamber_radius_m(chamber_radius_m)
        resolved_chamber_height_m = self._resolved_chamber_height_m(chamber_height_m)
        resolved_pressure_pa = self._resolved_pressure_pa(pressure_pa)
        plasma_relative_permittivity = self.compute_plasma_relative_permittivity(
            resolved_electron_temperature_ev,
            resolved_rf_power,
            self.RF_frequency,
            resolved_sheath_voltage,
            resolved_chamber_radius_m,
            resolved_chamber_height_m,
            resolved_pressure_pa,
        )
        return -1 * resolved_chamber_radius_m / (
            (2 * pi * resolved_rf_frequency) * self.constants.vacuum_permittivity * plasma_relative_permittivity * pi * resolved_chamber_radius_m * resolved_chamber_radius_m
        )
    
    def compute_plasma_cap_reactance(
        self,
        electron_temperature_ev: float | None = None,
        RF_power: float | None = None,
        RF_frequency: int | None = None,
        sheath_voltage: float | None = None,
        chamber_radius_m: float | None = None,
        chamber_height_m: float | None = None,
    ) -> float:
        """Return plasma resistance from plasma conductivity and effective length."""
        resolved_electron_temperature_ev = self._resolved_electron_temperature_ev(
            electron_temperature_ev
        )
        resolved_rf_power = self._resolved_rf_power(RF_power)
        resolved_rf_frequency = self._resolved_rf_frequency(RF_frequency)
        resolved_sheath_voltage = self._resolved_sheath_voltage(sheath_voltage)
        resolved_chamber_radius_m = self._resolved_chamber_radius_m(chamber_radius_m)
        resolved_chamber_height_m = self._resolved_chamber_height_m(chamber_height_m)
        
        return -1 * resolved_chamber_radius_m / (
            (2 * pi * resolved_rf_frequency) * self.constants.vacuum_permittivity * pi * resolved_chamber_radius_m * resolved_chamber_radius_m
        )