"""Plasma calculation primitives and default reactor conditions."""

from dataclasses import dataclass
from math import pi
import math

from .constants import MM_TO_M, TORR_TO_PA


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
    electrode_area_m2: float | None = None
    grounded_area_m2: float | None = None

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
    sheath_length_electrode_m: float = 1.0354 * MM_TO_M
    sheath_length_grounded_m: float = 1.0354 * MM_TO_M
    RF_power: float = 1000.0
    RF_frequency: int = 12_900_000
    Current_density: float = 100.0


@dataclass(frozen=True)
class PlasmaComputationResult:
    """Aggregated plasma calculation outputs for a single operating point."""

    electron_temperature_ev: float
    electron_temperature_iterations: int
    electron_temperature_target_value: float
    number_need_to_be_one: float
    elastic_collision_constant: float
    excitation_constant: float
    ionization_constant: float
    bohm_velocity: float
    gas_number_density: float
    effective_length: float
    collision_energy_loss: float
    electron_ion_energy_loss: float
    total_energy_loss: float
    plasma_density: float
    ion_mean_free_path_m: float
    collisional_frequency: float
    plasma_angular_frequency: float
    plasma_conductivity: float
    plasma_relative_permittivity: float
    debye_length_m: float
    plasma_resistance: float
    plasma_coil_reactance: float
    plasma_cap_reactance: float
    plasma_coil_henry: float
    plasma_cap_farad: float
    plasma_sheath_capacitance: float
    plasma_sheath_capacitance_electrode: float
    plasma_sheath_capacitance_grounded: float
    electron_velocity: float
    plasma_sheath_conductance: float
    plasma_sheath_resistance_electrode: float
    plasma_sheath_resistance_grounded: float


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

    def compute_plasma_properties(
        self,
        chamber: ChamberConditions | None = None,
        plasma_conditions: PlasmaConditions | None = None,
    ) -> PlasmaComputationResult:
        """Compute the plasma-side operating point from the provided conditions."""
        if chamber is not None:
            self.chamber = chamber
        if plasma_conditions is not None:
            self.plasma_conditions = plasma_conditions

        chamber = self.chamber
        conditions = self.plasma_conditions

        pressure_torr = chamber.pressure_torr
        pressure_pa = chamber.pressure_pa
        temperature_k = chamber.temperature_k
        chamber_radius_m = chamber.chamber_radius_m
        chamber_height_m = chamber.chamber_height_m
        sheath_voltage = conditions.sheath_voltage
        rf_power = conditions.RF_power
        rf_frequency = conditions.RF_frequency
        
        sheath_length_electrode_m = conditions.sheath_length_electrode_m
        sheath_length_grounded_m = conditions.sheath_length_grounded_m
        
        # Calculate bulk plasma height
        bulk_height_m = self.compute_bulk_plasma_height(
            chamber_height_m,
            sheath_length_electrode_m,
            sheath_length_grounded_m,
        )

        electron_temperature_ev, iterations, target_value = (
            self.solve_electron_temperature(
                start_ev=conditions.electron_temperature_ev,
                pressure_pa=pressure_pa,
                temperature_k=temperature_k,
                chamber_radius_m=chamber_radius_m,
                chamber_height_m=chamber_height_m,
            )
        )
        self.electron_temperature_ev = electron_temperature_ev

        number_need_to_be_one = self.compute_number_need_to_be_one(
            electron_temperature_ev=electron_temperature_ev,
            pressure_pa=pressure_pa,
            temperature_k=temperature_k,
            chamber_radius_m=chamber_radius_m,
            chamber_height_m=bulk_height_m,
        )
        elastic_collision_constant = self.compute_elastic_collision_constant(
            electron_temperature_ev=electron_temperature_ev,
        )
        excitation_constant = self.compute_exitation_constant(
            electron_temperature_ev=electron_temperature_ev,
        )
        ionization_constant = self.compute_ionization_constant(
            electron_temperature_ev=electron_temperature_ev,
        )
        bohm_velocity = self.compute_bohm_velocity(
            electron_temperature_ev=electron_temperature_ev,
        )
        gas_number_density = self.compute_gas_number_density(
            pressure_pa=pressure_pa,
            temperature_k=temperature_k,
        )
        effective_length = self.compute_effective_length(
            chamber_radius_m=chamber_radius_m,
            chamber_height_m=bulk_height_m,
        )
        collision_energy_loss = self.compute_collision_energy_loss(
            electron_temperature_ev=electron_temperature_ev,
        )
        electron_ion_energy_loss = self.compute_electron_ion_energy_loss(
            electron_temperature_ev=electron_temperature_ev,
            sheath_voltage=sheath_voltage,
        )
        total_energy_loss = self.compute_total_energy_loss(
            electron_temperature_ev=electron_temperature_ev,
            sheath_voltage=sheath_voltage,
        )
        plasma_density = self.compute_plasma_density(
            electron_temperature_ev=electron_temperature_ev,
            RF_power=rf_power,
            sheath_voltage=sheath_voltage,
            chamber_radius_m=chamber_radius_m,
            chamber_height_m=bulk_height_m,
        )
        ion_mean_free_path_m = self.compute_ion_mean_free_path_m(
            pressure_torr=pressure_torr,
        )
        collisional_frequency = self.compute_collisional_frequency(
            electron_temperature_ev=electron_temperature_ev,
            pressure_pa=pressure_pa,
            temperature_k=temperature_k,
        )
        plasma_angular_frequency = self.compute_plasma_angular_frequency(
            electron_temperature_ev=electron_temperature_ev,
            RF_power=rf_power,
            sheath_voltage=sheath_voltage,
            chamber_radius_m=chamber_radius_m,
            chamber_height_m=bulk_height_m,
        )
        plasma_conductivity = self.compute_plasma_conductivity(
            electron_temperature_ev=electron_temperature_ev,
            RF_power=rf_power,
            RF_frequency=rf_frequency,
            sheath_voltage=sheath_voltage,
            chamber_radius_m=chamber_radius_m,
            chamber_height_m=bulk_height_m,
            pressure_pa=pressure_pa,
            temperature_k=temperature_k,
        )
        plasma_relative_permittivity = self.compute_plasma_relative_permittivity(
            electron_temperature_ev=electron_temperature_ev,
            RF_power=rf_power,
            RF_frequency=rf_frequency,
            sheath_voltage=sheath_voltage,
            chamber_radius_m=chamber_radius_m,
            chamber_height_m=bulk_height_m,
            pressure_pa=pressure_pa,
            temperature_k=temperature_k,
        )
        debye_length_m = self.compute_debye_length_m(
            electron_temperature_ev=electron_temperature_ev,
            RF_power=rf_power,
            sheath_voltage=sheath_voltage,
            chamber_radius_m=chamber_radius_m,
            chamber_height_m=bulk_height_m,
        )

        plasma_resistance = self.compute_plasma_resistance(
            electron_temperature_ev=electron_temperature_ev,
            RF_power=rf_power,
            RF_frequency=rf_frequency,
            sheath_voltage=sheath_voltage,
            chamber_radius_m=chamber_radius_m,
            chamber_height_m=chamber_height_m,
            pressure_pa=pressure_pa,
            temperature_k=temperature_k,
            sheath_length_electrode_m=sheath_length_electrode_m,
            sheath_length_grounded_m=sheath_length_grounded_m,
        )
        plasma_coil_reactance = self.compute_plasma_coil_reactance(
            electron_temperature_ev=electron_temperature_ev,
            RF_power=rf_power,
            RF_frequency=rf_frequency,
            sheath_voltage=sheath_voltage,
            chamber_radius_m=chamber_radius_m,
            chamber_height_m=chamber_height_m,
            pressure_pa=pressure_pa,
            temperature_k=temperature_k,
            sheath_length_electrode_m=sheath_length_electrode_m,
            sheath_length_grounded_m=sheath_length_grounded_m,
        )
        plasma_cap_reactance = self.compute_plasma_cap_reactance(
            RF_frequency=rf_frequency,
            chamber_radius_m=chamber_radius_m,
            chamber_height_m=chamber_height_m,
            sheath_length_electrode_m=sheath_length_electrode_m,
            sheath_length_grounded_m=sheath_length_grounded_m,
        )
        plasma_coil_henry = self.compute_plasma_coil_henry(
            electron_temperature_ev=electron_temperature_ev,
            RF_power=rf_power,
            RF_frequency=rf_frequency,
            sheath_voltage=sheath_voltage,
            chamber_radius_m=chamber_radius_m,
            chamber_height_m=chamber_height_m,
            pressure_pa=pressure_pa,
            temperature_k=temperature_k,
            sheath_length_electrode_m=sheath_length_electrode_m,
            sheath_length_grounded_m=sheath_length_grounded_m,
        )
        plasma_cap_farad = self.compute_plasma_cap_farad(
            chamber_radius_m=chamber_radius_m,
            chamber_height_m=chamber_height_m,
            sheath_length_electrode_m=sheath_length_electrode_m,
            sheath_length_grounded_m=sheath_length_grounded_m,
        )
        plasma_sheath_capacitance = self.compute_plasma_sheath_capacitance(
            sheath_thickness_m=sheath_length_electrode_m,
        )
        plasma_sheath_capacitance_electrode = (
            self.compute_plasma_sheath_capacitance_electrode(
                sheath_thickness_m=sheath_length_electrode_m,
                chamber_radius_m=chamber_radius_m,
            )
        )
        plasma_sheath_capacitance_grounded = (
            self.compute_plasma_sheath_capacitance_grounded(
                sheath_thickness_m=sheath_length_grounded_m,
                chamber_radius_m=chamber_radius_m,
                chamber_height_m=chamber_height_m,
            )
        )
        electron_velocity = self.compute_electron_velocity(
            electron_temperature_ev=electron_temperature_ev,
        )
        plasma_sheath_conductance = self.compute_plasma_sheath_conductance(
            sheath_thickness_m=sheath_length_electrode_m,
            pressure_torr=pressure_torr,
            electron_temperature_ev=electron_temperature_ev,
            RF_power=rf_power,
            sheath_voltage=sheath_voltage,
            chamber_radius_m=chamber_radius_m,
            chamber_height_m=chamber_height_m,
        )
        plasma_sheath_resistance_electrode = (
            self.compute_plasma_sheath_resistance_electrode(
                sheath_thickness_m=sheath_length_electrode_m,
                pressure_torr=pressure_torr,
                electron_temperature_ev=electron_temperature_ev,
                RF_power=rf_power,
                sheath_voltage=sheath_voltage,
                chamber_radius_m=chamber_radius_m,
                chamber_height_m=chamber_height_m,
            )
        )
        plasma_sheath_resistance_grounded = (
            self.compute_plasma_sheath_resistance_grounded(
                sheath_thickness_m=sheath_length_grounded_m,
                pressure_torr=pressure_torr,
                electron_temperature_ev=electron_temperature_ev,
                RF_power=rf_power,
                sheath_voltage=sheath_voltage,
                chamber_radius_m=chamber_radius_m,
                chamber_height_m=chamber_height_m,
            )
        )

        return PlasmaComputationResult(
            electron_temperature_ev=electron_temperature_ev,
            electron_temperature_iterations=iterations,
            electron_temperature_target_value=target_value,
            number_need_to_be_one=number_need_to_be_one,
            elastic_collision_constant=elastic_collision_constant,
            excitation_constant=excitation_constant,
            ionization_constant=ionization_constant,
            bohm_velocity=bohm_velocity,
            gas_number_density=gas_number_density,
            effective_length=effective_length,
            collision_energy_loss=collision_energy_loss,
            electron_ion_energy_loss=electron_ion_energy_loss,
            total_energy_loss=total_energy_loss,
            plasma_density=plasma_density,
            ion_mean_free_path_m=ion_mean_free_path_m,
            collisional_frequency=collisional_frequency,
            plasma_angular_frequency=plasma_angular_frequency,
            plasma_conductivity=plasma_conductivity,
            plasma_relative_permittivity=plasma_relative_permittivity,
            debye_length_m=debye_length_m,
            plasma_resistance=plasma_resistance,
            plasma_coil_reactance=plasma_coil_reactance,
            plasma_cap_reactance=plasma_cap_reactance,
            plasma_coil_henry=plasma_coil_henry,
            plasma_cap_farad=plasma_cap_farad,
            plasma_sheath_capacitance=plasma_sheath_capacitance,
            plasma_sheath_capacitance_electrode=plasma_sheath_capacitance_electrode,
            plasma_sheath_capacitance_grounded=plasma_sheath_capacitance_grounded,
            electron_velocity=electron_velocity,
            plasma_sheath_conductance=plasma_sheath_conductance,
            plasma_sheath_resistance_electrode=plasma_sheath_resistance_electrode,
            plasma_sheath_resistance_grounded=plasma_sheath_resistance_grounded,
        )

    def compute_impedance(self, voltage: complex, current: complex) -> complex:
        """Return impedance from voltage and current."""
        if current == 0:
            raise ValueError("Current must be non-zero.")
        return voltage / current

    def compute_power_density(self, power: float, volume: float) -> float:
        """Return power density."""
        if volume == 0:
            raise ValueError("Volume must be non-zero.")
        return power / volume

    def compute_ion_mean_free_path_m(self, pressure_torr: float) -> float:
        """Return ion mean free path from chamber pressure in torr."""
        if pressure_torr == 0:
            raise ValueError("Chamber pressure must be non-zero.")
        return (1 / (pressure_torr * 330)) / 100

    def compute_gas_number_density(
        self,
        pressure_pa: float,
        temperature_k: float,
    ) -> float:
        """Return gas number density from pressure in Pa and temperature in K."""
        return pressure_pa / (self.constants.boltzmann_constant * temperature_k)

    def compute_effective_area(
        self,
        chamber_radius_m: float,
        chamber_height_m: float,
    ) -> float:
        """Return effective area from chamber radius in m."""
        if chamber_radius_m == 0:
            raise ValueError("Chamber radius must be non-zero.")
        return (
            2 * pi * chamber_radius_m * chamber_radius_m * 0.61
            + 2 * pi * chamber_radius_m * chamber_height_m * 0.61
        )

    def compute_effective_length(
        self,
        chamber_radius_m: float,
        chamber_height_m: float,
    ) -> float:
        """Return effective length from chamber height in m."""
        if chamber_height_m == 0:
            raise ValueError("Chamber height must be non-zero.")
        return (
            pi * chamber_radius_m * chamber_radius_m * chamber_height_m
        ) / self.compute_effective_area(chamber_radius_m, chamber_height_m)

    def compute_elastic_collision_constant(
        self,
        electron_temperature_ev: float,
    ) -> float:
        """Return elastic collision constant."""
        return (
            0.00000000000002336
            * (electron_temperature_ev**1.609)
            * math.exp(
                0.0618 * (math.log(electron_temperature_ev) ** 2)
                - 0.117 * (math.log(electron_temperature_ev) ** 3)
            )
        )

    def compute_exitation_constant(
        self,
        electron_temperature_ev: float,
    ) -> float:
        """Return excitation collision constant."""
        return (
            0.0000000000000248
            * (electron_temperature_ev**0.33)
            * math.exp(-12.78 / electron_temperature_ev)
        )

    def compute_ionization_constant(
        self,
        electron_temperature_ev: float,
    ) -> float:
        """Return ionization collision constant."""
        return (
            0.0000000000000234
            * (electron_temperature_ev**0.59)
            * math.exp(-17.44 / electron_temperature_ev)
        )

    def compute_number_need_to_be_one(
        self,
        electron_temperature_ev: float,
        pressure_pa: float,
        temperature_k: float,
        chamber_radius_m: float,
        chamber_height_m: float,
    ) -> float:
        """Return the target ratio that should approach one."""
        if electron_temperature_ev <= 0:
            raise ValueError("electron_temperature_ev must be positive.")
        if electron_temperature_ev < 0.1 or electron_temperature_ev > 100:
            raise ValueError("electron_temperature_ev is out of allowed range.")

        return (
            self.compute_ionization_constant(electron_temperature_ev)
            * self.compute_gas_number_density(pressure_pa, temperature_k)
            * self.compute_effective_length(chamber_radius_m, chamber_height_m)
        ) / self.compute_bohm_velocity(electron_temperature_ev)

    def solve_electron_temperature(
        self,
        start_ev: float,
        pressure_pa: float,
        temperature_k: float,
        chamber_radius_m: float,
        chamber_height_m: float,
        target: float = 1.0,
        step_ev: float = 0.01,
        tolerance: float = 0.01,
        max_iterations: int = 5000,
    ) -> tuple[float, int, float]:
        """Iteratively solve for the electron temperature that matches the target."""
        electron_temperature_ev = start_ev
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
        electron_temperature_ev: float,
    ) -> float:
        """Return Bohm velocity from electron temperature in eV."""
        return (
            self.constants.electron_charge
            * electron_temperature_ev
            / self.constants.argon_mass
        ) ** 0.5

    def compute_collision_energy_loss(
        self,
        electron_temperature_ev: float,
    ) -> float:
        """Return electron collision energy loss from electron temperature in eV."""
        return (
            (
                self.compute_elastic_collision_constant(electron_temperature_ev)
                * (3 * self.constants.electron_mass / self.constants.argon_mass)
                * electron_temperature_ev
                + self.compute_exitation_constant(electron_temperature_ev)
                * self.constants.excitation_energy_ev
                + self.compute_ionization_constant(electron_temperature_ev)
                * self.constants.ionization_energy_ev
            )
            / self.compute_ionization_constant(electron_temperature_ev)
        )

    def compute_electron_ion_energy_loss(
        self,
        electron_temperature_ev: float,
        sheath_voltage: float,
    ) -> float:
        """Return electron-ion energy loss from electron temperature in eV."""
        return (
            sheath_voltage
            + (0.5 * electron_temperature_ev)
            + (2 * electron_temperature_ev)
        )

    def compute_total_energy_loss(
        self,
        electron_temperature_ev: float,
        sheath_voltage: float,
    ) -> float:
        """Return total energy loss from electron temperature in eV."""
        return self.compute_collision_energy_loss(
            electron_temperature_ev
        ) + self.compute_electron_ion_energy_loss(
            electron_temperature_ev,
            sheath_voltage,
        )

    def compute_plasma_density(
        self,
        electron_temperature_ev: float,
        RF_power: float,
        sheath_voltage: float,
        chamber_radius_m: float,
        chamber_height_m: float,
    ) -> float:
        """Return plasma density from excitation and ionization energies."""
        return RF_power / (
            self.constants.electron_charge
            * self.compute_bohm_velocity(electron_temperature_ev)
            * self.compute_effective_area(chamber_radius_m, chamber_height_m)
            * self.compute_total_energy_loss(
                electron_temperature_ev,
                sheath_voltage,
            )
        )

    def compute_collisional_frequency(
        self,
        electron_temperature_ev: float,
        pressure_pa: float,
        temperature_k: float,
    ) -> float:
        """Return collisional frequency from electron temperature in eV and pressure in Pa."""
        return self.compute_gas_number_density(
            pressure_pa,
            temperature_k,
        ) * self.compute_elastic_collision_constant(electron_temperature_ev)

    def compute_plasma_angular_frequency(
        self,
        electron_temperature_ev: float,
        RF_power: float,
        sheath_voltage: float,
        chamber_radius_m: float,
        chamber_height_m: float,
    ) -> float:
        """Return plasma angular frequency in rad/s."""
        return (
            (self.compute_plasma_density(
                electron_temperature_ev,
                RF_power,
                sheath_voltage,
                chamber_radius_m,
                chamber_height_m,
            )
            * self.constants.electron_charge
            * self.constants.electron_charge
        ) / (self.constants.electron_mass * self.constants.vacuum_permittivity)
        ) ** 0.5
    
    def compute_plasma_conductivity(
        self,
        electron_temperature_ev: float,
        RF_power: float,
        RF_frequency: int,
        sheath_voltage: float,
        chamber_radius_m: float,
        chamber_height_m: float,
        pressure_pa: float,
        temperature_k: float,
    ) -> float:
        """Return plasma angular frequency in rad/s."""
        return (
            (self.constants.vacuum_permittivity * self.compute_collisional_frequency(
                electron_temperature_ev,
                pressure_pa,
                temperature_k)
                * (self.compute_plasma_angular_frequency(
                    electron_temperature_ev,
                    RF_power,
                    sheath_voltage,
                    chamber_radius_m,
                    chamber_height_m,
                    )
                ) ** (2)                
            )
            /
            (  
                (
                    RF_frequency ** 2
                ) 
            +   (
                    (self.compute_collisional_frequency(
                    electron_temperature_ev,
                    pressure_pa,
                    temperature_k
                    )) ** (2)
                )
            )
        )
    
    
    def compute_plasma_relative_permittivity(
        self,
        electron_temperature_ev: float,
        RF_power: float,
        RF_frequency: int,
        sheath_voltage: float,
        chamber_radius_m: float,
        chamber_height_m: float,
        pressure_pa: float,
        temperature_k: float,
    ) -> float:
        """Return plasma relative permittivity."""
        plasma_angular_frequency = self.compute_plasma_angular_frequency(
            electron_temperature_ev,
            RF_power,
            sheath_voltage,
            chamber_radius_m,
            chamber_height_m,
        )
        collisional_frequency = self.compute_collisional_frequency(
            electron_temperature_ev,
            pressure_pa,
            temperature_k,
        )
        return (
            -((plasma_angular_frequency**2) / (((RF_frequency * 2 * pi) ** 2) + (collisional_frequency**2)))
        )
    
    def compute_debye_length_m(
        self,
        electron_temperature_ev: float,
        RF_power: float,
        sheath_voltage: float,
        chamber_radius_m: float,
        chamber_height_m: float,
    ) -> float:
        """Return Debye length from plasma density and electron temperature."""
        plasma_density = self.compute_plasma_density(
            electron_temperature_ev,
            RF_power,
            sheath_voltage,
            chamber_radius_m,
            chamber_height_m,
        )
        return (
            self.constants.vacuum_permittivity
            * self.constants.boltzmann_constant
            * 11600
            * electron_temperature_ev
            / (plasma_density * self.constants.electron_charge * self.constants.electron_charge)
        ) ** 0.5

    def compute_bulk_plasma_height(
        self,
        chamber_height_m: float,
        sheath_length_electrode_m: float,
        sheath_length_grounded_m: float,
    ) -> float:
        """Return the bulk plasma height inside the chamber.

        The bulk plasma region is chamber height minus the sheath regions
        at both the electrode and the grounded surfaces.
        """
        if chamber_height_m <= 0:
            raise ValueError("Chamber height must be positive.")
        if sheath_length_electrode_m < 0:
            raise ValueError("Electrode sheath length must be non-negative.")
        if sheath_length_grounded_m < 0:
            raise ValueError("Grounded sheath length must be non-negative.")

        bulk_height_m = (
            chamber_height_m
            - sheath_length_electrode_m
            - sheath_length_grounded_m
        )
        if bulk_height_m <= 0:
            raise ValueError(
                "Bulk plasma height must be positive after subtracting both sheath lengths."
            )
        return bulk_height_m

    def compute_plasma_sheath_length_electrode(
        self,
        current_density_a_per_m2: float,
        rf_frequency_hz: float,
        pressure_torr: float,
        electron_temperature_ev: float,
        rf_power: float,
        sheath_voltage: float,
        chamber_radius_m: float,
        chamber_height_m: float,
    ) -> float:
        """Return electrode sheath length from current density and plasma properties."""
        return self.compute_plasma_sheath_length(
            current_density_a_per_m2=current_density_a_per_m2,
            rf_frequency_hz=rf_frequency_hz,
            pressure_torr=pressure_torr,
            electron_temperature_ev=electron_temperature_ev,
            rf_power=rf_power,
            sheath_voltage=sheath_voltage,
            chamber_radius_m=chamber_radius_m,
            chamber_height_m=chamber_height_m,
        )

    def compute_plasma_sheath_length_grounded(
        self,
        current_density_a_per_m2: float,
        rf_frequency_hz: float,
        pressure_torr: float,
        electron_temperature_ev: float,
        rf_power: float,
        sheath_voltage: float,
        chamber_radius_m: float,
        chamber_height_m: float,
    ) -> float:
        """Return grounded sheath length from current density and plasma properties."""
        return self.compute_plasma_sheath_length(
            current_density_a_per_m2=current_density_a_per_m2,
            rf_frequency_hz=rf_frequency_hz,
            pressure_torr=pressure_torr,
            electron_temperature_ev=electron_temperature_ev,
            rf_power=rf_power,
            sheath_voltage=sheath_voltage,
            chamber_radius_m=chamber_radius_m,
            chamber_height_m=chamber_height_m,
        )

    def compute_plasma_resistance(
        self,
        electron_temperature_ev: float,
        RF_power: float,
        RF_frequency: int,
        sheath_voltage: float,
        chamber_radius_m: float,
        chamber_height_m: float,
        pressure_pa: float,
        temperature_k: float,
        sheath_length_electrode_m: float,
        sheath_length_grounded_m: float,
    ) -> float:
        """Return plasma resistance from plasma conductivity and bulk plasma height."""
        plasma_conductivity = self.compute_plasma_conductivity(
            electron_temperature_ev,
            RF_power,
            RF_frequency,
            sheath_voltage,
            chamber_radius_m,
            chamber_height_m,
            pressure_pa,
            temperature_k,
        )
        bulk_height_m = self.compute_bulk_plasma_height(
            chamber_height_m,
            sheath_length_electrode_m,
            sheath_length_grounded_m,
        )
        return bulk_height_m / (
            plasma_conductivity * pi * chamber_radius_m * chamber_radius_m
        )
    
    def compute_plasma_coil_reactance(
        self,
        electron_temperature_ev: float,
        RF_power: float,
        RF_frequency: int,
        sheath_voltage: float,
        chamber_radius_m: float,
        chamber_height_m: float,
        pressure_pa: float,
        temperature_k: float,
        sheath_length_electrode_m: float,
        sheath_length_grounded_m: float,
    ) -> float:
        """Return plasma coil reactance from plasma permittivity and bulk height."""
        plasma_relative_permittivity = self.compute_plasma_relative_permittivity(
            electron_temperature_ev,
            RF_power,
            RF_frequency,
            sheath_voltage,
            chamber_radius_m,
            chamber_height_m,
            pressure_pa,
            temperature_k,
        )
        bulk_height_m = self.compute_bulk_plasma_height(
            chamber_height_m,
            sheath_length_electrode_m,
            sheath_length_grounded_m,
        )
        return -1 * bulk_height_m / (
            (2 * pi * RF_frequency)
            * self.constants.vacuum_permittivity
            * plasma_relative_permittivity
            * pi
            * chamber_radius_m
            * chamber_radius_m
        )
    
    def compute_plasma_cap_reactance(
        self,
        RF_frequency: int,
        chamber_radius_m: float,
        chamber_height_m: float,
        sheath_length_electrode_m: float,
        sheath_length_grounded_m: float,
    ) -> float:
        """Return plasma capacitive reactance from bulk plasma height."""
        bulk_height_m = self.compute_bulk_plasma_height(
            chamber_height_m,
            sheath_length_electrode_m,
            sheath_length_grounded_m,
        )
        return -1 * bulk_height_m / (
            (2 * pi * RF_frequency)
            * self.constants.vacuum_permittivity
            * pi
            * chamber_radius_m
            * chamber_radius_m
        )
    
    def compute_plasma_coil_henry(
        self,
        electron_temperature_ev: float,
        RF_power: float,
        RF_frequency: int,
        sheath_voltage: float,
        chamber_radius_m: float,
        chamber_height_m: float,
        pressure_pa: float,
        temperature_k: float,
        sheath_length_electrode_m: float,
        sheath_length_grounded_m: float,
    ) -> float:
        """Return plasma coil inductance from plasma permittivity and bulk height."""
        plasma_relative_permittivity = self.compute_plasma_relative_permittivity(
            electron_temperature_ev,
            RF_power,
            RF_frequency,
            sheath_voltage,
            chamber_radius_m,
            chamber_height_m,
            pressure_pa,
            temperature_k,
        )
        bulk_height_m = self.compute_bulk_plasma_height(
            chamber_height_m,
            sheath_length_electrode_m,
            sheath_length_grounded_m,
        )
        return -1 * bulk_height_m / (
            (2 * pi * RF_frequency)
            * (2 * pi * RF_frequency)
            * self.constants.vacuum_permittivity
            * plasma_relative_permittivity
            * pi
            * chamber_radius_m
            * chamber_radius_m
        )

    def compute_plasma_cap_farad(
        self,
        chamber_radius_m: float,
        chamber_height_m: float,
        sheath_length_electrode_m: float,
        sheath_length_grounded_m: float,
    ) -> float:
        """Return plasma capacitance from bulk plasma height."""
        bulk_height_m = self.compute_bulk_plasma_height(
            chamber_height_m,
            sheath_length_electrode_m,
            sheath_length_grounded_m,
        )
        return (
            self.constants.vacuum_permittivity * pi * chamber_radius_m * chamber_radius_m
        ) / bulk_height_m
    
    def compute_plasma_sheath_capacitance(
            self,
            sheath_thickness_m: float,
    ) -> float:
        return 1.52 * self.constants.vacuum_permittivity / sheath_thickness_m
    
    def compute_plasma_sheath_capacitance_electrode(
            self,
            sheath_thickness_m: float,
            chamber_radius_m: float,
            
        ) -> float:
        return self.compute_plasma_sheath_capacitance(sheath_thickness_m) * (pi * chamber_radius_m * chamber_radius_m)
    
    def compute_plasma_sheath_capacitance_grounded(
            self,
            sheath_thickness_m: float,
            chamber_radius_m: float,
            chamber_height_m: float,
        ) -> float:
        return (
                self.compute_plasma_sheath_capacitance(sheath_thickness_m) 
                * (pi * chamber_radius_m * chamber_radius_m 
                + 2 * pi * chamber_radius_m * chamber_height_m)
        )
    
    def compute_electron_velocity(
        self,
        electron_temperature_ev: float
    ) -> float:
        return ((8 * electron_temperature_ev) / (pi * self.constants.electron_mass)) ** 0.5

    def compute_plasma_sheath_conductance(
            self,
            sheath_thickness_m: float,
            pressure_torr: float,
            electron_temperature_ev: float,
            RF_power: float,
            sheath_voltage: float,
            chamber_radius_m: float,
            chamber_height_m: float,
        ) -> float:
        return ( 
            (2.17) * ((sheath_thickness_m / (self.compute_ion_mean_free_path_m(pressure_torr)))**(2/3))
            * (
                ((self.compute_plasma_density(electron_temperature_ev, RF_power
                                          , sheath_voltage, chamber_radius_m, chamber_height_m))
                * ((self.constants.electron_charge) **(2))
                )
                / (self.constants.electron_mass * self.compute_electron_velocity(electron_temperature_ev)
                )
            )
            * ( (self.compute_debye_length_m(electron_temperature_ev, RF_power, sheath_voltage, chamber_radius_m, chamber_height_m) 
               / sheath_thickness_m
               ) **(2/3)
            ) 
        )
    
    def compute_plasma_sheath_resistance_electrode(
            
            self,
            sheath_thickness_m: float,
            pressure_torr: float,
            electron_temperature_ev: float,
            RF_power: float,
            sheath_voltage: float,
            chamber_radius_m: float,
            chamber_height_m: float,
        ) -> float:
        return 1 /  (pi * chamber_radius_m * chamber_radius_m * self.compute_plasma_sheath_conductance(
            sheath_thickness_m,
            pressure_torr,
            electron_temperature_ev,
            RF_power,
            sheath_voltage,
            chamber_radius_m,
            chamber_height_m
            )
        )
    
    def compute_plasma_sheath_resistance_grounded(
            
            self,
            sheath_thickness_m: float,
            pressure_torr: float,
            electron_temperature_ev: float,
            RF_power: float,
            sheath_voltage: float,
            chamber_radius_m: float,
            chamber_height_m: float,
        ) -> float:
        return (1 
            / (
                (pi * chamber_radius_m * chamber_radius_m 
                + 2 * pi * chamber_radius_m * chamber_height_m) 
                * self.compute_plasma_sheath_conductance(
                sheath_thickness_m,
                pressure_torr,
                electron_temperature_ev,
                RF_power,
                sheath_voltage,
                chamber_radius_m,
                chamber_height_m
                )
            )
        )
    
    def compute_plasma_sheath_length(
        self,
        current_density_a_per_m2: float,
        rf_frequency_hz: float,
        pressure_torr: float,
        electron_temperature_ev: float,
        rf_power: float,
        sheath_voltage: float,
        chamber_radius_m: float,
        chamber_height_m: float,
    ) -> float:
        """Return sheath length from current density and plasma properties."""
        if current_density_a_per_m2 <= 0:
            raise ValueError("Current density must be positive.")
        if rf_frequency_hz <= 0:
            raise ValueError("RF frequency must be positive.")
        if pressure_torr <= 0:
            raise ValueError("Pressure must be positive.")

        ion_mean_free_path_m = self.compute_ion_mean_free_path_m(pressure_torr)
        debye_length_m = self.compute_debye_length_m(
            electron_temperature_ev,
            rf_power,
            sheath_voltage,
            chamber_radius_m,
            chamber_height_m,
        )
        plasma_density = self.compute_plasma_density(
            electron_temperature_ev,
            rf_power,
            sheath_voltage,
            chamber_radius_m,
            chamber_height_m,
        )

        omega_rf = 2 * pi * rf_frequency_hz
        density_ratio = current_density_a_per_m2 / (
            self.constants.electron_charge * omega_rf * plasma_density
        )

        return (
            1.95
            * (
                ((2 * ion_mean_free_path_m)
                / (pi * pi * debye_length_m * debye_length_m))
                * density_ratio
            ) ** 0.5
            * density_ratio
        )
