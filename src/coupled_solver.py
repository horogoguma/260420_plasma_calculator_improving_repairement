"""Coupled plasma and circuit iteration helpers."""

from dataclasses import dataclass, replace
from math import sqrt

from .plasma import ChamberConditions, PlasmaCalculator, PlasmaComputationResult, PlasmaConditions
from .spice import PlasmaCircuitParameters, PlasmaCircuitResult, SpiceSimulator


@dataclass(frozen=True)
class SelfConsistentPlasmaCircuitResult:
    """Final outputs from the coupled plasma-circuit iteration."""

    plasma_result: PlasmaComputationResult
    circuit_result: PlasmaCircuitResult
    current_density_a_per_m2: float
    current_density_rms_a_per_m2: float
    sheath_length_electrode_m: float
    sheath_length_grounded_m: float
    iterations: int
    converged: bool
    sheath_length_relative_change: float
    sheath_voltage_relative_change: float
    bulk_power_relative_change: float
    absorbed_bulk_power_w: float


def solve_self_consistent_plasma_circuit(
    plasma: PlasmaCalculator,
    simulator: SpiceSimulator,
    chamber: ChamberConditions,
    plasma_conditions: PlasmaConditions,
    max_iterations: int = 80,
    relative_tolerance: float = 1e-6,
    damping: float = 0.5,
) -> SelfConsistentPlasmaCircuitResult:
    """Iterate until plasma sheath lengths and sheath voltage are self-consistent."""
    if plasma.compute_electrode_area_m2(chamber) <= 0:
        raise ValueError("Electrode area must be positive to compute current density.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if not (0 < damping <= 1):
        raise ValueError("damping must be in the interval (0, 1].")

    working_conditions = replace(plasma_conditions)
    last_relative_change = float("inf")
    last_sheath_voltage_relative_change = float("inf")
    last_bulk_power_relative_change = float("inf")
    converged = False

    if working_conditions.absorbed_bulk_power_w is None:
        working_conditions = replace(
            working_conditions,
            absorbed_bulk_power_w=working_conditions.RF_power,
        )

    for iteration in range(1, max_iterations + 1):
        plasma_result = plasma.compute_plasma_properties(
            chamber=chamber,
            plasma_conditions=working_conditions,
        )
        plasma_circuit = PlasmaCircuitParameters(
            plasma_resistance=plasma_result.plasma_resistance,
            plasma_coil_henry=plasma_result.plasma_coil_henry,
            plasma_cap_farad=plasma_result.plasma_cap_farad,
            plasma_sheath_capacitance_electrode=plasma_result.plasma_sheath_capacitance_electrode,
            plasma_sheath_capacitance_grounded=plasma_result.plasma_sheath_capacitance_grounded,
            plasma_sheath_resistance_electrode=plasma_result.plasma_sheath_resistance_electrode,
            plasma_sheath_resistance_grounded=plasma_result.plasma_sheath_resistance_grounded,
            rf_frequency_hz=working_conditions.RF_frequency,
        )
        simulator.build_plasma_equivalent_circuit(
            plasma_circuit,
            target_power_w=working_conditions.RF_power,
        )
        circuit_result = simulator.compute_plasma_circuit_response()
        updated_absorbed_bulk_power_w = (
            (1 - damping) * working_conditions.absorbed_bulk_power_w
            + damping * circuit_result.plasma_resistance_power_w
        )

        current_density_electrode_rms, current_density_grounded_rms = (
            plasma.compute_sheath_current_densities(
                current_a=circuit_result.src_node_current_rms,
                chamber=chamber,
            )
        )
        current_density_electrode = current_density_electrode_rms * sqrt(2.0)
        current_density_grounded = current_density_grounded_rms * sqrt(2.0)
        current_density_a_per_m2 = current_density_electrode
        total_sheath_voltage = plasma.compute_voltage_sheath_total_sum(
            current_density_a_per_m2=current_density_electrode,
            rf_frequency_hz=working_conditions.RF_frequency,
            pressure_torr=chamber.pressure_torr,
            electron_temperature_ev=plasma_result.electron_temperature_ev,
            rf_power=working_conditions.RF_power,
            sheath_length_m=working_conditions.sheath_length_electrode_m,
            electrode_radius_m=plasma.compute_effective_electrode_radius_m(chamber),
            chamber_radius_m=chamber.chamber_radius_m,
            chamber_height_m=chamber.chamber_height_m,
            rf_voltage=circuit_result.source_voltage_peak,
        )
        updated_sheath_voltage = (
            (1 - damping) * working_conditions.sheath_voltage
            + damping * total_sheath_voltage
        )
        raw_sheath_length_electrode_m = plasma.compute_plasma_sheath_length_electrode(
            current_density_a_per_m2=current_density_electrode,
            rf_frequency_hz=working_conditions.RF_frequency,
            pressure_torr=chamber.pressure_torr,
            electron_temperature_ev=plasma_result.electron_temperature_ev,
            rf_power=working_conditions.RF_power,
            sheath_voltage=updated_sheath_voltage,
            chamber_radius_m=chamber.chamber_radius_m,
            chamber_height_m=chamber.chamber_height_m,
        )
        raw_sheath_length_grounded_m = plasma.compute_plasma_sheath_length_grounded(
            current_density_a_per_m2=current_density_grounded,
            rf_frequency_hz=working_conditions.RF_frequency,
            pressure_torr=chamber.pressure_torr,
            electron_temperature_ev=plasma_result.electron_temperature_ev,
            rf_power=working_conditions.RF_power,
            sheath_voltage=updated_sheath_voltage,
            chamber_radius_m=chamber.chamber_radius_m,
            chamber_height_m=chamber.chamber_height_m,
        )
        updated_sheath_length_electrode_m = (
            (1 - damping) * working_conditions.sheath_length_electrode_m
            + damping * raw_sheath_length_electrode_m
        )
        updated_sheath_length_grounded_m = (
            (1 - damping) * working_conditions.sheath_length_grounded_m
            + damping * raw_sheath_length_grounded_m
        )
        last_relative_change = abs(
            updated_sheath_length_electrode_m - working_conditions.sheath_length_electrode_m
        ) / max(abs(working_conditions.sheath_length_electrode_m), 1e-30)
        last_sheath_voltage_relative_change = abs(
            updated_sheath_voltage - working_conditions.sheath_voltage
        ) / max(abs(working_conditions.sheath_voltage), 1e-30)
        last_bulk_power_relative_change = abs(
            updated_absorbed_bulk_power_w - working_conditions.absorbed_bulk_power_w
        ) / max(abs(working_conditions.absorbed_bulk_power_w), 1e-30)
        working_conditions = replace(
            working_conditions,
            sheath_length_electrode_m=updated_sheath_length_electrode_m,
            sheath_length_grounded_m=updated_sheath_length_grounded_m,
            sheath_voltage=updated_sheath_voltage,
            Current_density=current_density_a_per_m2,
            electron_temperature_ev=plasma_result.electron_temperature_ev,
            rf_voltage=circuit_result.source_voltage_peak,
            absorbed_bulk_power_w=updated_absorbed_bulk_power_w,
        )

        if (
            last_relative_change < relative_tolerance
            and last_sheath_voltage_relative_change < relative_tolerance
            and last_bulk_power_relative_change < relative_tolerance
        ):
            converged = True
            break

    final_plasma_result = plasma.compute_plasma_properties(
        chamber=chamber,
        plasma_conditions=working_conditions,
    )
    final_plasma_circuit = PlasmaCircuitParameters(
        plasma_resistance=final_plasma_result.plasma_resistance,
        plasma_coil_henry=final_plasma_result.plasma_coil_henry,
        plasma_cap_farad=final_plasma_result.plasma_cap_farad,
        plasma_sheath_capacitance_electrode=final_plasma_result.plasma_sheath_capacitance_electrode,
        plasma_sheath_capacitance_grounded=final_plasma_result.plasma_sheath_capacitance_grounded,
        plasma_sheath_resistance_electrode=final_plasma_result.plasma_sheath_resistance_electrode,
        plasma_sheath_resistance_grounded=final_plasma_result.plasma_sheath_resistance_grounded,
        rf_frequency_hz=working_conditions.RF_frequency,
    )
    simulator.build_plasma_equivalent_circuit(
        final_plasma_circuit,
        target_power_w=working_conditions.RF_power,
    )
    final_circuit_result = simulator.compute_plasma_circuit_response()
    final_current_density_electrode_rms, final_current_density_grounded_rms = (
        plasma.compute_sheath_current_densities(
            current_a=final_circuit_result.src_node_current_rms,
            chamber=chamber,
        )
    )
    final_current_density_electrode = final_current_density_electrode_rms * sqrt(2.0)
    final_current_density_grounded = final_current_density_grounded_rms * sqrt(2.0)
    final_total_sheath_voltage = plasma.compute_voltage_sheath_total_sum(
        current_density_a_per_m2=final_current_density_electrode,
        rf_frequency_hz=working_conditions.RF_frequency,
        pressure_torr=chamber.pressure_torr,
        electron_temperature_ev=final_plasma_result.electron_temperature_ev,
        rf_power=working_conditions.RF_power,
        sheath_length_m=working_conditions.sheath_length_electrode_m,
        electrode_radius_m=plasma.compute_effective_electrode_radius_m(chamber),
        chamber_radius_m=chamber.chamber_radius_m,
        chamber_height_m=chamber.chamber_height_m,
        rf_voltage=final_circuit_result.source_voltage_peak,
    )
    final_updated_sheath_voltage = (
        (1 - damping) * working_conditions.sheath_voltage
        + damping * final_total_sheath_voltage
    )
    working_conditions = replace(
        working_conditions,
        Current_density=final_current_density_electrode,
        sheath_voltage=final_updated_sheath_voltage,
        rf_voltage=final_circuit_result.source_voltage_peak,
        absorbed_bulk_power_w=final_circuit_result.plasma_resistance_power_w,
    )
    final_plasma_result = plasma.compute_plasma_properties(
        chamber=chamber,
        plasma_conditions=working_conditions,
    )
    final_plasma_circuit = PlasmaCircuitParameters(
        plasma_resistance=final_plasma_result.plasma_resistance,
        plasma_coil_henry=final_plasma_result.plasma_coil_henry,
        plasma_cap_farad=final_plasma_result.plasma_cap_farad,
        plasma_sheath_capacitance_electrode=final_plasma_result.plasma_sheath_capacitance_electrode,
        plasma_sheath_capacitance_grounded=final_plasma_result.plasma_sheath_capacitance_grounded,
        plasma_sheath_resistance_electrode=final_plasma_result.plasma_sheath_resistance_electrode,
        plasma_sheath_resistance_grounded=final_plasma_result.plasma_sheath_resistance_grounded,
        rf_frequency_hz=working_conditions.RF_frequency,
    )
    simulator.build_plasma_equivalent_circuit(
        final_plasma_circuit,
        target_power_w=working_conditions.RF_power,
    )
    final_circuit_result = simulator.compute_plasma_circuit_response()
    final_current_density_electrode_rms, final_current_density_grounded_rms = (
        plasma.compute_sheath_current_densities(
            current_a=final_circuit_result.src_node_current_rms,
            chamber=chamber,
        )
    )
    final_current_density_electrode = final_current_density_electrode_rms * sqrt(2.0)
    final_current_density_grounded = final_current_density_grounded_rms * sqrt(2.0)

    return SelfConsistentPlasmaCircuitResult(
        plasma_result=final_plasma_result,
        circuit_result=final_circuit_result,
        current_density_a_per_m2=final_current_density_electrode,
        current_density_rms_a_per_m2=final_current_density_electrode_rms,
        sheath_length_electrode_m=working_conditions.sheath_length_electrode_m,
        sheath_length_grounded_m=working_conditions.sheath_length_grounded_m,
        iterations=iteration,
        converged=converged,
        sheath_length_relative_change=last_relative_change,
        sheath_voltage_relative_change=last_sheath_voltage_relative_change,
        bulk_power_relative_change=last_bulk_power_relative_change,
        absorbed_bulk_power_w=working_conditions.absorbed_bulk_power_w,
    )
