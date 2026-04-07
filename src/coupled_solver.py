"""Coupled plasma and circuit iteration helpers."""

from dataclasses import dataclass, replace

from .plasma import ChamberConditions, PlasmaCalculator, PlasmaComputationResult, PlasmaConditions
from .spice import PlasmaCircuitParameters, PlasmaCircuitResult, SpiceSimulator


@dataclass(frozen=True)
class SelfConsistentPlasmaCircuitResult:
    """Final outputs from the coupled plasma-circuit iteration."""

    plasma_result: PlasmaComputationResult
    circuit_result: PlasmaCircuitResult
    current_density_a_per_m2: float
    sheath_length_electrode_m: float
    sheath_length_grounded_m: float
    iterations: int
    converged: bool
    sheath_length_relative_change: float


def solve_self_consistent_plasma_circuit(
    plasma: PlasmaCalculator,
    simulator: SpiceSimulator,
    chamber: ChamberConditions,
    plasma_conditions: PlasmaConditions,
    max_iterations: int = 50,
    relative_tolerance: float = 1e-6,
    damping: float = 0.5,
) -> SelfConsistentPlasmaCircuitResult:
    """Iterate until plasma sheath lengths are consistent with circuit current density."""
    if chamber.electrode_area_m2 is None or chamber.electrode_area_m2 <= 0:
        raise ValueError("Electrode area must be set to compute current density.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if not (0 < damping <= 1):
        raise ValueError("damping must be in the interval (0, 1].")

    working_conditions = replace(plasma_conditions)
    last_relative_change = float("inf")
    converged = False

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

        current_density_a_per_m2 = (
            abs(circuit_result.src_node_current_rms) / chamber.electrode_area_m2
        )
        
        # Calculate electrode and grounded sheath lengths separately
        # using current density appropriate to each surface area
        current_density_electrode = (
            abs(circuit_result.src_node_current_rms) / chamber.electrode_area_m2
        )
        current_density_grounded = (
            abs(circuit_result.src_node_current_rms) / chamber.grounded_area_m2
        )
        
        raw_sheath_length_electrode_m = plasma.compute_plasma_sheath_length_electrode(
            current_density_a_per_m2=current_density_electrode,
            rf_frequency_hz=working_conditions.RF_frequency,
            pressure_torr=chamber.pressure_torr,
            electron_temperature_ev=plasma_result.electron_temperature_ev,
            rf_power=working_conditions.RF_power,
            sheath_voltage=working_conditions.sheath_voltage,
            chamber_radius_m=chamber.chamber_radius_m,
            chamber_height_m=chamber.chamber_height_m,
        )
        raw_sheath_length_grounded_m = plasma.compute_plasma_sheath_length_grounded(
            current_density_a_per_m2=current_density_grounded,
            rf_frequency_hz=working_conditions.RF_frequency,
            pressure_torr=chamber.pressure_torr,
            electron_temperature_ev=plasma_result.electron_temperature_ev,
            rf_power=working_conditions.RF_power,
            sheath_voltage=working_conditions.sheath_voltage,
            chamber_radius_m=chamber.chamber_radius_m,
            chamber_height_m=chamber.chamber_height_m,
        )
        
        # Apply damping to both sheath lengths
        updated_sheath_length_electrode_m = (
            (1 - damping) * working_conditions.sheath_length_electrode_m
            + damping * raw_sheath_length_electrode_m
        )
        updated_sheath_length_grounded_m = (
            (1 - damping) * working_conditions.sheath_length_grounded_m
            + damping * raw_sheath_length_grounded_m
        )
        
        # Track relative change in electrode sheath length
        last_relative_change = abs(
            updated_sheath_length_electrode_m - working_conditions.sheath_length_electrode_m
        ) / max(abs(working_conditions.sheath_length_electrode_m), 1e-30)
        
        working_conditions = replace(
            working_conditions,
            sheath_length_electrode_m=updated_sheath_length_electrode_m,
            sheath_length_grounded_m=updated_sheath_length_grounded_m,
            Current_density=current_density_a_per_m2,
            electron_temperature_ev=plasma_result.electron_temperature_ev,
        )

        if last_relative_change < relative_tolerance:
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
    final_current_density_electrode = (
        abs(final_circuit_result.src_node_current_rms) / chamber.electrode_area_m2
    )
    final_current_density_grounded = (
        abs(final_circuit_result.src_node_current_rms) / chamber.grounded_area_m2
    )
    working_conditions.Current_density = final_current_density_electrode

    return SelfConsistentPlasmaCircuitResult(
        plasma_result=final_plasma_result,
        circuit_result=final_circuit_result,
        current_density_a_per_m2=final_current_density_electrode,
        sheath_length_electrode_m=working_conditions.sheath_length_electrode_m,
        sheath_length_grounded_m=working_conditions.sheath_length_grounded_m,
        iterations=iteration,
        converged=converged,
        sheath_length_relative_change=last_relative_change,
    )
