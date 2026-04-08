"""Simple workflow test script."""

from src.coupled_solver import solve_self_consistent_plasma_circuit
from src.plasma import ChamberConditions, PlasmaCalculator, PlasmaConditions
from src.plasma.constants import MM_TO_M
from src.spice import SpiceSimulator



def main() -> None:
    sim = SpiceSimulator()

    # chamber conditions 내부에 선언하면 다른 변수에 접근할 수 없어서 chamber condtions 밖에 선언
    # ChamberConditions 밖에서 먼저 변수 선언
    chamber_radius_m = 170 * MM_TO_M
    electrode_radius_m = 150 * MM_TO_M
    chamber_height_m = 9 * MM_TO_M

    chamber = ChamberConditions(
        pressure_torr=3.5,
        temperature_k=423.0,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
        electrode_radius_m=electrode_radius_m,
    )

    plasma_conditions = PlasmaConditions(
        electron_temperature_ev=1.5,
        sheath_voltage=441.0,
        sheath_length_electrode_m=1.035 * MM_TO_M,
        sheath_length_grounded_m=1.035 * MM_TO_M,
        RF_power=900.0,
        RF_frequency=12.9e6,
    )
    plasma = PlasmaCalculator(
        chamber=chamber,
        plasma_conditions=plasma_conditions,
    )

    coupled_result = solve_self_consistent_plasma_circuit(
        plasma=plasma,
        simulator=sim,
        chamber=chamber,
        plasma_conditions=plasma_conditions,
    )
    plasma_result = coupled_result.plasma_result
    plasma_spice_result = coupled_result.circuit_result
    plasma_voltage_bias = plasma.compute_plasma_voltage_bias(
        current_density_a_per_m2=coupled_result.current_density_a_per_m2,
        rf_frequency_hz=plasma_conditions.RF_frequency,
        pressure_torr=chamber.pressure_torr,
        electron_temperature_ev=plasma_result.electron_temperature_ev,
        rf_power=plasma_conditions.RF_power,
        sheath_length_m=coupled_result.sheath_length_electrode_m,
        electrode_radius_m=plasma.compute_effective_electrode_radius_m(chamber),
        chamber_radius_m=chamber.chamber_radius_m,
        chamber_height_m=chamber.chamber_height_m,
        rf_voltage=plasma_spice_result.source_voltage_peak,
    )

    print(
        "Target value close to 1: "
        f"{plasma_result.electron_temperature_target_value} "
        f"(iterations: {plasma_result.electron_temperature_iterations})"
    )
    print(
        "Coupled iteration converged: "
        f"{coupled_result.converged} "
        f"(iterations: {coupled_result.iterations}, "
        f"relative sheath change: {coupled_result.sheath_length_relative_change})"
    )
    print(f"Self-consistent electrode sheath length: {coupled_result.sheath_length_electrode_m} m")
    print(f"Self-consistent grounded sheath length: {coupled_result.sheath_length_grounded_m} m")
    print(f"Electrode radius: {plasma.compute_effective_electrode_radius_m(chamber)} m")
    print(f"Electrode area: {plasma.compute_electrode_area_m2(chamber)} m^2")
    print(f"Grounded area: {plasma.compute_grounded_area_m2(chamber)} m^2")
    
    bulk_plasma_height_m = (
        chamber_height_m 
        - coupled_result.sheath_length_electrode_m 
        - coupled_result.sheath_length_grounded_m
    )
    print(f"Bulk plasma height: {bulk_plasma_height_m} m")

    print(
        "Self-consistent current density: "
        f"{coupled_result.current_density_a_per_m2} A/m^2"
    )
    print(f"Computed electron temperature: {plasma_result.electron_temperature_ev} eV")
    print(f"Number that should be 1: {plasma_result.number_need_to_be_one}")
    print(f"Elastic collision constant: {plasma_result.elastic_collision_constant}")
    print(f"Excitation constant: {plasma_result.excitation_constant}")
    print(f"Debye length: {plasma_result.debye_length_m} m")
    print(f"Ionization constant: {plasma_result.ionization_constant}")
    print(f"Bohm velocity: {plasma_result.bohm_velocity} m/s")
    print(f"Gas number density: {plasma_result.gas_number_density} m^-3")
    print(f"Effective length: {plasma_result.effective_length} m")
    print(f"Collision energy loss: {plasma_result.collision_energy_loss} eV")
    print(f"Electron-ion energy loss: {plasma_result.electron_ion_energy_loss} eV")
    print(f"Total energy loss: {plasma_result.total_energy_loss} eV")
    print(f"Plasma density: {plasma_result.plasma_density} m^-3")
    print(f"Ion mean free path: {plasma_result.ion_mean_free_path_m} m")
    print(f"Collisional frequency: {plasma_result.collisional_frequency} /s")
    print(f"Plasma angular frequency: {plasma_result.plasma_angular_frequency} rad/s")
    print(f"Plasma conductivity: {plasma_result.plasma_conductivity} S/m")
    print(f"Plasma relative permittivity: {plasma_result.plasma_relative_permittivity}")
    print(f"Plasma resistance: {plasma_result.plasma_resistance} ohm")
    print(f"Plasma coil reactance: {plasma_result.plasma_coil_reactance} ohm")
    print(f"Plasma capacitive reactance: {plasma_result.plasma_cap_reactance} ohm")
    print(f"Plasma coil inductance: {plasma_result.plasma_coil_henry} H")
    print(f"Plasma cap farad: {plasma_result.plasma_cap_farad} F")
    print(f"Plasma sheath capacitance: {plasma_result.plasma_sheath_capacitance} F/m^2")
    print(
        "Plasma sheath capacitance (electrode): "
        f"{plasma_result.plasma_sheath_capacitance_electrode} F"
    )
    print(
        "Plasma sheath capacitance (grounded): "
        f"{plasma_result.plasma_sheath_capacitance_grounded} F"
    )
    print(f"Electron velocity: {plasma_result.electron_velocity} m/s")
    print(f"Plasma sheath conductance: {plasma_result.plasma_sheath_conductance} S/m^2")
    print(
        "Plasma sheath resistance (electrode): "
        f"{plasma_result.plasma_sheath_resistance_electrode} ohm"
    )
    print(
        "Plasma sheath resistance (grounded): "
        f"{plasma_result.plasma_sheath_resistance_grounded} ohm"
    )
    print(f"Plasma target power: {plasma_spice_result.target_power_w} W")
    print(f"Plasma source voltage peak: {plasma_spice_result.source_voltage_peak} V")
    print(f"Plasma source voltage rms: {plasma_spice_result.source_voltage_rms} V")
    print(f"Plasma voltage bias: {plasma_voltage_bias} V")
    print(f"Plasma electrode sheath impedance: {plasma_spice_result.electrode_sheath_impedance} ohm")
    print(f"Plasma bulk impedance: {plasma_spice_result.bulk_plasma_impedance} ohm")
    print(f"Plasma grounded sheath impedance: {plasma_spice_result.grounded_sheath_impedance} ohm")
    print(f"Plasma total impedance: {plasma_spice_result.total_impedance} ohm")
    print(f"Plasma source current rms: {plasma_spice_result.source_current_rms} A")
    print(f"Plasma src node current rms: {plasma_spice_result.src_node_current_rms} A")
    print(
        "Plasma src node resistor current rms: "
        f"{plasma_spice_result.src_node_resistor_current_rms} A"
    )
    print(
        "Plasma src node capacitor current rms: "
        f"{plasma_spice_result.src_node_capacitor_current_rms} A"
    )
    print(
        "Power dissipated in plasma_sheath_resistance_electrode: "
        f"{plasma_spice_result.electrode_sheath_resistor_power_w} W"
    )
    print(
        "Power dissipated in plasma_resistance: "
        f"{plasma_spice_result.plasma_resistance_power_w} W"
    )
    print(
        "Power dissipated in plasma_sheath_resistance_grounded: "
        f"{plasma_spice_result.grounded_sheath_resistor_power_w} W"
    )
    print(
        "Total resistor power: "
        f"{plasma_spice_result.total_resistor_power_w} W"
    )
    print(f"Plasma average power: {plasma_spice_result.average_power_w} W")
    


if __name__ == "__main__":
    main()
