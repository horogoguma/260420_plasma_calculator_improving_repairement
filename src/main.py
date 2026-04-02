"""Simple workflow test script."""

from PySpice.Unit import *

from src.plasma import ChamberConditions, PlasmaCalculator, PlasmaConditions
from src.plasma.constants import MM_TO_M
from src.spice import SpiceSimulator



def main() -> None:
    sim = SpiceSimulator()
    sim.build_rc_lowpass(1 @ u_kOhm, 1 @ u_uF)
    analysis = sim.run_ac(1 @ u_Hz, 1 @ u_MHz, points=20)

    voltage = complex(analysis.n2[0])
    current = voltage / float(sim.R.resistance)
    power = voltage * current

    # chamber conditions 내부에 선언하면 다른 변수에 접근할 수 없어서 chamber condtions 밖에 선언
    # ChamberConditions 밖에서 먼저 변수 선언
    chamber_radius_m = 170 * MM_TO_M
    chamber_height_m = 9 * MM_TO_M

    chamber = ChamberConditions(
        pressure_torr=3.5,
        temperature_k=423.0,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
        electrode_area_m2=3.14 * (chamber_radius_m) ** 2,
        grounded_area_m2=3.14 * (chamber_radius_m) ** 2
        + 2 * 3.14 * chamber_radius_m * chamber_height_m,
    )

    plasma_conditions = PlasmaConditions(
        electron_temperature_ev=1.5,
        sheath_voltage=441.0,
        sheath_length_m=1.035 * MM_TO_M,
        RF_power=900.0,
        RF_frequency=12.9e6,
    )
    plasma = PlasmaCalculator(
        chamber=chamber,
        plasma_conditions=plasma_conditions,
    )

    impedance = plasma.compute_impedance(voltage, current)
    print(f"Computed impedance: {impedance} ohm")
    print(f"Computed power: {power} W")

    pressure_torr = chamber.pressure_torr
    pressure_pa = chamber.pressure_pa
    temperature_k = chamber.temperature_k
    chamber_radius_m = chamber.chamber_radius_m
    chamber_height_m = chamber.chamber_height_m
    sheath_voltage = plasma_conditions.sheath_voltage
    rf_power = plasma_conditions.RF_power
    rf_frequency = plasma_conditions.RF_frequency

    electron_temperature_ev, iterations, target_value = plasma.solve_electron_temperature(
        start_ev=plasma_conditions.electron_temperature_ev,
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
    )
    plasma.electron_temperature_ev = electron_temperature_ev

    print(f"Target value close to 1: {target_value} (iterations: {iterations})")
    print(f"Computed electron temperature: {electron_temperature_ev} eV")

    number_need_to_be_one = plasma.compute_number_need_to_be_one(
        electron_temperature_ev=electron_temperature_ev,
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
    )

    k_el = plasma.compute_elastic_collision_constant(
        electron_temperature_ev=electron_temperature_ev,
    )
    k_ex = plasma.compute_exitation_constant(
        electron_temperature_ev=electron_temperature_ev,
    )
    k_iz = plasma.compute_ionization_constant(
        electron_temperature_ev=electron_temperature_ev,
    )
    u_b = plasma.compute_bohm_velocity(
        electron_temperature_ev=electron_temperature_ev,
    )
    n_g = plasma.compute_gas_number_density(
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
    )
    d_effective = plasma.compute_effective_length(
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
    )
    loss_collision = plasma.compute_collision_energy_loss(
        electron_temperature_ev=electron_temperature_ev,
    )
    loss_electron_ion = plasma.compute_electron_ion_energy_loss(
        electron_temperature_ev=electron_temperature_ev,
        sheath_voltage=sheath_voltage,
    )
    loss_total = plasma.compute_total_energy_loss(
        electron_temperature_ev=electron_temperature_ev,
        sheath_voltage=sheath_voltage,
    )
    n_e = plasma.compute_plasma_density(
        electron_temperature_ev=electron_temperature_ev,
        RF_power=rf_power,
        sheath_voltage=sheath_voltage,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
    )

    d_ion = plasma.compute_ion_mean_free_path_m(pressure_torr=pressure_torr)

    f_collisional = plasma.compute_collisional_frequency(
        electron_temperature_ev=electron_temperature_ev,
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
    )
    w_pe = plasma.compute_plasma_angular_frequency(
        electron_temperature_ev=electron_temperature_ev,
        RF_power=rf_power,
        sheath_voltage=sheath_voltage,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
    )
    conductivity_pe = plasma.compute_plasma_conductivity(
        electron_temperature_ev=electron_temperature_ev,
        RF_power=rf_power,
        RF_frequency=rf_frequency,
        sheath_voltage=sheath_voltage,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
    )
    relative_permittivity_pe = plasma.compute_plasma_relative_permittivity(
        electron_temperature_ev=electron_temperature_ev,
        RF_power=rf_power,
        RF_frequency=rf_frequency,
        sheath_voltage=sheath_voltage,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
    )
    debye_length = plasma.compute_debye_length_m(
        electron_temperature_ev=electron_temperature_ev,
        RF_power=rf_power,
        sheath_voltage=sheath_voltage,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
    )

    plasma_resistance = plasma.compute_plasma_resistance(
        electron_temperature_ev=electron_temperature_ev,
        RF_power=rf_power,
        RF_frequency=rf_frequency,
        sheath_voltage=sheath_voltage,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
    )

    plasma_coil_reactance = plasma.compute_plasma_coil_reactance(
        electron_temperature_ev=electron_temperature_ev,
        RF_power=rf_power,
        RF_frequency=rf_frequency,
        sheath_voltage=sheath_voltage,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
    )

    plasma_cap_reactance = plasma.compute_plasma_cap_reactance(
        RF_frequency=rf_frequency,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
    )

    plasma_coil_henry = plasma.compute_plasma_coil_henry(
        electron_temperature_ev=electron_temperature_ev,
        RF_power=rf_power,
        RF_frequency=rf_frequency,
        sheath_voltage=sheath_voltage,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
    )

    plasma_cap_farad = plasma.compute_plasma_cap_farad(
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
    )

    plasma_sheath_capacitance = plasma.compute_plasma_sheath_capacitance(
        sheath_thickness_m = plasma_conditions.sheath_length_m,
    )

    plasma_sheath_capacitance_electrode = plasma.compute_plasma_sheath_capacitance_electrode(
        sheath_thickness_m = plasma_conditions.sheath_length_m,
        chamber_radius_m = chamber.chamber_radius_m,
    )

    plasma_sheath_capacitance_grounded = plasma.compute_plasma_sheath_capacitance_grounded(
        sheath_thickness_m = plasma_conditions.sheath_length_m,
        chamber_radius_m = chamber.chamber_radius_m,
        chamber_height_m = chamber.chamber_height_m,
    )

    u_e = plasma.compute_electron_velocity(electron_temperature_ev=electron_temperature_ev)

    plasma_sheath_conductance = plasma.compute_plasma_sheath_conductance(
        sheath_thickness_m = plasma_conditions.sheath_length_m,
        pressure_torr = pressure_torr,
        electron_temperature_ev=electron_temperature_ev,
        RF_power=rf_power,
        sheath_voltage=sheath_voltage,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
    )

    plasma_sheath_resistance_electrode = plasma.compute_plasma_sheath_resistance_electrode(
        sheath_thickness_m = plasma_conditions.sheath_length_m,
        pressure_torr = pressure_torr,
        electron_temperature_ev=electron_temperature_ev,
        RF_power=rf_power,
        sheath_voltage=sheath_voltage,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
    )

    plasma_sheath_resistance_grounded = plasma.compute_plasma_sheath_resistance_grounded(   
        sheath_thickness_m = plasma_conditions.sheath_length_m,
        pressure_torr = pressure_torr,
        electron_temperature_ev=electron_temperature_ev,
        RF_power=rf_power,
        sheath_voltage=sheath_voltage,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
    )



    
    print(f"Number that should be 1: {number_need_to_be_one}")
    print(f"Elastic collision constant: {k_el}")
    print(f"Excitation constant: {k_ex}")
    print(f"Debye length: {debye_length} m")
    print(f"Ionization constant: {k_iz}")
    print(f"Bohm velocity: {u_b} m/s")
    print(f"Gas number density: {n_g} m^-3")
    print(f"Effective length: {d_effective} m")
    print(f"Collision energy loss: {loss_collision} eV")
    print(f"Electron-ion energy loss: {loss_electron_ion} eV")
    print(f"Total energy loss: {loss_total} eV")
    print(f"Plasma density: {n_e} m^-3")
    print(f"Ion mean free path: {d_ion} m")
    print(f"Collisional frequency: {f_collisional} /s")
    print(f"Plasma angular frequency: {w_pe} rad/s")
    print(f"Plasma conductivity: {conductivity_pe} S/m")
    print(f"Plasma relative permittivity: {relative_permittivity_pe}")
    print(f"Plasma resistance: {plasma_resistance} ohm")
    print(f"Plasma coil reactance: {plasma_coil_reactance} ohm")
    print(f"Plasma capacitive reactance: {plasma_cap_reactance} ohm")
    print(f"Plasma coil inductance: {plasma_coil_henry} H")
    print(f"Plasma cap farad: {plasma_cap_farad} F")
    print(f"Plasma sheath capacitance: {plasma_sheath_capacitance} F/m^2")
    print(f"Plasma sheath capacitance (electrode): {plasma_sheath_capacitance_electrode} F")
    print(f"Plasma sheath capacitance (grounded): {plasma_sheath_capacitance_grounded} F")
    print(f"Electron velocity: {u_e} m/s")
    print(f"Plasma sheath conductance: {plasma_sheath_conductance} S/m^2")
    print(f"Plasma sheath resistance (electrode): {plasma_sheath_resistance_electrode} ohm")
    print(f"Plasma sheath resistance (grounded): {plasma_sheath_resistance_grounded} ohm")
    


if __name__ == "__main__":
    main()
