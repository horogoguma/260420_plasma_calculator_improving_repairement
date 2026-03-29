"""Simple workflow test script."""

from PySpice.Unit import *

from src.plasma import ChamberConditions, PlasmaCalculator, PlasmaConditions
from src.spice import SpiceSimulator


def main() -> None:
    sim = SpiceSimulator()
    sim.build_rc_lowpass(1 @ u_kOhm, 1 @ u_uF)
    analysis = sim.run_ac(1 @ u_Hz, 1 @ u_MHz, points=20)

    voltage = float(analysis.n2[0])
    current = voltage / float(sim.R.resistance)
    power = voltage * current

    chamber = ChamberConditions(
        pressure_torr=7.5,
        temperature_k=423.0,
        chamber_radius_m=0.170,
        chamber_height_m=0.009,
    )
    plasma_conditions = PlasmaConditions(
        electron_temperature_ev=1.5,
        sheath_voltage=441.0,
        RF_power=1200.0,
    )
    plasma = PlasmaCalculator(
        chamber=chamber,
        plasma_conditions=plasma_conditions,
    )

    impedance = plasma.compute_impedance(voltage, current)
    print(f"Computed impedance: {impedance} ohm")
    print(f"Computed power: {power} W")

    electron_temperature_ev, iterations, target_value = plasma.solve_electron_temperature()
    plasma.electron_temperature_ev = electron_temperature_ev
    sheath_voltage = plasma_conditions.sheath_voltage
    rf_power = plasma_conditions.RF_power
    rf_frequency = plasma_conditions.RF_frequency
    pressure_pa = chamber.pressure_pa
    temperature_k = chamber.temperature_k
    chamber_radius_m = chamber.chamber_radius_m
    chamber_height_m = chamber.chamber_height_m

    print(f"Target value close to 1: {target_value} (iterations: {iterations})")
    print(f"Computed electron temperature: {electron_temperature_ev} eV")

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
    f_collisional = plasma.compute_collisional_frequency(
        electron_temperature_ev=electron_temperature_ev,
        pressure_pa=pressure_pa,
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
    )
    relative_permittivity_pe = plasma.compute_plasma_relative_permittivity(
        electron_temperature_ev=electron_temperature_ev,
        RF_power=rf_power,
        RF_frequency=rf_frequency,
        sheath_voltage=sheath_voltage,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
        pressure_pa=pressure_pa,
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
        sheath_voltage=sheath_voltage,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
        pressure_pa=pressure_pa,
    )

    plasma_coil_reactance = plasma.compute_plasma_coil_reactance(
        electron_temperature_ev=electron_temperature_ev,
        RF_power=rf_power,
        RF_frequency=rf_frequency,
        sheath_voltage=sheath_voltage,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
        pressure_pa=pressure_pa,
    )

    plasma_cap_reactance = plasma.compute_plasma_cap_reactance(
        electron_temperature_ev=electron_temperature_ev,
        RF_power=rf_power,
        RF_frequency=rf_frequency,
        sheath_voltage=sheath_voltage,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
    )

    number_need_to_be_one = plasma.compute_number_need_to_be_one(
        electron_temperature_ev=electron_temperature_ev,
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
        chamber_radius_m=chamber_radius_m,
        chamber_height_m=chamber_height_m,
    )

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
    print(f"Collisional frequency: {f_collisional} /s")
    print(f"Plasma angular frequency: {w_pe} rad/s")
    print(f"Plasma conductivity: {conductivity_pe} S/m")
    print(f"Plasma relative permittivity: {relative_permittivity_pe}")
    print(f"Plasma resistance: {plasma_resistance} ohm")
    print(f"Plasma coil reactance: {plasma_coil_reactance} ohm")
    print(f"Plasma capacitive reactance: {plasma_cap_reactance} ohm")
    print(f"Number that should be 1: {number_need_to_be_one}")


if __name__ == "__main__":
    main()
