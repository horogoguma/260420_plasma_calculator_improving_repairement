"""Simple workflow test script."""

from src.app import FixedInputs, format_simulation_result, run_single_simulation


def main() -> None:
    inputs = FixedInputs(
        chamber_height_mm=9.0,
        chamber_radius_mm=170.0,
        electrode_radius_mm=150.0,
        pressure_torr=3.5,
        temperature_k=423.0,
        electron_temperature_ev=1.5,
        sheath_voltage=441.0,
        sheath_length_electrode_mm=1.035,
        sheath_length_grounded_mm=1.035,
        rf_power=900.0,
        rf_frequency=12.9e6,
    )
    result = run_single_simulation(inputs)
    print(format_simulation_result(result))
    


if __name__ == "__main__":
    main()
