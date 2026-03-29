import PySpice
import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

logger = Logging.setup_logging()

circuit = Circuit('Voltage Divider')

circuit.V('input', 'vin', circuit.gnd, 10@u_V)
circuit.R(1, 'vin', 'vout', 9@u_kOhm)
circuit.R(2, 'vout', circuit.gnd, 1@u_kOhm)

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.operating_point()

print(float(analysis.nodes['vin'][0]))
print(float(analysis.nodes['vout'][0]))