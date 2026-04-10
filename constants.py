import numpy as np

# Simulation Constants
DELTA_T = 0.01         # Time step
PASSOS_TEMPO = 200     # Number of simulation steps
EPSILON = 1.0           # Softening parameter
G = 1.0                 # Gravitational constant
MASSA_MIN = 10.0        # Minimum particle mass
MASSA_MAX = 50.0        # Maximum particle mass
VELOCIDADE_MIN = -1.0   # Minimum particle velocity component
VELOCIDADE_MAX = 1.0    # Maximum particle velocity component

# Benchmark/System Constants
DENSIDADE_ALVO = 0.002 # Target density for dynamic box size calculation