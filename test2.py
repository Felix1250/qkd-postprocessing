import strawberryfields as sf
from strawberryfields import ops
import numpy as np
import random

def run_trial(send_A, send_B, alpha_A, alpha_B, phase_A, phase_B loss=0.0):
    prog = sf.Program(2)

    with prog.context as q:
        # State preparation
        if send_A:
            ops.Dgate(alpha_A) | q[0]
        else:
            ops.Vacuum() | q[0]
        if send_B:
            ops.Dgate(alpha_B) | q[1]
        else:
            ops.Vacuum() | q[1]

        # Channel loss
        if loss > 0:
            ops.LossChannel(loss) | q[0]
            ops.LossChannel(loss) | q[1]

        # Interference
        ops.BSgate(np.pi/4, 0) | (q[0], q[1])

        #phase shift
        

        # Detection
        ops.MeasureFock() | q

    eng = sf.Engine("fock", backend_options={"cutoff_dim": 10}) # use fock to access single Photons
    result = eng.run(prog)
    return result.samples[0]


# Simulation parameters
N = 500
decoy_levels = [0.1, 0.3, 0.5]

data = []

for _ in range(N):
    send_A = random.choice([0, 1])
    send_B = random.choice([0, 1])

    alpha_A = random.choice(decoy_levels)
    alpha_B = random.choice(decoy_levels)

    clicks = run_trial(send_A, send_B, alpha_A, alpha_B)

    data.append((clicks))



    
print(data)