import random
import numpy as np
import math
import strawberryfields as sf
from strawberryfields import ops
from tqdm import tqdm



import tf_utils

do_random_phases = True

pd0 = 0 #math.pow(10,-50) #darkcount of each detector individually
pd1 = 0 #math.pow(10,-50)

eta1 = 1 #0.9 #detection efficiency of the detectors
eta2 = 1 #0.9 

mu1 = 0.1 #  intesity of decoy states
mu2 = 0.298
mu3 = 0.422 # intensity of the 1 in  X window

p_1 = 0.846 # probability of weak decoy state 
p_2 = 0.076 # probability of strong decoy state

p_x = 0 #1 - 0.735 #probability of decoy state
p_z = 1 - 0.269 # probability of not sending in signal base

loss_1 = 0#math.exp(-100/22) #losses for both distances, as the distances do not have to be equal
loss_2 = 0#math.exp(-100/22)


n_phaseSlice = 8 #number of phase slices
s_phaseSlice = 2 *math.pi /n_phaseSlice # size of a phase slice


def generate_random_phaseShift(seed,length):
    #random.seed(seed)
    psi_AB = np.zeros(length)
    for i in range(0,length):
        psi_AB[i] = random.uniform(0,2*math.pi)
    return psi_AB

def generate_random_Qbits(seed, length):
    random.seed(seed)



    #generate random qubits
    bit_vector = np.zeros(length)
    phase_vector = np.zeros(length)
    amplitude_vector = np.zeros(length)
    window_vector = np.zeros(length)


    # 1 = X window = signal window 
    # 0 = Z window = decoy window
    for i in range(0,length):
        if random.random() < p_x:
            window_vector[i] = 0
            j = random.random()                
            if j < p_1:
                bit_vector[i] = 1
                phase_vector[i] = random.uniform(0,2*math.pi)
                amplitude_vector[i] = np.sqrt(mu1)
            elif j < p_1 + p_2:
                bit_vector[i] = 1
                phase_vector[i] = random.uniform(0,2*math.pi)
                amplitude_vector[i] = np.sqrt(mu2)
            else :
                bit_vector[i] = 0
                phase_vector[i] = 0
                amplitude_vector[i] = 0
        else:
            window_vector[i] = 1

            if random.random() < p_z:
                bit_vector[i] = 0
                phase_vector[i] = 0
                amplitude_vector[i] = 0
            else: 
                bit_vector[i] = 1
                phase_vector[i] = random.uniform(0,2*math.pi)
                amplitude_vector[i] = np.sqrt(mu3)
        if not do_random_phases:
            phase_vector[i] = 0
        
    
    qbits = [window_vector,bit_vector,phase_vector, amplitude_vector]
    return qbits



def run_trial(phase_A, phase_B, amplitude_A, amplitude_B):
    prog = sf.Program(2)
    with prog.context as q:
        # State preparation
        if amplitude_A != 0:
            ops.Coherent(amplitude_A, phase_A) | q[0]
        else:
            ops.Vacuum() | q[0]
        if amplitude_B != 0:
            ops.Coherent(amplitude_B, phase_B) | q[1]
        else:
            ops.Vacuum() | q[1]

        # Channel loss
        if loss_1 > 0:
            ops.LossChannel(loss_1) | q[0]
            ops.LossChannel(loss_2) | q[1]

        # Interference
        ops.BSgate(math.pi/4, math.pi/2) | (q[0], q[1]) #np.pi/4 ? dann aber alle quanten zustände auf einer seite

        # Detection
        ops.MeasureFock() | q

    eng = sf.Engine("fock", backend_options={"cutoff_dim": 10}) # use fock to access single Photons
    result = eng.run(prog)
    for i in range(result.samples[0][0]):
        if random.random() > eta1:
            result.samples[0][0] -= 1
    for i in range(result.samples[0][1]):
        if random.random() > eta2:
            result.samples[0][1] -= 1


    # dark counts. is an additional photon
    if random.random() < pd0:
        result.samples[0][0] += 1
    if random.random() < pd1:
        result.samples[0][1] += 1
    return result.samples[0]
    
'''
do this

'''
def aftercomm(alice_bits,bob_bits,measures,length,psi_AB):
    signal_bits = []
    decoy_bits = []
    
    #determine weather the events are effective

    for i in tqdm(range(length),"effective events"):
        # still have to figure out what to do with the phase
        if alice_bits[0][i] == 0 and bob_bits[0][i] == 0: # check whether both use the Z basis

            if measures[i][0] == 0 and measures[i][0] != measures[i][1]: #only if detector 2 clicks
                decoy_bits.append((i,1))
            elif measures[i][1] == 0 and measures[i][0] != measures[i][1]: #only if detector 1 clicks
                decoy_bits.append((i,0))
        elif alice_bits[0][i] == 1 and bob_bits[0][i] == 1: #check whether both use the x basis
             #check phase slices
            #if math.floor(alice_bits[2][i]/s_phaseSlice) == math.floor(bob_bits[2][i]/s_phaseSlice): if you actually slice the thing in the number of slices
            if abs(alice_bits[2][i] - bob_bits[2][i] - psi_AB[i]) < s_phaseSlice or abs(alice_bits[2][i] - bob_bits[2][i] - psi_AB[i] - math.pi) < s_phaseSlice:
                # chekc if only one detector clicked
                if measures[i][0] == 0 and measures[i][0] != measures[i][1]:
                    signal_bits.append((i,1))
                elif measures[i][1] == 0 and measures[i][0] != measures[i][1]:
                    signal_bits.append((i,0))
    print("signal bits " + str(len(signal_bits)))
    print("decoy bits: " + str(len(decoy_bits)))


    # generate Keys

    alice_key = np.zeros(len(signal_bits))
    bob_key = np.zeros(len(signal_bits))
    for i in range(len(alice_key)):
        if alice_bits[1][signal_bits[i][0]] == 1:
            alice_key[i] = 1
        if bob_bits[1][signal_bits[i][0]] == 0:
            bob_key[i] = 1
    tf_utils.print_error_rate(alice_key,bob_key)
    tf_utils.make_heatmap(measures,"figures/heatmap.png")

    # Active odd parity paring 
    pairs = tf_utils.map_pairs(len(signal_bits))
    #bit mask that if = 1 keeps the bit
    drop_bits = np.zeros(len(signal_bits))
    for i in tqdm(range(len(pairs)),"odd parity paring"):
        parity_alice = alice_key[pairs[i][0]] + alice_key[pairs[i][1]] % 2
        parity_bob = bob_key[pairs[i][0]] + bob_key[pairs[i][1]] % 2
        if parity_bob == parity_alice:
            select_bit = random.randint(0,1)
            drop_bits[pairs[i][select_bit]] = 1
    aopp_length = np.count_nonzero(drop_bits)
    alice_key_aopp = np.zeros(aopp_length)
    bob_key_aopp=  np.zeros(aopp_length)
    counter = 0
    for i in range(len(drop_bits)):
        if drop_bits[i] == 1:
            alice_key_aopp[counter]= alice_key[i]
            bob_key_aopp[counter]= bob_key[i]
            counter +=1

    print("length after aopp: " + str(aopp_length))
    print("percentage reduction: " + str(aopp_length/len(signal_bits)))
    tf_utils.print_error_rate(alice_key_aopp,bob_key_aopp)
    


def tf_communicate(alice_seed, bob_seed,length):
    alice_bits = generate_random_Qbits(alice_seed,length)
    bob_bits = generate_random_Qbits(bob_seed,length)
    psi_AB = generate_random_phaseShift(tf_utils.generate_seed,length)
    measures = np.empty((length, 2))

 
    for i in tqdm(range(length),"Twin field"): #optimize 
        temp =run_trial(  #alice_bits[0][i],bob_bits[0][i], #window also unncecessary
                    #alice_bits[1][i],bob_bits[1][i], # not the bit vector. unnecessary
                    alice_bits[2][i],bob_bits[2][i],
                    alice_bits[3][i],bob_bits[3][i],)
        measures[i][0]= temp[0]
        measures[i][1]= temp[1]
    print(measures)
    sent_photons =sum(alice_bits[3]) + sum(bob_bits[3])
    print("sent photons: " + str(sent_photons) )
    rec_photons = tf_utils.make_heatmap(measures,"figures/heatmap.png")
    print(rec_photons/sent_photons)
    np.savez("output_extra_long.npz",
             first=alice_bits,
             second=bob_bits,
             third=measures,
             fourth=length,
             fifth= psi_AB)
    #data = np.load("output.npz")
    #aftercomm(data["first"],data["second"],data["third"],data["fourth"])

def tf_communicat_load(path="output_long.npz"):
    data = np.load(path)
    aftercomm(data["first"],data["second"],data["third"],data["fourth"],data["fifth"])

tf_communicate(tf_utils.generate_seed(),tf_utils.generate_seed(),10000)
tf_communicat_load("output_extra_long.npz")
#run_trial(0.1,0.1,0,0,0)

#noise(generate_random_Qbits(generate_seed,10,0.5,0.5))