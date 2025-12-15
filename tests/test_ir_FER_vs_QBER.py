import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
# Add 'src' to the system path
sys.path.append(os.path.join(project_root, 'src'))

from information_reconciliation import ldpc_spa_decoder_qkd as ldpc
from information_reconciliation import matrix_utils as mu

# ===============================================================================================================
# TEST Frame Error Rate (FER) with parity-check matrix [Hb_324_648] from IEEE 802.11n Standard

# FER = (number of incorrectly decoded frames) / total number of transmitted frames
# --> it measures how many entire codewords (frames) are decoded incorrectly
# To track the FER, we count the number of codewords that are incorrect, after each full decoding
# (the algorithm reaches max iterations), over a total of N codewords (random strings with a certain QBER)
# ===============================================================================================================

def test1_FER(iterations):

    errors_values = [10, 20, 30, 40, 50, 55, 60, 62, 65, 70]
    N = 150  # number of frames (keys) to decode for each errors in the list
    FER_values = []

    # Generate the full parity-check matrix H from Hb_324_648
    z = 27  # submatrix size
    H = mu.expand_ldpc_base_matrix(mu.Hb_324_648, z)
    # Generate the Tanner graph
    vn, cn = mu.generate_tanner_graph(H)

    for errors in tqdm(errors_values):
        qber = errors / 648 # qber = errors / len(string_X)
        incorrectly_decoded_frames = 0
        for _ in range(N):
            # Parameters (key length == 648 bits)
            # Alice's binary string X (corresponds to the valid codeword)
            string_X = mu.generate_random_bit_string(648) # n = H.shape[1] # Codeword length
            # Bob's binary string Y=X+E (corresponds to the wrong codeword)
            string_Y = mu.introduce_n_errors(string_X, errors)

            diff_count = sum(bit1 != bit2 for bit1, bit2 in zip(string_X, string_Y)) # count positions where the two codewords differ
            assert diff_count == errors # check that the number of positions where the two strings differ is equal to the errors introduced

            # Alice computes the syndrome S = H @ X.T (mod 2) and sends it to Bob through the classical authenticated channel (error-free)
            string_X = np.array([int(bit) for bit in string_X]) # from binary string to np array (true_codeword must be a numpy array)
            syndrome_Sa = np.mod(H @ string_X.T, 2) # syndrome length == 324 (n - k), H.shape[0]
    
            # Run the decoding algorithm
            decoded_codeword, qber_list, iteration_number = ldpc.ldpc_spa_syndrome_decoder(string_Y, (vn ,cn), qber, iterations, syndrome_Sa, H, string_X)
            
            # Check decoding correctness: if decoding failed, increase the number of incorrectly decoded frames
            if (np.array_equal(decoded_codeword, string_X) != True):
                incorrectly_decoded_frames += 1

        # Compute the FER for this QBER level (errors)
        FER_values.append(incorrectly_decoded_frames / N) # 0 <= FER value <= 1


    # QBER = (errors/len(string_X)) * 100
    QBER_values = [(errors/len(string_X)) * 100 for errors in errors_values]

    # Plot FER evolution vs QBER
    # At low QBER: FER \approx 0 --> all frames decoded correctly
    # At high QBER: FER \approx 1 --> almost all frames fail (discarded)
    # The transition region (where FER rises sharply) shows the decoder’s correction limit
    plt.figure()
    plt.plot(QBER_values, FER_values, marker='o')
    plt.xlabel('QBER (%)')
    plt.ylabel('Frame Error Rate (FER)')
    #plt.title('FER vs QBER — LDPC-based Information Reconciliation (max iterations = %d)' % iterations)
    plt.title('FER vs QBER for SPA using Hb_324_648')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# ===============================================================================================================
# TEST Frame Error Rate (FER) with parity-check matrix [Hb_972_1944] from IEEE 802.11n Standard

# FER = (number of incorrectly decoded frames) / total number of transmitted frames
# --> it measures how many entire codewords (frames) are decoded incorrectly
# To track the FER, we count the number of codewords that are incorrect, after each full decoding
# (the algorithm reaches max iterations), over a total of N codewords (random strings with a certain QBER)
# ===============================================================================================================

def test2_FER(iterations):

    errors_values = [20, 50, 80, 110, 130, 150, 165, 175, 180, 190, 200]
    N = 150  # number of frames (keys) to decode for each errors in the list
    FER_values = []

    # Generate the full parity-check matrix H from Hb_972_1944
    z = 81  # submatrix size
    H = mu.expand_ldpc_base_matrix(mu.Hb_972_1944, z)
    # Generate the Tanner graph
    vn, cn = mu.generate_tanner_graph(H)

    for errors in tqdm(errors_values):
        qber = errors / 1944 # qber = errors / len(string_X)
        incorrectly_decoded_frames = 0
        for _ in range(N):
            # Parameters (key length == 1944 bits)
            # Alice's binary string X (corresponds to the valid codeword)
            string_X = mu.generate_random_bit_string(1944) # n = H.shape[1] # Codeword length
            # Bob's binary string Y=X+E (corresponds to the wrong codeword)
            string_Y = mu.introduce_n_errors(string_X, errors)

            diff_count = sum(bit1 != bit2 for bit1, bit2 in zip(string_X, string_Y)) # count positions where the two codewords differ
            assert diff_count == errors # check that the number of positions where the two strings differ is equal to the errors introduced

            # Alice computes the syndrome S = H @ X.T (mod 2) and sends it to Bob through the classical authenticated channel (error-free)
            string_X = np.array([int(bit) for bit in string_X]) # from binary string to np array (true_codeword must be a numpy array)
            syndrome_Sa = np.mod(H @ string_X.T, 2) # syndrome length == 324 (n - k), H.shape[0]
    
            # Run the decoding algorithm
            decoded_codeword, qber_list, iteration_number = ldpc.ldpc_spa_syndrome_decoder(string_Y, (vn ,cn), qber, iterations, syndrome_Sa, H, string_X)
            
            # Check decoding correctness: if decoding failed, increase the number of incorrectly decoded frames
            if (np.array_equal(decoded_codeword, string_X) != True):
                incorrectly_decoded_frames += 1

        # Compute the FER for this QBER level (errors)
        FER_values.append(incorrectly_decoded_frames / N) # 0 <= FER value <= 1


    # QBER = (errors/len(string_X)) * 100
    QBER_values = [(errors/len(string_X)) * 100 for errors in errors_values]

    # Plot FER evolution vs QBER
    # At low QBER: FER \approx 0 --> all frames decoded correctly
    # At high QBER: FER \approx 1 --> almost all frames fail (discarded)
    # The transition region (where FER rises sharply) shows the decoder’s correction limit
    plt.figure()
    plt.plot(QBER_values, FER_values, marker='o')
    plt.xlabel('QBER (%)')
    plt.ylabel('Frame Error Rate (FER)')
    #plt.title('FER vs QBER — LDPC-based Information Reconciliation (max iterations = %d)' % iterations)
    plt.title('FER vs QBER for SPA using Hb_972_1944')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("Test2-FER.png")  #  Save the figure automatically
    plt.show()

# ===============================================================================================================

# TEST Frame Error Rate (FER) with parity-check matrix [Hb_324_648]
#test1_FER(iterations=100) # 150*7 = 1050 keys (each of length 648)

# TEST Frame Error Rate (FER) with parity-check matrix [Hb_972_1944]
test2_FER(iterations=100) # 150*11 = 1650 keys (each of length 1944)