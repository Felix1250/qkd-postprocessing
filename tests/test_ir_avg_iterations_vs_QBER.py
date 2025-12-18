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
# TEST average iteration number of LDPC decoder with parity-check matrix [Hb_324_648] from IEEE 802.11n Standard

# We evaluate how many iterations, on average, the decoding algorithm takes to correct a binary string, as a function of the QBER.
# ===============================================================================================================

def test1_avg_iterations(max_iterations):

    errors_values = [10, 20, 30, 40, 50, 57, 62, 65, 66, 70, 72]
    N = 150  # number of binary strings to decode for each errors in the list
    average_iteration_values_per_qber = []

    # Generate the full parity-check matrix H from Hb_324_648
    z = 27  # submatrix size
    H = mu.expand_ldpc_base_matrix(mu.Hb_324_648, z)
    # Generate the Tanner graph
    vn, cn = mu.generate_tanner_graph(H)

    for errors in tqdm(errors_values):
        qber = errors / 648 # qber = errors / len(string_X)
        total_number_of_iterations = 0 # we will consider only the successfull runs of the algorithm
        successful_runs = 0
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
            decoded_codeword, qber_list, iteration_number = ldpc.ldpc_spa_syndrome_decoder(string_Y, (vn ,cn), qber, max_iterations, syndrome_Sa, H, string_X)
            
            # If decoding was successfull, save the number of iterations it took
            if (np.array_equal(decoded_codeword, string_X) == True):
                successful_runs += 1 # the decoding algorithm succeded to correct the binary string
                total_number_of_iterations += iteration_number

        # If, after trying to decode N binary strings, the number of successful runs is 0,
        # this means that the QBER level is too high and the decoding algorithm will never succeed
        if (successful_runs == 0):
            average_iteration_values_per_qber.append(max_iterations)
        else:
            # Compute the average number of iterations for this QBER level
            average_iteration_values_per_qber.append(total_number_of_iterations / successful_runs)

    # QBER = (errors/len(string_X)) * 100
    QBER_values = [(errors/len(string_X)) * 100 for errors in errors_values]

    # Plot the evolution of average number of iterations vs QBER
    # At low QBER: average number of iterations is expected to be low
    # At high QBER: average number of iterations is expected to increase
    plt.figure()
    plt.plot(QBER_values, average_iteration_values_per_qber, marker='o')
    plt.xlabel('QBER (%)')
    plt.ylabel('Average iteration number of SPA decoding')
    plt.title('Average iteration number vs QBER for SPA using Hb_324_648')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("Test1_avg_iterations.png")  #  Save the figure automatically
    plt.show()

# ===============================================================================================================
# TEST average iteration number of LDPC decoder with parity-check matrix [Hb_972_1944] from IEEE 802.11n Standard

# We evaluate how many iterations, on average, the decoding algorithm takes to correct a binary string, as a function of the QBER.
# ===============================================================================================================

def test2_avg_iterations(max_iterations):

    errors_values = [20, 50, 80, 110, 130, 150, 165, 180, 190, 200, 203, 205, 210, 214]
    N = 150  # number of binary strings to decode for each errors in the list
    average_iteration_values_per_qber = []

    # Generate the full parity-check matrix H from Hb_972_1944
    z = 81  # submatrix size
    H = mu.expand_ldpc_base_matrix(mu.Hb_972_1944, z)
    # Generate the Tanner graph
    vn, cn = mu.generate_tanner_graph(H)

    for errors in tqdm(errors_values):
        qber = errors / 1944 # qber = errors / len(string_X)
        total_number_of_iterations = 0 # we will consider only the successfull runs of the algorithm
        successful_runs = 0
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
            syndrome_Sa = np.mod(H @ string_X.T, 2) # syndrome length == 972 (n - k), H.shape[0]
    
            # Run the decoding algorithm
            decoded_codeword, qber_list, iteration_number = ldpc.ldpc_spa_syndrome_decoder(string_Y, (vn ,cn), qber, max_iterations, syndrome_Sa, H, string_X)
            
            # If decoding was successfull, save the number of iterations it took
            if (np.array_equal(decoded_codeword, string_X) == True):
                successful_runs += 1 # the decoding algorithm succeded to correct the binary string
                total_number_of_iterations += iteration_number

        # If, after trying to decode N binary strings, the number of successful runs is 0,
        # this means that the QBER level is too high and the decoding algorithm will never succeed
        if (successful_runs == 0):
            average_iteration_values_per_qber.append(max_iterations)
        else:
            # Compute the average number of iterations for this QBER level
            average_iteration_values_per_qber.append(total_number_of_iterations / successful_runs)

    # QBER = (errors/len(string_X)) * 100
    QBER_values = [(errors/len(string_X)) * 100 for errors in errors_values]

    # Plot the evolution of average number of iterations vs QBER
    # At low QBER: average number of iterations is expected to be low
    # At high QBER: average number of iterations is expected to increase
    plt.figure()
    plt.plot(QBER_values, average_iteration_values_per_qber, marker='o')
    plt.xlabel('QBER (%)')
    plt.ylabel('Average iteration number of SPA decoding')
    plt.title('Average iteration number vs QBER for SPA using Hb_972_1944')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("Test2_avg_iterations.png")  #  Save the figure automatically
    plt.show()

# ===============================================================================================================

# TEST average iteration number of LDPC decoder with parity-check matrix [Hb_324_648]
#test1_avg_iterations(max_iterations=100)

# TEST average iteration number of LDPC decoder with parity-check matrix [Hb_972_1944]
test2_avg_iterations(max_iterations=100)