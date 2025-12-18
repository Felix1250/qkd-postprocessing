import sys
import os
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
# Add 'src' to the system path
sys.path.append(os.path.join(project_root, 'src'))

from information_reconciliation import ldpc_spa_decoder_qkd as ldpc
from information_reconciliation import matrix_utils as mu

# ===============================================================================================================
# TEST [QBER vs Iterations] with parity-check matrix [Hb_324_648] from IEEE 802.11n Standard
# QBER = (number of incorrect bits) / (total number of bits)
# Display how the QBER evolves after each iteration of the decoding algorithm (unitl convergence to 0 or fail to decode)
# ===============================================================================================================

def test1_QBER(errors, iterations):

    # Generate the full parity-check matrix H from Hb_324_648
    z = 27  # submatrix size
    H = mu.expand_ldpc_base_matrix(mu.Hb_324_648, z)

    # Generate the Tanner graph
    vn, cn = mu.generate_tanner_graph(H)

    # Parameters (key length == 648 bits)
    # Alice's binary string X (corresponds to the valid codeword)
    string_X = mu.generate_random_bit_string(648) # n = H.shape[1] # Codeword length
    # Bob's binary string Y=X+E (corresponds to the wrong codeword)
    string_Y = mu.introduce_n_errors(string_X, errors)
    
    qber = errors / len(string_X)
    QBER = (errors/len(string_X)) * 100
    diff_count = sum(bit1 != bit2 for bit1, bit2 in zip(string_X, string_Y)) # count positions where the two codewords differ
    assert diff_count == errors # check that the number of positions where the two strings differ is equal to the errors introduced

    # Alice computes the syndrome Sa = H @ X.T (mod 2) and sends it to Bob through the classical authenticated channel (which is error-free)
    string_X = np.array([int(bit) for bit in string_X]) # from binary string to np array (string_X must be a numpy array)
    syndrome_Sa = np.mod(H @ string_X.T, 2) # syndrome length == 324 (n - k), H.shape[0]
    
    # Run the decoding algorithm
    decoded_codeword, qber_list, iteration_number = ldpc.ldpc_spa_syndrome_decoder(string_Y, (vn ,cn), qber, iterations, syndrome_Sa, H, string_X)

    # Check decoding correctness
    if np.array_equal(decoded_codeword, string_X) == False:
        raise ValueError("Max iterations reached! The decoding algorithm failed.")
    
    QBER_values = [q * 100 for q in qber_list]

    # Print information
    print("Test 1")
    print("Codeword Length:", len(string_X))
    print(f"Initial QBER: {QBER:.3f}%")
    print("Iterations:", iteration_number)

    # Display QBER evolution
    plt.plot(QBER_values, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("QBER (%)")
    plt.title("QBER vs Iterations for SPA using Hb_324_648")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

    return


# ===============================================================================================================
# TEST [QBER vs Iterations] with parity-check matrix [Hb_972_1944] from IEEE 802.11n Standard
# QBER = (number of incorrect bits) / (total number of bits)
# Display how the QBER evolves after each iteration of the decoding algorithm (unitl convergence to 0 or fail to decode)
# ===============================================================================================================

def test2_QBER(errors, iterations):

    # Generate the full parity-check matrix H from Hb_972_1944
    z = 81  # submatrix size
    H = mu.expand_ldpc_base_matrix(mu.Hb_972_1944, z)

    # Generate the Tanner graph
    vn, cn = mu.generate_tanner_graph(H)

    # Parameters (key length == 1944 bits)
    # Alice's binary string X (corresponds to the valid codeword)
    string_X = mu.generate_random_bit_string(1944) # n = H.shape[1] # Codeword length
    # Bob's binary string Y=X+E (corresponds to the wrong codeword)
    string_Y = mu.introduce_n_errors(string_X, errors)

    qber = errors / len(string_X)
    QBER = (errors/len(string_X)) * 100
    diff_count = sum(bit1 != bit2 for bit1, bit2 in zip(string_X, string_Y)) # count positions where the two codewords differ
    assert diff_count == errors # check that the number of positions where the two strings differ is equal to the errors introduced

    # Alice computes the syndrome S = H @ X.T (mod 2) and sends it to Bob through the classical authenticated channel (which is error-free)
    string_X = np.array([int(bit) for bit in string_X]) # from binary string to np array (string_X must be a numpy array)
    syndrome_Sa = np.mod(H @ string_X.T, 2) # syndrome length == 972 (n - k), H.shape[0]

    # Run the decoding algorithm
    decoded_codeword, qber_list, iteration_number = ldpc.ldpc_spa_syndrome_decoder(string_Y, (vn ,cn), qber, iterations, syndrome_Sa, H, string_X)

    # Check decoding correctness
    if np.array_equal(decoded_codeword, string_X) == False:
        raise ValueError("Max iterations reached! The decoding algorithm failed.")
    
    QBER_values = [q * 100 for q in qber_list]

    # Print information
    print("Test 2")
    print("Codeword Length:", len(string_X))
    print(f"Initial QBER: {QBER:.3f}%")
    print("Iterations:", iteration_number)

    # Display QBER evolution
    plt.plot(QBER_values, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("QBER (%)")
    plt.title("QBER vs Iterations for SPA using Hb_972_1944")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

    return

# ===============================================================================================================

'''
number_of_failures = 0
number_of_successes = 0
for _ in range(150):
    result = test1_QBER(errors=60, iterations=100)
    if result == 1:
        number_of_failures += 1
    else:
        number_of_successes += 1

print("Failures: ", number_of_failures)
print("Successes: ", number_of_successes)
'''

# parity-check matrix [Hb_324_648]
# --> key length == 648 bits (n)
# --> syndrome length == 324 (n - k)
#test1_QBER(errors=60, iterations=100) # Max errors == 60?


# parity-check matrix [Hb_972_1944]
# --> key length == 1944 bits (n)
# --> syndrome length == 972 bits (n - k)
test2_QBER(errors=182, iterations=100) # Max errors == 180?