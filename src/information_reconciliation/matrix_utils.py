import numpy as np
import random

# ===============================================================================================================
# IEEE 802.11n Standard parity-check matrices (http://ieeexplore.ieee.org/document/5307322/?arnumber=5307322)

# The IEEE 802.11n Standard defines 12 parity-check matrices for LDPC codes in the form of quasi‑cyclic base matrices with circulant shift values (see 802.11n‑2009 Annex R, Tables R.1–R.3).
# The standard defines four code rates R (1/2, 2/3, 3/4 and 5/6), and three block lengths n (648, 1296 and 1944), for a total of 12 possible codes.
# In particular, the matrices are represented in a "base" form Hb and the full parity-check matrix H, of size (n-k) x n, is obtained by
# replacing each entry in the base matrix Hb with the corresponding ZxZ sparse matrix. In particular:
# - for codeword length n=648 bits, subblock size is Z=27 bits (Table R.1)
# - for codeword length n=1296 bits, subblock size is Z=54 bits (Table R.2)
# - for codeword length n=1944 bits, subblock size is Z=81 bits (Table R.3)
# So, each entry in the base matrix Hb represents a ZxZ matrix:
# "-1" --> a ZxZ zero matrix (all zeros)
# "0"  --> a ZxZ identity matrix
# "p"  --> a ZxZ identity matrix cyclically shifted right by p positions
# ===============================================================================================================

# Parity-check matrix (a) from Table R.1 (Annex R): H is 324 x 648
# - codeword block length n=648 bits
# - subblock size Z=27 bits
# - coding rate R=1/2 
Hb_324_648 = [
    [0, -1, -1, -1, 0, 0, -1, -1, 0, -1, -1, 0, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [22, 0, -1, -1, 17, -1, 0, 0, 12, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [6, -1, 0, -1, 10, -1, -1, -1, 24, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, -1, -1,  0, 20, -1, -1, -1, 25, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
    [23, -1, -1, -1, 3, -1, -1, -1, 0, -1, 9, 11, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
    [24, -1, 23, 1, 17, -1,  3, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
    [25, -1, -1, -1, 8, -1, -1, -1, 7, 18, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
    [13, 24, -1, -1, 0, -1, 8, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
    [7, 20, -1, 16, 22, 10, -1, -1, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1],
    [11, -1, -1, -1, 19, -1, -1, -1, 13, -1, 3, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
    [25, -1, 8, -1, 23, 18, -1, 14, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
    [3, -1, -1, -1, 16, -1, -1, 2, 25, 5, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]
]

# Parity-check matrix (a) from Table R.3 (Annex R): H is 972 x 1944
# - codeword block length n=1944 bits
# - subblock size Z=81 bits
# - coding rate R=1/2 
Hb_972_1944 = [
    [57, -1, -1, -1, 50, -1, 11, -1, 50, -1, 79, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, -1, 28, -1, 0, -1, -1, -1, 55, 7, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [30, -1, -1, -1, 24, 37, -1, -1, 56, 14, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
    [62, 53, -1, -1, 53, -1, -1, 3, 35, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
    [40, -1, -1, 20, 66, -1, -1, 22, 28, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
    [0, -1, -1, -1, 8, -1, 42, -1, 50, -1, -1, 8, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
    [69, 79, 79, -1, -1, -1, 56, -1, 52, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
    [65, -1, -1, -1, 38, 57, -1, -1, 72, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
    [64, -1, -1, -1, 14, 52, -1, -1, 30, -1, -1, 32, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1],
    [-1, 45, -1, 70, 0, -1, -1, -1, 77, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
    [2, 56, -1, 57, 35, -1, -1, -1, -1, -1, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
    [24, -1, 61, -1, 60, -1, -1, 27, 51, -1, -1, 16, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]
]


# ===============================================================================================================
# HELPER FUNCTIONS to expand the base matrix Hb into a full parity-check matrix H
# ===============================================================================================================

# Return a ZxZ identity matrix cyclically shifted right by p positions.
# A cyclic right-shift of the identity matrix by p means: row i has a 1 at column (i + p) mod Z
def shift_identity_matrix(z, p):
    I = np.eye(z, dtype=int)
    return np.roll(I, p, axis=1) # we are shifting along columns (axis 1 is columns)

# Expand base LDPC matrix Hb into full H matrix using subblock size Z:
# if entry in Hb is "-1" --> replace with a ZxZ zero matrix (all zeros)
# if entry in Hb is "0"  --> replace with a ZxZ identity matrix
# if entry in Hb is "p"  --> replace with a ZxZ identity matrix cyclically shifted right by p positions
def expand_ldpc_base_matrix(Hb, z):
    rows = []
    for row in Hb:
        expanded_row = []
        for val in row:
            if val == -1:
                block = np.zeros((z, z), dtype=int)
            elif val == 0:
                block = np.eye(z, dtype=int)
            else:  # positive shift value
                block = shift_identity_matrix(z, val)
            expanded_row.append(block)
        rows.append(np.hstack(expanded_row))
    return np.vstack(rows)


# ===============================================================================================================
# CHECK the size of the full-parity check matrix H
# ===============================================================================================================

H_324_648 = expand_ldpc_base_matrix(Hb_324_648, z=27)
assert H_324_648.shape == (324, 648)

H_972_1944 = expand_ldpc_base_matrix(Hb_972_1944, z=81)
assert H_972_1944.shape == (972, 1944)


# ===============================================================================================================
# Generate the Tanner graph corresponding to the parity-check matrix H. If H has size (n-k) x n, the Tanner graph consists of:
# --> (n-k) check nodes (c-nodes): the number of parity bits
# --> (n) variable nodes (v-nodes): the number of bits in a codeword
# Check node i is connected to variable node j if the element h_ij of H is a 1
# ===============================================================================================================

def generate_tanner_graph(parity_check_matrix):
    variable_nodes = [] # each element is a list containing the check nodes connected to the corresponding variable node (with index equal to the index of the list)
    check_nodes = []    # each element is a list containing the variable nodes connected to the corresponding check node
    columns = len(parity_check_matrix[0])
    rows = len(parity_check_matrix)

    # Iterate over the columns of the parity-check matrix, one column at a time (each column corresponds to a variable node)
    for col_index in range(columns):
        check_nodes_list = [] # list containing the check nodes connected to the variable node
        for row_index, row in enumerate(parity_check_matrix):
          if row[col_index] == 1:
              check_nodes_list.append(row_index) # row_index corresponds to the index of the check node
        variable_nodes.append(check_nodes_list)      

    # Iterate horizontally over the parity-check matrix, one row at a time (each row corresponds to a check node)
    for row in parity_check_matrix:
        variable_nodes_list = [] # list containing the variable nodes connected to the check node
        for elem_index, elem in enumerate(row):
            if elem == 1:
                variable_nodes_list.append(elem_index) # elem_index corresponds to the index of the variable node
        check_nodes.append(variable_nodes_list) 

    return variable_nodes, check_nodes


# ===============================================================================================================
# Test function for adding n errors to a valid codeword in accordance with given level of quantum bit error rate (QBER)
# ===============================================================================================================

def introduce_n_errors(bit_string, n):
    string_list = list(bit_string)

    # Pick unique positions to flip
    random_positions = random.sample(range(len(string_list)), n)

    for pos in random_positions:
            if string_list[pos] == "0":
                string_list[pos] = "1"
            else:
                string_list[pos] = "0"
       
    return ''.join(string_list)


# ===============================================================================================================
# Test function for generating a random bit string with given length (used for generating a random key in the tests)
# ===============================================================================================================

def generate_random_bit_string(length):
    random_bit_string = ""
    for i in range(length):
        random_bit_string += str(random.randint(0, 1))
    return random_bit_string