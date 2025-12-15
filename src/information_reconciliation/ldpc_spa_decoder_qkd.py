import math
import numpy as np

# ============================================================================================================================
# Iterative decoding of binary LDPC codes using the Sum-Product Algorithm (SPA), a soft decision message-passing algorithm.
# The decoding steps of information reconciliation for QKD is similar to the classical SPA except the modification at step 2.
# ============================================================================================================================
def ldpc_spa_syndrome_decoder(string_Y, tanner_graph, qber, max_iterations, syndrome_Sa, parity_check_matrix, string_X):

    """
    Sum-Product (LLR) decoder (syndrome-based) for LDPC in a QKD Information Reconciliation scenario. The BSC channel is assumed.

    - string_Y: Bob's binary string Y=X+E (Bob's received bits with errors), equivalent to a received codeword in an error-correcting algorithm
    - tanner_graph: (variable_nodes, check_nodes)
    - qber: Estimated crossover probability of the quantum channel (0 < qber < 1)
    - max_iterations: Maximum number of iterations
    - syndrome_Sa: The syndrome Sa = H@X.T (mod 2) computed by Alice and sent to Bob through the classical authenticated channel
    - parity_check_matrix: H, used for the termination check
    - string_X: Alice's original binary string, equivalent to the error-free original codeword. It is only used to keep track of the QBER during simulation, not for decoding (in a real scenario, this is not available to Bob)
    
    Returns: (decoded_codeword, qber_per_iteration, iter_num)
    """
    
    variable_nodes = tanner_graph[0] # list of lists: each list index is a variable node and each list contains the check nodes connected to that variable node
    check_nodes = tanner_graph[1] # list of lists: each list index is a check node and each list contains the variable nodes connected to that check node

    msg_received_by_cn = {} # used to store the message sent by a variable node to a check node
    msg_received_by_vn = {} # used to store the message sent by a check node to a variable node

    # Pre-calculate the initial LLRs and store them for efficiency
    initial_llr = np.zeros(len(string_Y)) # each element in the array is the initial channel LLR for the corresponding variable node
    for i, bit in enumerate(string_Y): # the index i of the codeword corresponds also to the index of the variable node
        if bit == "0":
            initial_llr[i] = math.log((1-qber)/qber)
        else: # bit == "1"
            initial_llr[i] = math.log(qber/(1-qber))


    # Step 1: Initialization (at this stage, a v-node only knows the corresponding received bit of the codeword)
    for vn_index, cn_list in enumerate(variable_nodes):
        for cn in cn_list:
            if cn not in msg_received_by_cn:
                msg_received_by_cn[cn] = {}
            # Store the log-likelihood ratio (LLR) for a Binary Symmetric Channel (BSC) as a message sent by the v-node "vn_index" to the c-node "cn"
            '''
            if codeword[vn_index] == "0":
                msg_received_by_cn[cn][vn_index] = math.log((1-qber)/qber)
            elif codeword[vn_index] == "1":
                msg_received_by_cn[cn][vn_index] = math.log(qber/(1-qber))
            '''
            msg_received_by_cn[cn][vn_index] = initial_llr[vn_index]

    qber_per_iteration = []  # to store QBER evolution
    qber_per_iteration.append(qber)

    # ITERATE steps 2, 3 and termination check
    for iter_num in range(1, max_iterations + 1):

        # Step 2: CN --> VN messages
        msg_received_by_vn = {} # a VN should only store the latest messages coming from the CNs in that iteration
        for cn, vn_dict in msg_received_by_cn.items():
            for vn in vn_dict: # for every variable node "vn" connected to the check node "cn"
                product = 1.0
                for vn2, llr in vn_dict.items():
                    if vn2 != vn: # do not consider the current v-node "vn"
                        tanh_term = math.tanh(llr / 2.0) # llr == msg_received_by_cn[cn][vn2]
                        product *= tanh_term 


                # --- MODIFICATION FOR QKD  SYNDROME DECODING ---
                # In the standard SPA algorithm, a check node assumes the sum of its connected variable nodes (bits) must be 0 (even parity).
                # In QKD syndrome decoding, the sum must equal the corresponding syndrome bit, s_j.
                # If the syndrome bit is 1, the sum must be 1 (odd parity), which flips the sign of the belief product (a log-likelihood ratio).
                # This is equivalent to multiplying by (-1)^s_j
                if syndrome_Sa[cn] == 1:
                    product *= -1.0

                # Clip product to avoid atanh overflow
                product = max(min(product, 0.999999), -0.999999) # keep product between -0.999999 and +0.999999
                #msg_from_cn_to_vn = math.log((1 + product) / (1 - product))
                msg_from_cn_to_vn = 2.0 * math.atanh(product)

                # store the message sent by cn to vn
                if vn not in msg_received_by_vn:
                    msg_received_by_vn[vn] = {}
                msg_received_by_vn[vn][cn] = msg_from_cn_to_vn


        # Step 3: VN --> CN messages
        for vn, cn_dict in msg_received_by_vn.items():
            for cn in cn_dict: # for every check node "cn" connected to the variable node "vn"
                sum = 0
                for cn2, llr in cn_dict.items():
                    if cn2 != cn: # do not consider the current c-node "cn"
                        sum += llr # llr == msg_received_by_vn[vn][cn2]
                
                msg_from_vn_to_cn = initial_llr[vn] + sum

                # the v-node can now send the update message to the check node "cn"
                msg_received_by_cn[cn][vn] = msg_from_vn_to_cn


        decoded_codeword = np.zeros(len(string_Y), dtype=int) # store current decoded estimate of the codeword

        # Total LLR + Termination Check
        for vn, cn_dict in msg_received_by_vn.items():
            total_sum = 0
            for cn, llr in cn_dict.items(): # we now use the information from every c-node
                total_sum += llr
            
            total_llr_for_vn = initial_llr[vn] + total_sum

            # The v-node "vn" uses the total llr to update its current estimation of the variable c_i (i-th bit of the received codeword)
            if total_llr_for_vn < 0:
                current_estimate_i = 1
            else:
                current_estimate_i = 0

            decoded_codeword[vn] = current_estimate_i

        # ---- QBER TRACKING for simulation ----
        bit_errors = np.sum(decoded_codeword != string_X)
        current_qber = bit_errors / len(string_Y)
        qber_per_iteration.append(current_qber)  # store QBER of this iteration
        
        '''
        After each iteration, Bob calculates the syndrome of the current best guess (decoded_codeword) and compares it directly to the syndrome that Alice sent.
        If they match, the decoding is successful.
        '''
        # The syndrome computed by Bob: Sb = H @ Y.T (mod 2). If Sb == Sa, then Y == X and the algorithm stops.
        syndrome_Sb = np.mod(parity_check_matrix @ decoded_codeword.T, 2)
        # Check if H @ Y.T (mod 2) == syndrome_Sa
        if np.array_equal(syndrome_Sb, syndrome_Sa):
            # Success --> return the key (Y=X+E has been corrected to X, Alice and Bob now share the same binary string)
            return decoded_codeword, qber_per_iteration, iter_num

    # max iterations reached: failure --> discard Y
    return decoded_codeword, qber_per_iteration, iter_num 