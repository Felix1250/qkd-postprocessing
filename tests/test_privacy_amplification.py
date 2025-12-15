import sys
import os
from galois import GF2

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
# Add 'src' to the system path
sys.path.append(os.path.join(project_root, 'src'))

from privacy_amplification import pa_utils
from privacy_amplification import toeplitz_hashing

def test_privacy_amplification(raw_key_alice, relative_source_entropy, error_bound, use_Modified_TH):

    secure_key_alice = toeplitz_hashing.qkd_privacy_amplification(raw_key_alice, relative_source_entropy, error_bound, use_Modified_TH)
    secure_key_alice_str = pa_utils.gf2_to_str(secure_key_alice)

    print("Secret key length: ", len(secure_key_alice_str))
    print("Secret key: ", secure_key_alice_str)

    return

'''
n = 648 bits or 1944 bits (according to the values used in the LDPC code for Information Reconciliation)

 Given n = 648, relative_source_entropy=0.5, error_bound=1e-6 (10^(-6)) --> output length m = 286
 --> Standard Toeplitz hashing: seed_length = n + m - 1 = 933 
 --> Modified Toeplitz hashing: seed_length = n - 1 = 647 

 Given n = 1944, relative_source_entropy=0.5, error_bound=1e-6 (10^(-6)) --> output length m = 934
 --> Standard Toeplitz hashing: seed_length = n + m - 1 = 2877
 --> Modified Toeplitz hashing: seed_length = n - 1 = 1943
'''

#n = 648 
n = 1944
raw_key_alice = GF2.Random(n) # the key obtained after the Information Reconciliation step

test_privacy_amplification(raw_key_alice, relative_source_entropy=0.5, error_bound=1e-6, use_Modified_TH=True)