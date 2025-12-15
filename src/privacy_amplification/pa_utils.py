import os
import numpy as np
from galois import GF2

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "seeds.txt")

# Read the first bitstring (seed) in the file and remove it: consume one seed per call, so next call gets the next seed
def read_seed_from_file():
    with open(file_path, "r") as f:
        first_line = f.readline()
        remaining = f.read() # everything else

    if not first_line:
        raise ValueError(f"No seeds left in file: {file_path}")

    # Take the first seed
    bitstring = first_line.strip()
    assert len(bitstring) == 512 # each line in the file is 512 bits long

    # Write back the remaining lines (without the first line)
    with open(file_path, "w") as f:
        f.write(remaining)

    #print("Loaded:", bitstring)
    return bitstring

def print_seeds_from_file():
    with open(file_path) as f:
        for line in f:
            bitstring = line.strip()
            assert len(bitstring) == 512 # each line in the file is 512 bits long
            print("Loaded:", bitstring, "length:", len(bitstring))

# Converts a GF(2) array into a bit-string
def gf2_to_str(gf2_arr):
    arr = np.array(gf2_arr)
    arr_str = (arr + ord("0")).tobytes()
    return arr_str.decode()

# Converts a bit-string into a GF(2) array
def str_to_gf2(s):
    b = s.encode()
    arr = np.frombuffer(b, dtype=np.uint8)
    arr = arr - ord("0")
    return GF2(arr)