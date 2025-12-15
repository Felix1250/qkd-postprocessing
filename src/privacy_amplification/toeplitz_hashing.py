import math
from numbers import Integral, Real
import numpy as np
import scipy as sp
from galois import GF2
from privacy_amplification import pa_utils

# Compute the optimal output length for the (modified) Toeplitz hashing using the quantum Leftover Hash Lemma
def calculate_length(input_length: Integral, relative_source_entropy: Real, error_bound: Real) -> int:
    
    """
    Arguments:
        input_length: The length of the bit string from the weak random source X.

        relative_source_entropy: Lower bound on the conditional min-entropy of the weak random source,
            normalized by the input length (1/n * H_min(X|E) >= relative_source_entropy). It must be a real number in the range (0, 1].

        error_bound: Upper bound on the randomness extractor's error, i.e., a measure of how far the output of the
            extractor is from the ideal (uniform) output. It must be a real number in the range (0, 1].
    """

    # Output length m given by the quantum Leftover Hash Lemma (the formula is the same also for the case with classical side information)
    output_length = math.floor(relative_source_entropy*input_length - 2*math.log2(1 / error_bound) + 2)

    # Extractors cannot extract more uniform bits than available in the weak random source X
    max_output_length = math.floor(input_length * relative_source_entropy) # max_output_length = H_min(X|E)
    if output_length > max_output_length:
        return max_output_length
    elif output_length < 0:
        return 0
    return output_length


# Compute the matrix-vector multiplication (Toeplitz hashing) using the Fast Fourier Transform (FFT) by embedding the Toeplitz matrix into a square circulant matrix
def fast_matrix_vector_multiplication(input_vector: GF2, seed: GF2, output_length: Integral) -> GF2:

    """
    Compute the matrix-vector multiplication: Toeplitz_matrix @ vector_X (in GF(2)) using FFT convolution.

    Arguments:
        input_vector: Binary array from the weak random source X
        seed: Uniform seed used to populate the Toeplitz matrix
        output_length: m bits

    Returns:
        GF2 vector of length m.
    """

    # Pad the input vector X with 0s so that it has the same length of the input seed Y
    padded_input_vector = np.append(input_vector, GF2.Zeros(seed.size - input_vector.size))
    assert padded_input_vector.size == seed.size

    # Compute convolution via FFT: convolution in the time domain == pointwise multiplication in the frequency domain

    # Take FFT of both seed vector and padded input vector (converts both into frequency domain)
    fft_seed = sp.fft.rfft(seed) # computes the discrete Fourier Transform (DFT) using the Fast Fourier Transform (FFT) algorithm
    fft_input = sp.fft.rfft(padded_input_vector)
    # FFT-based multiplication
    fft_result = fft_seed * fft_input

    # Compute Inverse FFT (transform back: returns the convolution result in the time domain)
    result = sp.fft.irfft(fft_result, padded_input_vector.size) # computes the inverse discrete Fourier Transform

    output = (np.round(result).astype(np.uint8) % 2)

    # The Toeplitz multiplication output is only the first m entries --> truncate the output of the Inverse FFT to the first m bits
    return GF2(output[:output_length])


# Implementation of the Standard Toeplitz hashing randomness extractor (FFT-based Toeplitz hashing)
def standard_toeplitz_hashing(extractor_input: GF2, seed: GF2, output_length: Integral) -> GF2:
    
    """
    Given n bits from a weak random source with at least (relative_source_entropy * n)
    bits of entropy, it outputs an almost uniform binary array up to an error error_bound.

    The output of this randomness extractor is the result of doing a matrix-vector multiplication, where the matrix is a
    Toeplitz matrix determined by the uniform seed, and the vector is the bit string from the weak random source.
    The matrix-vector multiplication is computed using the Fast Fourier Transform (FFT) by embedding the Toeplitz matrix into a square circulant matrix. 

    Arguments:
        extractor_input: Binary array (bit string) from the weak random source.
        seed: Uniform seed used to populate the Toeplitz matrix (seed length = m + n - 1).
        output_length: Length of the randomness extractor's output. The optimal output length,
            given some constraints on the tolerated error and the weak random source, is computed using calculate_output_length().

    Returns:
        An almost uniform (up to an error error_bound) binary array.
    """

    secure_key = fast_matrix_vector_multiplication(extractor_input, seed, output_length)

    return secure_key


# Implementation of the Modified Toeplitz hashing randomness extractor (FFT-based Toeplitz hashing)
def modified_toeplitz_hashing(extractor_input: GF2, seed: GF2, output_length: Integral) -> GF2:
    
    """
    The output of the randomness extractor is the result of doing a matrix-vector multiplication.
    The matrix is the result of concatenating a smaller Toeplitz matrix, determined by the uniform seed, together with an
    identity matrix of size output_length. The vector is the bit string from the weak random source.

    Because the Toeplitz matrix has dimension output_length * (input_length - output_length), i.e.,
    m x (n - m), the required seed has length (input_length - 1), i.e., n - 1, smaller than with the Standard Toeplitz hashing:

    seed length = m + (n - m) - 1 = n - 1

    The matrix-vector multiplication is computed using the Fast Fourier Transform (FFT) by
    embedding the Modified Toeplitz matrix into a square circulant matrix.
    """

    n = extractor_input.size
    m = output_length

    # extractor_input = (extractor_input_left, extractor_input_right) --> length = (n - m) + m = n
    extractor_input_left = extractor_input[:n-m]  # length = n - m
    extractor_input_right = extractor_input[n-m:] # length = n - (n - m) = m

    # extractor output = (modified_T_matrix || Id_mxm) @ extractor_input = (modified_T_matrix @ extractor_input_left) XOR extractor_input_right
    secure_key = fast_matrix_vector_multiplication(extractor_input_left, seed, output_length) + GF2(extractor_input_right) # GF2 addition is the same as the logical XOR operation

    return secure_key


def qkd_privacy_amplification(raw_key_alice: GF2, relative_source_entropy, error_bound, use_Modified_TH: bool = False) -> GF2:
    
    """
    QKD privacy amplification step using the Standard Toeplitz hashing by default (use_Modified_TH = False).
    """
    
    # Input length
    n = len(raw_key_alice)

    # Calculate secure output length using the quantum Leftover Hash Lemma
    m = calculate_length(n, relative_source_entropy, error_bound)

    if use_Modified_TH == False:
        # Standard Toeplitz hashing
        seed_length = n + m - 1 
    else:
        # Modified Toeplitz hashing
        seed_length = n - 1

    print("Seed length: ", seed_length)

    # Fetch a random seed (publicly shared): the seed is generated by reading seeds from the seeds.txt file, until the correct length is reached
    seed_str = ""
    while (len(seed_str) < seed_length):
        fetched_seed = pa_utils.read_seed_from_file()
        seed_str = seed_str + fetched_seed

    # Truncate the seed so that its length is equal to seed_length
    seed_str = seed_str[:seed_length]
    assert len(seed_str) == seed_length
    seed = pa_utils.str_to_gf2(seed_str)

    # Apply privacy amplification (Standard or Modified Toeplitz Hashing)
    if use_Modified_TH == False:
        secure_key_alice = standard_toeplitz_hashing(raw_key_alice, seed, m) # Standard Toeplitz hashing --> seed length = n + m - 1
    else:
        secure_key_alice = modified_toeplitz_hashing(raw_key_alice, seed, m) # Modified Toeplitz hashing --> seed length = n - 1

    return secure_key_alice