import numpy as np
from collections import Counter
def calculate_mse(original_signal, reconstructed_signal, bit_depth=16):
    # Take the shorter length of the two signals
    
    squared_diff = np.square(original_signal.astype(float) - reconstructed_signal.astype(float))
    mse = np.mean(squared_diff)
    
    # Normalize MSE based on bit depth
    # if bit_depth == 8:
        # For 8-bit, range is 256 values (2^8)
        # dynamic_range = 256 ** 2
    # else:  # 16-bit
        # For 16-bit, range is 65536 values (2^16)
    #     dynamic_range = 65536 ** 2
    
    # Return only the normalized MSE
    total_se = np.sum(squared_diff)
    
    return total_se, mse

def calculate_compression_Ratio(train_signal_array, num_bit, encoded_data_bit):

    original_bitstream_size = len(train_signal_array) * num_bit
    compressed_bitstream_size = len(encoded_data_bit)
    compression_ratio = original_bitstream_size / compressed_bitstream_size
    compression_rate = (1 - compressed_bitstream_size / original_bitstream_size) * 100
    return compression_ratio, compression_rate

def calculate_entropy(data):
    counts = Counter(data)
    probabilities = np.array([count / len(data) for count in counts.values()])
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy