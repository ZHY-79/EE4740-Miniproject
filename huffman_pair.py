import numpy as np
import heapq
from collections import Counter
import math
import matplotlib.pyplot as plt
class Huffman_Node:
    def __init__(self, symbol, freq, left=None, right=None) -> None:
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, nextNode):
        return self.freq < nextNode.freq

def create_pair_symbols(signal_array):
    signal_array = np.clip(signal_array, -32768, 32767).astype(np.int16)
    
    pair_symbols = []
    for i in range(0, len(signal_array) - 1, 2):
        pair = (signal_array[i], signal_array[i+1])
        pair_symbols.append(pair)
    
    # Handle odd length array by adding a padding sample if needed
    if len(signal_array) % 2 != 0:
        # Use the last sample paired with a zero
        pair = (signal_array[-1], 0)
        pair_symbols.append(pair)
    
    return pair_symbols

def build_huffman_tree(freq_dict):
    priority_queue = []
    
    for symbol, freq in freq_dict.items():
        node = Huffman_Node(symbol=symbol, freq=freq)
        heapq.heappush(priority_queue, node)
    
    if len(priority_queue) == 1:
        node = heapq.heappop(priority_queue)
        new_node = Huffman_Node(symbol=None, freq=node.freq, left=node, right=None)
        return new_node
    
    while len(priority_queue) > 1:
        left_node = heapq.heappop(priority_queue)
        right_node = heapq.heappop(priority_queue)
        
        merged_freq = left_node.freq + right_node.freq
        new_node = Huffman_Node(symbol=None, freq=merged_freq, left=left_node, right=right_node)
        
        heapq.heappush(priority_queue, new_node)
    
    return priority_queue[0]

def generate_huffman_codes(node, current_code, codes_dict):
    if node.left is None and node.right is None:
        codes_dict[node.symbol] = current_code
        return
    
    if node.left:
        generate_huffman_codes(node.left, current_code + "0", codes_dict)
    
    if node.right:
        generate_huffman_codes(node.right, current_code + "1", codes_dict)

def calculate_entropy_and_avg_length(pairs_freq_dict, codes_dict, total_pairs):
    # Calculate probabilities
    probabilities = [freq/total_pairs for freq in pairs_freq_dict.values()]
    
    # Calculate entropy: -sum(p * log2(p))
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    
    # Calculate average code length
    total_bits = 0
    for pair, freq in pairs_freq_dict.items():
        if pair in codes_dict:
            total_bits += len(codes_dict[pair]) * freq
    
    avg_length = total_bits / total_pairs
    entropy = entropy / 2
    # Calculate per-sample metrics
    

    
    return entropy, avg_length

def train_pair_huffman(train_signal_array):
    """
    Train Huffman coding model using consecutive sample pairs (non-overlapping)
    """
    # Create pair symbols (non-overlapping)
    pair_symbols = create_pair_symbols(train_signal_array)
    
    # Calculate frequency of each pair symbol
    pair_freq_dict = Counter(pair_symbols)
    
    # Build Huffman tree
    huffman_root = build_huffman_tree(pair_freq_dict)
    
    # Generate Huffman code table
    codes_dict = {}
    generate_huffman_codes(huffman_root, "", codes_dict)
    
    # Calculate entropy and average code length
    total_pairs = len(pair_symbols)
    entropy, avg_length = calculate_entropy_and_avg_length(
        pair_freq_dict, codes_dict, total_pairs
    )
    
    return huffman_root, codes_dict, entropy, avg_length

def encode_signal_with_pairs(signal_array, codes_dict):
    # Create pair symbols (non-overlapping)
    pair_symbols = create_pair_symbols(signal_array)
    
    # Encode
    encoded_data = ""
    unknown_pairs = 0
    
    # Convert code table keys to list for handling unknown symbols
    known_pairs = list(codes_dict.keys())
    
    for pair in pair_symbols:
        if pair in codes_dict:
            encoded_data += codes_dict[pair]
        else:
            unknown_pairs += 1
            # Use the first known pair's code as a fallback
            encoded_data += codes_dict[known_pairs[0]]
    
    if unknown_pairs > 0:
        print(f"Warning: {unknown_pairs} unknown pairs encountered during encoding")
    
    # Calculate compression metrics
    original_bits = len(signal_array) * 16  # Assuming 16-bit samples
    compressed_bits = len(encoded_data)
    compression_ratio = original_bits / compressed_bits
    compression_rate = (1 - compressed_bits / original_bits) * 100
    
    return encoded_data, compression_ratio, compression_rate

def decode_pair_huffman(encoded_data, huffman_root, original_length):
    # Decode pair symbols
    decoded_pairs = []
    node = huffman_root
    
    for bit in encoded_data:
        if bit == '0':
            node = node.left
        else:
            node = node.right
        
        if node is not None and node.left is None and node.right is None:
            decoded_pairs.append(node.symbol)
            node = huffman_root
    
    # Recover signal from pair symbols - now non-overlapping
    decoded_signal = []
    for pair in decoded_pairs:
        first_value, second_value = pair
        decoded_signal.append(first_value)
        decoded_signal.append(second_value)
    
    # Trim to original length (in case we added padding)
    decoded_signal = decoded_signal[:original_length]
    
    return np.array(decoded_signal, dtype=np.int16)

def check_perfect_reconstruction(original, decoded):
    """
    Check if reconstruction is perfect
    """
    if len(original) != len(decoded):
        print(f"Length mismatch: original={len(original)}, decoded={len(decoded)}")
        return False
    
    # Check element-wise equality
    is_equal = np.array_equal(original, decoded)
    
    if not is_equal:
        # Find first mismatch for debugging
        mismatch_indices = np.where(original != decoded)[0]
        if len(mismatch_indices) > 0:
            first_idx = mismatch_indices[0]
            print(f"First mismatch at index {first_idx}: original={original[first_idx]}, decoded={decoded[first_idx]}")
            
            # Show a few more mismatches
            for i in range(min(5, len(mismatch_indices))):
                idx = mismatch_indices[i]
                print(f"Mismatch at index {idx}: original={original[idx]}, decoded={decoded[idx]}")
    
    return is_equal

def test_train_signal_reconstruction(train_signal):
    """
    Test if the training signal can be perfectly reconstructed
    """
    print("Training Huffman model...")
    huffman_root, codes_dict, entropy, avg_length = train_pair_huffman(train_signal)
    
    print("Encoding training signal...")
    encoded_data, compression_ratio, compression_rate = encode_signal_with_pairs(train_signal, codes_dict)
    
    print("Decoding training signal...")
    decoded_signal = decode_pair_huffman(encoded_data, huffman_root, len(train_signal))
    
    print("Checking reconstruction...")
    is_perfect = check_perfect_reconstruction(train_signal, decoded_signal)
    
    print("\nCompression metrics:")
    print(f"Entropy: {entropy:.4f} bits/pair")
    print(f"Average code length: {avg_length:.4f} bits/pair")
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    print(f"Compression rate: {compression_rate:.2f}%")
    
    if is_perfect:
        print("Training signal perfectly reconstructed")
    else:
        print("Training signal reconstruction has errors")
    
    return (is_perfect, decoded_signal, compression_ratio, compression_rate, 
            entropy, avg_length)

def get_pair_symbol_distribution(signal_array):
    pair_symbols = create_pair_symbols(signal_array)
    freq_dict = Counter(pair_symbols)
    num_unique_pairs = len(freq_dict)
    
    return freq_dict, num_unique_pairs, pair_symbols

def calculate_symbol_distribution(signal_array, plot=True, title="Audio Symbol Probability Distribution"):
    # Ensure 16-bit integer data type
    signal_array = np.clip(signal_array, -32768, 32767).astype(np.int16)
    
    # Count occurrence of each amplitude value (symbol)
    symbol_counts = Counter(signal_array)
    
    # Calculate total symbols (including zeros)
    total_symbols = len(signal_array)
    
    # Calculate probabilities for each symbol
    symbol_probs = {symbol: count/total_symbols for symbol, count in symbol_counts.items()}
    
    # Create result dictionary
    distribution_result = {
        "symbol_counts": dict(symbol_counts),
        "symbol_probabilities": symbol_probs,
        "total_symbols": total_symbols
    }
    
    # Extract symbols and probabilities for plotting
    symbols = list(symbol_probs.keys())
    probabilities = list(symbol_probs.values())
    
    # Calculate entropy (only for non-zero probabilities)
    probs_array = np.array(probabilities)
    non_zero_probs = probs_array[probs_array > 0]  # Filter out zero probabilities
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
    
    
    if plot:
        plt.figure(figsize=(8, 6))
        
        # Standard probability distribution
        plt.bar(symbols, probabilities, width=1.0, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f'{title}\nTotal Samples: {total_symbols}, Entropy: {entropy:.4f} bits')
        plt.xlabel('Symbol Value (Amplitude)')
        plt.ylabel('Probability')
        plt.grid(True, alpha=0.3)
        plt.ylim((0, 0.005))
        plt.tight_layout()
        plt.show()
    
    # Add entropy to result
    distribution_result["entropy"] = entropy
    
    return distribution_result

def calculate_pair_symbol_distribution(signal_array, plot=True, title="Pair Symbol Probability Distribution"):
    # Ensure 16-bit integer data type
    signal_array = np.clip(signal_array, -32768, 32767).astype(np.int16)
    
    # Create pair symbols from consecutive samples
    pair_symbols = []
    for i in range(len(signal_array) - 1):
        # Combine two consecutive samples as a tuple
        pair = (signal_array[i], signal_array[i+1])
        pair_symbols.append(pair)
    
    # Count occurrence of each pair symbol
    pair_counts = Counter(pair_symbols)
    
    # Calculate total pair symbols
    total_pairs = len(pair_symbols)
    
    # Calculate probabilities for each pair symbol
    pair_probs = {pair: count/total_pairs for pair, count in pair_counts.items()}
    
    # Create result dictionary
    distribution_result = {
        "pair_counts": dict(pair_counts),
        "pair_probabilities": pair_probs,
        "total_pairs": total_pairs,
        "unique_pairs": len(pair_counts)
    }
    
    # Extract and sort pairs by probability for plotting
    sorted_pairs = sorted(pair_probs.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate entropy (only for non-zero probabilities)
    probabilities = np.array([prob for _, prob in sorted_pairs])
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    if plot:
        plt.figure(figsize=(14, 10))
        
        # Plot all pairs as a distribution (sorted by probability)
        plt.subplot(2, 1, 1)
        plt.bar(range(len(sorted_pairs)), [prob for _, prob in sorted_pairs], 
                width=1.0, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f'{title}\nTotal Pairs: {total_pairs}, Unique Pairs: {len(pair_counts)}, Entropy: {entropy:.4f} bits')
        plt.xlabel('Pair Index (sorted by probability)')
        plt.ylabel('Probability')
        plt.grid(True, alpha=0.3)
        
        # Log scale view to better see the tail of the distribution
        plt.subplot(2, 1, 2)
        plt.bar(range(len(sorted_pairs)), [prob for _, prob in sorted_pairs], 
                width=1.0, color='lightgreen', edgecolor='black', alpha=0.7)
        plt.yscale('log')
        plt.title(f'Pair Symbol Distribution (Log Scale)')
        plt.xlabel('Pair Index (sorted by probability)')
        plt.ylabel('Probability (Log Scale)')
        plt.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.show()
        
        # Also visualize the cumulative distribution
        plt.figure(figsize=(12, 8))
        cumulative_probs = np.cumsum([prob for _, prob in sorted_pairs])
        plt.plot(range(len(cumulative_probs)), cumulative_probs, 'b-', linewidth=2)
        
        # Add reference lines
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
        plt.axhline(y=0.9, color='g', linestyle='--', alpha=0.7)
        plt.axhline(y=0.99, color='m', linestyle='--', alpha=0.7)
        
        # Find indices where the cumulative probability crosses key thresholds
        idx_50 = np.argmax(cumulative_probs >= 0.5)
        idx_90 = np.argmax(cumulative_probs >= 0.9)
        idx_99 = np.argmax(cumulative_probs >= 0.99)
        
        # Annotate the key points
        plt.annotate(f'50% covered by top {idx_50+1} pairs ({(idx_50+1)/len(pair_counts)*100:.2f}%)', 
                    xy=(idx_50, 0.5), xytext=(idx_50+len(pair_counts)/20, 0.55),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8))
        
        plt.annotate(f'90% covered by top {idx_90+1} pairs ({(idx_90+1)/len(pair_counts)*100:.2f}%)', 
                    xy=(idx_90, 0.9), xytext=(idx_90+len(pair_counts)/20, 0.85),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8))
        
        plt.annotate(f'99% covered by top {idx_99+1} pairs ({(idx_99+1)/len(pair_counts)*100:.2f}%)', 
                    xy=(idx_99, 0.99), xytext=(idx_99+len(pair_counts)/20, 0.94),
                    arrowprops=dict(facecolor='magenta', shrink=0.05, width=1.5, headwidth=8))
        
        plt.title('Cumulative Probability Distribution of Pair Symbols')
        plt.xlabel('Number of Most Frequent Pairs')
        plt.ylabel('Cumulative Probability')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Add entropy to result
    distribution_result["entropy"] = entropy
    
    return distribution_result