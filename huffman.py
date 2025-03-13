import numpy as np
import heapq
import matplotlib.pyplot as plt
import math


class Huffman_Node:
    def __init__(self, symbol, freq, left=None, right=None) -> None:
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, nextNode):
        return self.freq < nextNode.freq
        # compare huffman node
    
def get_unique(signal_array):
    # {symbol: counts}
    unique_values, num_counts = np.unique(signal_array, return_counts=True)
    num_unique_values = len(unique_values)
    freq_dict = dict(zip(unique_values, num_counts))
    return freq_dict, num_unique_values

def create_huffman_tree(freq_dict):
    queue = []
    for symbol, freq in freq_dict.items():
        node = Huffman_Node(symbol=symbol, freq=freq)
        heapq.heappush(queue, node)

# create huffman tree
    while len(queue) > 1:
        node_1 = heapq.heappop(queue)
        node_2 = heapq.heappop(queue)
        merged_freq = node_1.freq + node_2.freq
        new_node = Huffman_Node(symbol=None, freq=merged_freq, left=node_1, right=node_2)
        heapq.heappush(queue, new_node)
    return queue[0]

def encoding_huffman(node, curr_code, codes_dict):
    if node is None:
        return
    
    if node.left is None and node.right is None:
        codes_dict[node.symbol] = curr_code
        return

    if node.left:
        encoding_huffman(node.left, curr_code + '0', codes_dict)

    if node.right:
        encoding_huffman(node.right, curr_code + '1', codes_dict)

def decode_huffman(encoded_data, huffman_root, original_dtype=None):
    decoded_signal = []
    node = huffman_root
    
    for bit in encoded_data:
        # 根据比特值选择左子树或右子树
        node = node.left if bit == '0' else node.right
        
        # 如果到达叶子节点，保存符号并重置节点到根节点
        if node.left is None and node.right is None:
            decoded_signal.append(node.symbol)
            node = huffman_root  # go to root

    if original_dtype is not None:
        return np.array(decoded_signal, dtype=original_dtype)
    else:
        return np.array(decoded_signal)
    
def encode_with_nearest_symbol(signal, codes_dict):
    encoded_data = ""
    known_symbols = list(codes_dict.keys())  # 所有已知符号的列表
    unknown_count = 0
    
    for symbol in signal:
        if symbol in codes_dict:
            encoded_data += codes_dict[symbol]
        else:
            # 找到数值上最接近的符号 - 使用INT32或INT64来计算距离
            min_distance = float('inf')
            nearest_symbol = None
            
            for known_symbol in known_symbols:
                # note that the nearest is related to the format of data
                distance = abs(int(known_symbol) - int(symbol))
                if distance < min_distance:
                    min_distance = distance
                    nearest_symbol = known_symbol
            
            encoded_data += codes_dict[nearest_symbol]
            unknown_count += 1

    
    if unknown_count > 0:
        print(f"{unknown_count} Unknown Symbols(Replaced Already)")
    
    return encoded_data


def plot_signal_comparison(original_signal, reconstructed_signal, title="Comparison of Original and Reconstructed Signals"):
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(original_signal, linewidth=1, color='orangered', label="Original Test Signal")
    plt.xlabel('Sample Index', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.title('Testing Signal Waveform', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplot(2, 1, 2)
    plt.plot(reconstructed_signal, linewidth=1, color='blue', label="Reconstructed Signal")
    plt.xlabel('Sample Index', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.title('Reconstructed Signal Waveform', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.show()

def calculate_entropy_pair(freq_dict, total_samples):
    probabilities = [freq / total_samples for freq in freq_dict.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    
    return entropy

def calculate_avg_code_length(freq_dict, codes_dict, total_samples):
    total_bits = sum(len(codes_dict[symbol]) * freq for symbol, freq in freq_dict.items() if symbol in codes_dict)
    avg_length = total_bits / total_samples
    
    return avg_length


