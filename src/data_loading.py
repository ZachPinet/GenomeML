import numpy as np


# Get batch ID from filename (0=BroadHistone, 1=OpenChromChip, 2=UwTfbs)
def get_batch_id(filename):
    name = filename.lower()
    if 'broadhistone' in name:
        return 0
    elif 'openchromchip' in name:
        return 1
    elif 'uwtfbs' in name:
        return 2
    else:
        return -1  # Unknown batch


# This one-hot encodes any sequence and returns it.
def one_hot_encode(sequence):
    mapping = {
        'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'T': [0, 0, 1, 0], 
        'G': [0, 0, 0, 1], 'N': [0, 0, 0, 0]
    }
    return np.array([mapping[nuc] for nuc in sequence], dtype=np.float32)


# This one-hot-decodes any sequence back into letters and returns it.
def one_hot_decode(encoded_sequence):
    reverse_mapping = {
        (1, 0, 0, 0): 'A', (0, 1, 0, 0): 'C', (0, 0, 1, 0): 'T', 
        (0, 0, 0, 1): 'G', (0, 0, 0, 0): 'N'
    }
    decoded_sequence = ''.join(
        reverse_mapping.get(tuple(vec), 'N') for vec in encoded_sequence
    )
    return decoded_sequence


# This gets the relevant data and balances the ratio of values.
def load_data(fasta_file, values_file, max_seqs):
    sequences = []
    
    # Read sequences from FASTA file and join each to one line
    with open(fasta_file, 'r') as f:
        lines = f.read().splitlines()
        # FASTA has 1 useless header line followed by 7 sequence lines
        for i in range(0, len(lines), 8):
            full_sequence = ''.join(lines[i+1:i+8])
            sequences.append(full_sequence)

    # Read numerical values from a column
    values = np.loadtxt(values_file, dtype=np.float32)
    assert len(sequences) == len(values), "Mismatch between seq and value len."

    # Shuffle pairs and truncate to max_seqs
    all_pairs = list(zip(sequences, values))
    np.random.shuffle(all_pairs)
    final_pairs = all_pairs[:max_seqs]

    # One-hot encode the sequences to finalize the pairs
    final_sequences = [one_hot_encode(seq) for (seq, val) in final_pairs]
    final_seq_array = np.array(final_sequences)
    final_values = [val for (seq, val) in final_pairs]
    final_val_array = np.array(final_values, dtype=np.float32)

    # Extract batch ID from filename
    batch_id = get_batch_id(str(values_file))
    batch_ids = np.full(len(final_sequences), batch_id, dtype=np.int32)
    
    return final_seq_array, final_val_array, batch_ids


# This loads every value file. Can use all of them or just one.
def load_all_columns(columns_dir, reference_file, max_seqs):    
    # Load the reference values to get the shuffle order
    reference_values = np.loadtxt(reference_file, dtype=np.float32)
    
    # Load all column values
    all_values = []
    for file in sorted(columns_dir.glob("*.txt")):
        values = np.loadtxt(file, dtype=np.float32)
        all_values.append(values)
    
    y_all = np.stack(all_values, axis=1)  # Shape: (n_samples, n_cols)
    
    # Shuffle pairs and truncate to max_seqs
    all_indices = np.arange(len(reference_values))
    np.random.shuffle(all_indices)
    final_indices = all_indices[:max_seqs]
    
    return y_all[final_indices]