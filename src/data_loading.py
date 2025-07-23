import numpy as np


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
    print(f"Unbalanced. Using {len(final_pairs)} total sequences.")

    # One-hot encode the sequences to finalize the pairs
    final_sequences = [one_hot_encode(seq) for (seq, val) in final_pairs]
    final_values = [val for (seq, val) in final_pairs]

    return np.array(final_sequences), np.array(final_values, dtype=np.float32)


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