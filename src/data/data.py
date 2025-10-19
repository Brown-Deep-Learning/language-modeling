"""
data.py - Data loading for Transformer Language Model
Simple loader for preprocessed mystery corpus pickle file
"""

import tensorflow as tf
import numpy as np
import pickle
from typing import List, Tuple, Dict
import re

class TextTokenizer:
    """
    Tokenizer wrapper for preprocessed vocabulary.
    """

    def __init__(self, vocab: List[str], word_to_idx: Dict[str, int]):
        """
        Initialize tokenizer with preprocessed vocabulary.

        Args:
            vocab: List of vocabulary tokens (i.e., words)
            word_to_idx: Dictionary mapping tokens to indices
        """
        self.vocab = vocab
        self.token_to_idx = word_to_idx
        self.idx_to_token = {idx: token for token, idx in word_to_idx.items()}

        # Special tokens (you'll get used to these the more you see language assignments)
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.eos_token = '<EOS>'

    def decode(self, indices: List[int]) -> str:
        """
        Convert token indices back to text.

        Args:
            indices: List of token indices

        Returns:
            Decoded text string
        """
        # TODO: Convert indices to tokens using self.idx_to_token
        # TODO: Handle special tokens appropriately
        tokens = []
        
        # Join with spaces and fix punctuation spacing (provided for convenience and these are just for display)
        text = ' '.join(tokens)
        text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
        text = text.replace(' :', ':').replace(' ;', ';').replace(' "', '"')

        return text

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token indices (basic implementation).

        Args:
            text: Input text to encode

        Returns:
            List of token indices
        """
        # Tokenize text (split on whitespace and punctuation; provided for convenience)
        # this returns a list of tokens you can iterate over (hint hint ;)
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        
        # TODO: Convert tokens to indices using self.token_to_idx
        # TODO: Handle out-of-vocabulary words with unknown token
        indices = []
        
        return indices

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def get_pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.token_to_idx.get(self.pad_token, 0)

    def get_unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self.token_to_idx.get(self.unk_token, 1)

    def get_eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return self.token_to_idx.get(self.eos_token, 2)

    def __len__(self) -> int:
        return len(self.vocab)

def limit_vocabulary(train_tokens: List[int], test_tokens: List[int],
                    tokenizer: TextTokenizer, vocab_size: int) -> Tuple[List[int], List[int], TextTokenizer]:
    """
    Limit vocabulary size by keeping only the most frequent tokens.
    Remap all tokens to the reduced vocabulary.

    Args:
        train_tokens: Training token sequence
        test_tokens: Test token sequence
        tokenizer: Original tokenizer
        vocab_size: Target vocabulary size

    Returns:
        Tuple of (remapped_train_tokens, remapped_test_tokens, new_tokenizer)
    """
    # Keep the most frequent vocab_size tokens (they should already be sorted by frequency)
    # Ensure we keep special tokens
    special_tokens = ['<PAD>', '<UNK>', '<EOS>']

    # Keep top vocab_size tokens
    new_vocab = tokenizer.vocab[:vocab_size]

    # Make sure special tokens are included
    for special_token in special_tokens:
        if special_token not in new_vocab and special_token in tokenizer.vocab:
            # Replace least frequent non-special token
            for i in range(len(new_vocab)-1, -1, -1):
                if new_vocab[i] not in special_tokens:
                    new_vocab[i] = special_token
                    break

    # Create new word_to_idx mapping
    new_word_to_idx = {token: idx for idx, token in enumerate(new_vocab)}
    unk_idx = new_word_to_idx.get('<UNK>', 1)  # Default to index 1 if UNK not found

    # Remap tokens any token not in new vocab becomes UNK
    def remap_tokens(tokens):
        remapped = []
        for token_idx in tokens:
            original_token = tokenizer.idx_to_token.get(token_idx, '<UNK>')
            if original_token in new_word_to_idx:
                remapped.append(new_word_to_idx[original_token])
            else:
                remapped.append(unk_idx)
        return remapped

    new_train_tokens = remap_tokens(train_tokens)
    new_test_tokens = remap_tokens(test_tokens)

    # Create new tokenizer
    new_tokenizer = TextTokenizer(new_vocab, new_word_to_idx)

    print(f"  Vocabulary reduced from {len(tokenizer.vocab)} to {len(new_vocab)} tokens")
    print(f"  Special tokens: {[t for t in special_tokens if t in new_vocab]}")

    return new_train_tokens, new_test_tokens, new_tokenizer

def load_mystery_data(pickle_path: str = 'mystery_data.pkl') -> Tuple[List[int], List[int], TextTokenizer]:
    """
    Load preprocessed mystery corpus data from pickle file.

    Args:
        pickle_path: Path to the mystery_data.pkl file

    Returns:
        Tuple of (train_tokens, test_tokens, tokenizer)
    """
    print(f"Loading mystery data from {pickle_path}...")

    with open(pickle_path, 'rb') as f:
        data_dict = pickle.load(f)

    train_data = data_dict['train_data']
    test_data = data_dict['test_data']
    vocab = data_dict['vocab']
    word_to_idx = data_dict['word_to_idx']

    # Create tokenizer
    tokenizer = TextTokenizer(vocab, word_to_idx)

    return train_data, test_data, tokenizer

def create_sequences(tokens: List[int], seq_length: int) -> np.ndarray:
    """
    Create 1-token overlapping input sequences for language modeling.
    Targets will be computed during training by shifting the inputs.

    Args:
        tokens: List of token indices
        seq_length: Sequence length for training

    Returns:
        Input sequences array
    """
    sequences = []
    # TODO: Create 1-token overlapping sequences from token list
    
    return np.array(sequences, dtype=np.int32)

def create_tf_datasets(train_tokens: List[int], test_tokens: List[int],
                      seq_length: int = 256, batch_size: int = 16) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create TensorFlow datasets from token sequences.

    Args:
        train_tokens: Training token sequence
        test_tokens: Test token sequence
        seq_length: Sequence length for training
        batch_size: Batch size

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # TODO: Create sequences using create_sequences
    # TODO: Convert to TensorFlow constants (tf.constant will be your friend here)
    # TODO: Create datasets with batching and shuffling (look up the documentation for tf.data.Dataset)
    pass

def prepare_data(pickle_path: str = 'mystery_data.pkl', seq_length: int = 256,
                batch_size: int = 16, vocab_size: int = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, TextTokenizer]:
    """
    Complete data preparation pipeline for mystery corpus. You do not need to know the details of this function,
    but you should understand its inputs and outputs.

    Args:
        pickle_path: Path to mystery_data.pkl file
        seq_length: Sequence length for training
        batch_size: Batch size for datasets
        vocab_size: Maximum vocabulary size (None = use full vocab, smaller values = faster training)

    Returns:
        Tuple of (train_dataset, test_dataset, tokenizer)
    """
    print("=" * 60)
    print("PREPARING MYSTERY CORPUS DATA FOR BRUNO")
    print("=" * 60)

    # Load preprocessed data
    train_tokens, test_tokens, tokenizer = load_mystery_data(pickle_path)

    # Optionally limit vocabulary size for faster training
    if vocab_size is not None and vocab_size < len(tokenizer.vocab):
        print(f"Limiting vocabulary from {len(tokenizer.vocab)} to {vocab_size} tokens...")
        train_tokens, test_tokens, tokenizer = limit_vocabulary(
            train_tokens, test_tokens, tokenizer, vocab_size
        )

    # Create TensorFlow datasets
    train_dataset, test_dataset = create_tf_datasets(
        train_tokens, test_tokens, seq_length, batch_size
    )

    print("Data preparation complete!")
    print("=" * 60)

    return train_dataset, test_dataset, tokenizer

if __name__ == "__main__":
    """Test the data loading pipeline."""

    print("Testing mystery data loading...")
    tf.random.set_seed(42)
    np.random.seed(42)

    try:
        # Test loading the mystery data
        train_dataset, test_dataset, tokenizer = prepare_data(
            pickle_path='data/mystery_data.pkl',
            seq_length=128,  # Shorter sequences for testing
            batch_size=8
        )

        print(f"\nTokenizer info:")
        print(f"  Vocabulary size: {len(tokenizer)}")
        print(f"\nFirst 20 vocabulary tokens:")
        for i in range(min(20, len(tokenizer.vocab))):
            print(f"  {i}: '{tokenizer.vocab[i]}'")

        print("\nData Attributes (Test and Train):")
        for batch in train_dataset.take(1):
            print(f"  Train batch shape: {batch.shape}")

        for batch in test_dataset.take(1):
            print(f"  Test batch shape: {batch.shape}")

        print("\nTesting tokenizer encode/decode:")
        test_text = "Holmes examined the mysterious evidence carefully."
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        print(f"  Original: '{test_text}'")
        print(f"  Encoded: {encoded}")
        print(f"  Decoded: '{decoded}'")

        print("\nData loading test successful!")

    except FileNotFoundError:
        print("ERROR: mystery_data.pkl not found!")
        print("Make sure you have run your preprocessing script and the pickle file exists.")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()