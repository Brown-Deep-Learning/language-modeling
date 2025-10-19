import pytest
import numpy as np
import tensorflow as tf
from src.data.data import TextTokenizer, create_sequences, create_tf_datasets


class TestTextTokenizer:
    """Tests for TextTokenizer encode and decode methods"""

    @pytest.fixture
    def simple_tokenizer(self):
        """Create a simple tokenizer for testing"""
        vocab = ['<PAD>', '<UNK>', '<EOS>', 'the', 'cat', 'sat', 'on', 'mat', '.']
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        return TextTokenizer(vocab, word_to_idx)

    def test_encode_basic(self, simple_tokenizer):
        """Test that encode converts text to indices"""
        text = "the cat sat"
        indices = simple_tokenizer.encode(text)

        # Check it returns a list
        assert isinstance(indices, list), "encode should return a list"
        # Check it returns non-empty result
        assert len(indices) > 0, "encode should return non-empty list for non-empty text"
        # Check all elements are integers
        assert all(isinstance(i, (int, np.integer)) for i in indices), "All indices should be integers"

    def test_decode_basic(self, simple_tokenizer):
        """Test that decode converts indices back to text"""
        indices = [4, 5, 6]  # cat sat on
        text = simple_tokenizer.decode(indices)

        # Check it returns a string
        assert isinstance(text, str), "decode should return a string"
        # Check it's not empty
        assert len(text) > 0, "decode should return non-empty string for non-empty indices"

    def test_encode_decode_roundtrip(self, simple_tokenizer):
        """Test that encode and decode are roughly inverse operations"""
        original_text = "the cat sat on the mat"

        # Encode then decode
        indices = simple_tokenizer.encode(original_text)
        decoded_text = simple_tokenizer.decode(indices)

        # The words should be preserved (though spacing/punctuation might differ)
        original_words = set(original_text.lower().split())
        decoded_words = set(decoded_text.lower().split())

        # All original words should appear in decoded text
        assert original_words.issubset(decoded_words) or len(original_words.intersection(decoded_words)) >= len(original_words) * 0.8, \
            "Decoded text should preserve most/all original words"


class TestCreateSequences:
    """Tests for create_sequences function"""

    def test_creates_sequences(self):
        """Test that create_sequences returns a numpy array"""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        seq_length = 3

        sequences = create_sequences(tokens, seq_length)

        # Check it returns a numpy array
        assert isinstance(sequences, np.ndarray), "create_sequences should return numpy array"
        # Check dtype is int32
        assert sequences.dtype == np.int32, "Sequences should be int32 dtype"

    def test_sequence_shape(self):
        """Test that sequences have correct shape"""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        seq_length = 3

        sequences = create_sequences(tokens, seq_length)

        # Should have shape [num_sequences, seq_length + 1]
        # (seq_length + 1 to include both input and target)
        assert len(sequences.shape) == 2, "Sequences should be 2D array"
        assert sequences.shape[1] == seq_length + 1, f"Each sequence should have length {seq_length + 1} (input + target)"


class TestCreateTFDatasets:
    """Tests for create_tf_datasets function"""

    def test_returns_datasets(self):
        """Test that create_tf_datasets returns TensorFlow datasets"""
        # Create sample token lists
        train_tokens = list(range(100))
        test_tokens = list(range(50))

        train_ds, test_ds = create_tf_datasets(train_tokens, test_tokens, seq_length=10, batch_size=4)

        # Check that both are TensorFlow datasets
        assert isinstance(train_ds, tf.data.Dataset), "Should return tf.data.Dataset for train"
        assert isinstance(test_ds, tf.data.Dataset), "Should return tf.data.Dataset for test"

    def test_dataset_batching(self):
        """Test that datasets are properly batched"""
        train_tokens = list(range(200))
        test_tokens = list(range(100))
        batch_size = 8
        seq_length = 10

        train_ds, test_ds = create_tf_datasets(train_tokens, test_tokens,
                                               seq_length=seq_length, batch_size=batch_size)

        # Get a batch and check its shape
        for batch in train_ds.take(1):
            # Batch should have shape [batch_size, seq_length + 1] or smaller for last batch
            assert len(batch.shape) == 2, "Batch should be 2D"
            assert batch.shape[0] <= batch_size, f"Batch size should be <= {batch_size}"
            assert batch.shape[1] == seq_length + 1, f"Sequence length should be {seq_length + 1} (input + target)"
            break

    def test_dataset_elements(self):
        """Test that dataset elements are valid sequences"""
        train_tokens = list(range(100))
        test_tokens = list(range(50))

        train_ds, test_ds = create_tf_datasets(train_tokens, test_tokens, seq_length=10, batch_size=4)

        # Check that we can iterate and get valid data
        for batch in train_ds.take(1):
            # Check dtype
            assert batch.dtype in [tf.int32, tf.int64], "Batch should contain integers"
            # Check that values are in valid range
            assert tf.reduce_min(batch) >= 0, "Token indices should be non-negative"
            assert tf.reduce_max(batch) < 100, "Token indices should be within range"
            break


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
