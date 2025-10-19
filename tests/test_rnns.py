import pytest
import numpy as np
import tensorflow as tf
from src.models.RNNs import VanillaRNN, LSTM, create_rnn_language_model


class TestVanillaRNN:
    """Tests for VanillaRNN implementation"""

    @pytest.fixture
    def small_rnn(self):
        """Create a small RNN for testing"""
        vocab_size = 100
        hidden_size = 32
        seq_length = 10
        model = VanillaRNN(vocab_size, hidden_size, seq_length)
        return model

    def test_initialization(self, small_rnn):
        """Test that VanillaRNN can be instantiated"""
        assert small_rnn is not None, "RNN should instantiate successfully"
        assert hasattr(small_rnn, 'vocab_size'), "RNN should have vocab_size attribute"
        assert hasattr(small_rnn, 'hidden_size'), "RNN should have hidden_size attribute"

    def test_forward_pass(self, small_rnn):
        """Test that RNN can process a batch of sequences"""
        batch_size = 4
        seq_length = 10
        vocab_size = 100

        # Create random input
        inputs = tf.random.uniform([batch_size, seq_length], maxval=vocab_size, dtype=tf.int32)

        # Forward pass
        try:
            outputs = small_rnn(inputs, training=False)
            assert outputs is not None, "Forward pass should return output"
        except NotImplementedError:
            pytest.fail("Forward pass raises NotImplementedError - RNN not fully implemented")

    def test_output_shape(self, small_rnn):
        """Test that output has correct shape"""
        batch_size = 4
        seq_length = 10
        vocab_size = 100

        inputs = tf.random.uniform([batch_size, seq_length], maxval=vocab_size, dtype=tf.int32)

        try:
            outputs = small_rnn(inputs, training=False)

            # Output should be [batch_size, seq_length, vocab_size]
            expected_shape = (batch_size, seq_length, vocab_size)
            assert outputs.shape == expected_shape, \
                f"Output shape should be {expected_shape}, got {outputs.shape}"
        except NotImplementedError:
            pytest.skip("RNN not fully implemented")

    def test_output_dtype(self, small_rnn):
        """Test that output has correct dtype"""
        batch_size = 2
        seq_length = 5
        vocab_size = 100

        inputs = tf.random.uniform([batch_size, seq_length], maxval=vocab_size, dtype=tf.int32)

        try:
            outputs = small_rnn(inputs, training=False)
            assert outputs.dtype == tf.float32, "Output should be float32"
        except NotImplementedError:
            pytest.skip("RNN not fully implemented")

class TestLSTM:
    """Tests for LSTM implementation"""

    @pytest.fixture
    def small_lstm(self):
        """Create a small LSTM for testing"""
        vocab_size = 100
        hidden_size = 32
        seq_length = 10
        model = LSTM(vocab_size, hidden_size, seq_length)
        return model

    def test_initialization(self, small_lstm):
        """Test that LSTM can be instantiated"""
        assert small_lstm is not None, "LSTM should instantiate successfully"
        assert hasattr(small_lstm, 'vocab_size'), "LSTM should have vocab_size attribute"
        assert hasattr(small_lstm, 'hidden_size'), "LSTM should have hidden_size attribute"


    def test_forward_pass(self, small_lstm):
        """Test that LSTM can process a batch of sequences"""
        batch_size = 4
        seq_length = 10
        vocab_size = 100

        inputs = tf.random.uniform([batch_size, seq_length], maxval=vocab_size, dtype=tf.int32)

        try:
            outputs = small_lstm(inputs, training=False)
            assert outputs is not None, "Forward pass should return output"
        except (NotImplementedError, AttributeError):
            pytest.fail("Forward pass raises error - LSTM not fully implemented")

    def test_output_shape(self, small_lstm):
        """Test that output has correct shape"""
        batch_size = 4
        seq_length = 10
        vocab_size = 100

        inputs = tf.random.uniform([batch_size, seq_length], maxval=vocab_size, dtype=tf.int32)

        try:
            outputs = small_lstm(inputs, training=False)

            # Output should be [batch_size, seq_length, vocab_size]
            expected_shape = (batch_size, seq_length, vocab_size)
            assert outputs.shape == expected_shape, \
                f"Output shape should be {expected_shape}, got {outputs.shape}"
        except (NotImplementedError, AttributeError):
            pytest.skip("LSTM not fully implemented")

    def test_output_dtype(self, small_lstm):
        """Test that output has correct dtype"""
        batch_size = 2
        seq_length = 5
        vocab_size = 100

        inputs = tf.random.uniform([batch_size, seq_length], maxval=vocab_size, dtype=tf.int32)

        try:
            outputs = small_lstm(inputs, training=False)
            assert outputs.dtype == tf.float32, "Output should be float32"
        except (NotImplementedError, AttributeError):
            pytest.skip("LSTM not fully implemented")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
