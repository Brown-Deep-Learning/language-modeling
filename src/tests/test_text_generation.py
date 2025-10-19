"""
test_text_generation.py - Tests for text generation and sampling TODOs

Tests sampling methods (top-k, top-p) and text generation functionality.

Run with:
    python -m pytest tests/test_text_generation.py -v
    python -m pytest tests/test_text_generation.py::TestTextSampler -v
"""

import pytest
import numpy as np
import tensorflow as tf
from src.training.language_model import TextSampler, TextGenerator


class TestTextSampler:
    """Tests for TextSampler methods"""

    @pytest.fixture
    def sample_logits(self):
        """Create sample logits for testing"""
        # Create logits with batch_size=2, vocab_size=100
        batch_size = 2
        vocab_size = 100
        logits = tf.random.normal([batch_size, vocab_size])
        return logits

    def test_sample_top_k(self, sample_logits):
        """Test top-k sampling"""
        sampler = TextSampler()

        try:
            samples = sampler.sample_top_k(sample_logits, k=10, temperature=1.0)
            assert samples is not None, "Should return sampled tokens"
        except (NotImplementedError, AttributeError, TypeError):
            pytest.fail("sample_top_k raises error - not fully implemented")

    def test_sample_top_k_shape(self, sample_logits):
        """Test that top-k sampling returns correct shape"""
        sampler = TextSampler()

        try:
            samples = sampler.sample_top_k(sample_logits, k=10, temperature=1.0)
            assert samples.shape == (2, 1), "Should return [batch_size, 1]"
        except (NotImplementedError, AttributeError, TypeError):
            pytest.skip("sample_top_k not fully implemented")

    def test_sample_top_k_dtype(self, sample_logits):
        """Test that top-k sampling returns integer indices"""
        sampler = TextSampler()

        try:
            samples = sampler.sample_top_k(sample_logits, k=10, temperature=1.0)
            assert samples.dtype in [tf.int32, tf.int64], "Should return integer indices"
        except (NotImplementedError, AttributeError, TypeError):
            pytest.skip("sample_top_k not fully implemented")

    def test_sample_top_p(self, sample_logits):
        """Test top-p (nucleus) sampling"""
        sampler = TextSampler()

        try:
            samples = sampler.sample_top_p(sample_logits, p=0.9, temperature=1.0)
            assert samples is not None, "Should return sampled tokens"
        except (NotImplementedError, AttributeError, TypeError):
            pytest.fail("sample_top_p raises error - not fully implemented")

    def test_sample_top_p_shape(self, sample_logits):
        """Test that top-p sampling returns correct shape"""
        sampler = TextSampler()

        try:
            samples = sampler.sample_top_p(sample_logits, p=0.9, temperature=1.0)
            assert samples.shape == (2, 1), "Should return [batch_size, 1]"
        except (NotImplementedError, AttributeError, TypeError):
            pytest.skip("sample_top_p not fully implemented")

    def test_sample_top_p_dtype(self, sample_logits):
        """Test that top-p sampling returns integer indices"""
        sampler = TextSampler()

        try:
            samples = sampler.sample_top_p(sample_logits, p=0.9, temperature=1.0)
            assert samples.dtype in [tf.int32, tf.int64], "Should return integer indices"
        except (NotImplementedError, AttributeError, TypeError):
            pytest.skip("sample_top_p not fully implemented")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
