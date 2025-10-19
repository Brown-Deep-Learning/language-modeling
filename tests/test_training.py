"""
test_training.py - Tests for training loop TODOs

Tests basic training functionality including optimizer setup,
checkpointing, and training loop structure.

Run with:
    python -m pytest tests/test_training.py -v
"""

import pytest
import numpy as np
import tensorflow as tf
import tempfile
import shutil
from pathlib import Path
from src.training.train import train, calculate_perplexity


class TestTraining:
    """Tests for training loop"""

    @pytest.fixture
    def dummy_model(self):
        """Create a simple dummy model for testing"""
        # Create a simple model that outputs logits
        class DummyModel(tf.keras.Model):
            def __init__(self, vocab_size=50, d_model=32):
                super().__init__()
                self.vocab_size = vocab_size
                self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
                self.dense = tf.keras.layers.Dense(vocab_size)

            def call(self, inputs, training=None):
                x = self.embedding(inputs)
                x = tf.reduce_mean(x, axis=1, keepdims=True)
                x = tf.tile(x, [1, tf.shape(inputs)[1], 1])
                return self.dense(x)

        return DummyModel(vocab_size=50, d_model=32)

    @pytest.fixture
    def dummy_dataset(self):
        """Create dummy train and test datasets"""
        # Create simple datasets
        seq_length = 10
        batch_size = 4
        num_batches = 5
        vocab_size = 50

        def create_dataset():
            data = []
            for _ in range(num_batches):
                batch = tf.random.uniform([batch_size, seq_length], maxval=vocab_size, dtype=tf.int32)
                data.append(batch)
            return tf.data.Dataset.from_tensor_slices(data)

        train_ds = create_dataset()
        test_ds = create_dataset()

        return train_ds, test_ds

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoints"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_train_runs(self, dummy_model, dummy_dataset, temp_checkpoint_dir):
        """Test that training loop can execute without errors"""
        train_ds, test_ds = dummy_dataset

        try:
            trained_model = train(
                model=dummy_model,
                train_dataset=train_ds,
                test_dataset=test_ds,
                epochs=1,
                learning_rate=1e-3,
                wandb_run=None,
                checkpoint_dir=temp_checkpoint_dir,
                continue_training=False
            )
            assert trained_model is not None, "Training should return a model"
        except NotImplementedError:
            pytest.fail("Training loop raises NotImplementedError - not fully implemented")
        except Exception as e:
            # Allow skipping if implementation is incomplete
            if "NotImplementedError" in str(type(e).__name__):
                pytest.skip("Training loop not fully implemented")
            else:
                raise

    def test_train_returns_model(self, dummy_model, dummy_dataset, temp_checkpoint_dir):
        """Test that training returns the model"""
        train_ds, test_ds = dummy_dataset

        try:
            returned_model, _ = train(
                model=dummy_model,
                train_dataset=train_ds,
                test_dataset=test_ds,
                epochs=1,
                checkpoint_dir=temp_checkpoint_dir
            )
            assert returned_model is dummy_model, "Should return the same model instance"
        except (NotImplementedError, AttributeError):
            pytest.skip("Training loop not fully implemented")

    def test_training_history_structure(self, dummy_model, dummy_dataset, temp_checkpoint_dir):
        """Test that training history has correct structure"""
        train_ds, test_ds = dummy_dataset

        try:
            _, history = train(
                model=dummy_model,
                train_dataset=train_ds,
                test_dataset=test_ds,
                epochs=1,
                checkpoint_dir=temp_checkpoint_dir
            )
            assert isinstance(history, dict), "History should be a dictionary"
            for key in ['train_loss', 'val_loss', 'perplexity']:
                assert key in history, f"History should contain key '{key}'"
                assert isinstance(history[key], list), f"History['{key}'] should be a list"
        except (NotImplementedError, AttributeError):
            pytest.skip("Training loop not fully implemented. You should return the model AND history dictionary.")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
