"""
train.py: Training utilities and loops for Transformer Language Model
TensorFlow implementation for mystery corpus training

Author: Eric Ewing
"""

import tensorflow as tf
import math
import os
import json
from typing import Tuple, Dict

# Import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss."""
    return math.exp(loss)

def train(model, train_dataset, test_dataset, epochs=5, learning_rate=1e-3,
          wandb_run=None, checkpoint_dir="checkpoints", continue_training=False):
    """
    Complete training function for language models.

    Args:
        model: Language model to train
        train_dataset: Training dataset
        test_dataset: Test dataset
        epochs: Number of epochs
        learning_rate: Learning rate
        wandb_run: Wandb run for logging
        tokenizer: Tokenizer for text generation
        checkpoint_dir: Directory to save checkpoints

    Returns:
        model: Trained model
    """
    print("*" * 60)
    print("STARTING TRAINING")
    print("*" * 60)

    # Ensure checkpoint directory exists otherwise create it
    os.makedirs(checkpoint_dir, exist_ok=True) 
    # TODO: Initialize your optimizer and loss function
    # TODO: Set up TensorFlow checkpointing with Checkpoint and CheckpointManager
    checkpoint = NotImplementedError
    checkpoint_manager = NotImplementedError
    
    # Handle checkpoint restoration for continue training
    start_epoch = 0
    if continue_training:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            # Extract epoch number from checkpoint name
            try:
                start_epoch = int(latest_checkpoint.split('-')[-1])
                print(f"Resuming from epoch {start_epoch}")
            except:
                print("Could not determine start epoch, starting from 0")
        else:
            print("No checkpoint found, starting fresh")
    
    # This is to keep track of model's performance during training
    history = {'train_loss': [], 'val_loss': [], 'perplexity': []}

    # TODO: Train your model, keep track of metrics and log to wandb
    for epoch in range(epochs):
        current_epoch = start_epoch + epoch + 1
        total_epochs = start_epoch + epochs
        print(f"\nEpoch {current_epoch}/{total_epochs}")
        print("-" * 40)

        # TODO: Iterate over the training dataset and update model weights

        # TODO: Iterate over the test dataset and compute validation loss

        # TODO: Calculate perplexity from validation loss
        
        # TODO: Append metrics to history dictionary

        # TODO : Save model checkpoint periodically or if validation loss improves
            
        # Log metrics to wandb if available
        if wandb_run:
            wandb_run.log({
                "epoch": None, # TODO: Current epoch number (one-index, so add 1)
                "train_loss": None,  # TODO: Calculate training loss
                "val_loss": None,  # TODO: Calculate validation loss
                "perplexity": None  # TODO: Calculate perplexity
            })
        

    return model