#!/usr/bin/env python3
"""
Main training script for Transformer Language Model on Mystery Corpus
Uses preprocessed pickle file with tokenized data
"""
import sys
import os

# Force output to be unbuffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import tensorflow as tf
import numpy as np
import os
import json
import argparse
from datetime import datetime

# Enable eager execution to avoid graph mode issues with RNN implementations
tf.config.run_functions_eagerly(True)

# Import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Wandb not available - install with: pip install wandb")
    WANDB_AVAILABLE = False

# Import your modules
from src.training.language_model import TextGenerator, TextSampler
from src.training.train import train
from src.data.data import prepare_data
from src.models.RNNs import create_rnn_language_model
from src.models.transformer import create_language_model


# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
tf.keras.backend.clear_session()
tf.keras.backend.set_floatx('float32')
tf.keras.mixed_precision.set_global_policy('float32')

class Config:
    """Configuration class for training parameters."""

    def __init__(self):
        """Initialize configuration with default values."""
        # Data parameters
        self.data_path = 'data/mystery_data.pkl'
        self.seq_length = 256
        self.batch_size = 64

        # Model parameters
        self.model_type = "transformer"  # Options: "transformer", "vanilla_rnn", "lstm"
        self.vocab_size = 10000  # Can be reduced for faster training (e.g., 2000-5000)
        self.d_model = 512
        self.n_heads = 8
        self.n_layers = 6
        self.d_ff = 2048
        self.dropout_rate = 0.1

        # Training parameters
        self.epochs = 10
        self.learning_rate = 1e-3
        self.use_lr_schedule = False

        # Paths
        self.checkpoint_dir = 'checkpoints'
        self.logs_dir = 'logs'
        self.model_save_path = 'final_model'

        # Generation parameters
        self.generation_length = 100
        self.generation_temperature = 0.8
        self.generation_top_k = 40
        self.generation_top_p = 0.9

        # Wandb configuration
        self.use_wandb = False
        self.wandb_project = "mystery-transformer"
        self.wandb_entity = None
        self.wandb_run_name = None

def count_parameters(model):
    """Count the total number of trainable parameters in the model."""
    return sum([tf.size(w).numpy() for w in model.trainable_weights])

def setup_directories(config: Config):
    """Create necessary directories."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)

def save_config(config: Config, filepath: str):
    """Save configuration to JSON file."""
    config_dict = {}
    for k, v in config.__dict__.items():
        if not k.startswith('_'):
            # Convert numpy types to Python types for JSON serialization
            if hasattr(v, 'item'):
                config_dict[k] = v.item()
            elif isinstance(v, (np.integer, np.floating)):
                config_dict[k] = v.item()
            else:
                config_dict[k] = v
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_config_from_file(filepath: str, config: Config):
    """Load configuration from JSON file and update config object."""
    try:
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Update config with loaded values
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
        
        return True
    except Exception as e:
        print(f"Could not load config from {filepath}: {e}")
        return False

def find_latest_config(model_type: str):
    """Find the most recent config file for a given model type."""
    config_pattern = f"config_*_{model_type}.json"
    logs_dir = 'logs'
    
    if not os.path.exists(logs_dir):
        return None
    
    # Look for config files
    config_files = []
    for file in os.listdir(logs_dir):
        if file.startswith('config_') and file.endswith('.json'):
            config_files.append(os.path.join(logs_dir, file))
    
    if not config_files:
        return None
    
    # Return the most recent one
    config_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return config_files[0]

def load_mystery_data(config: Config):
    """Load the mystery corpus data from preprocessed pickle file."""
    print("LOADING MYSTERY CORPUS DATA")

    # Load preprocessed data - no tokenization needed!
    train_dataset, test_dataset, tokenizer = prepare_data(
        pickle_path=config.data_path,
        seq_length=config.seq_length,
        batch_size=config.batch_size,
        vocab_size=config.vocab_size  # Use vocab_size from config
    )

    # Update config with actual vocab size from the pickle file
    config.vocab_size = len(tokenizer)
    print(f"Vocabulary size from pickle: {config.vocab_size:,}")

    return train_dataset, test_dataset, tokenizer

# Remove setup_gpu function as it's now integrated into main()

def create_model(config: Config):
    """Create and initialize the language model."""
    print(f"CREATING {config.model_type.upper()} MODEL")

    if config.model_type == "transformer":
        model = create_language_model(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_length=config.seq_length,
            dropout_rate=config.dropout_rate
        )
    elif config.model_type in ["vanilla_rnn", "lstm"]:
        rnn_type = "vanilla" if config.model_type == "vanilla_rnn" else "lstm"
        model = create_rnn_language_model(
            vocab_size=config.vocab_size,
            hidden_size=config.d_model,
            seq_length=config.seq_length,
            model_type=rnn_type
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}. Use 'transformer', 'vanilla_rnn', or 'lstm'")

    # Build model efficiently with a dummy input to create weights
    print("Building model...")
    dummy_input = tf.zeros((1, config.seq_length), dtype=tf.int32)
    output = model(dummy_input)

    # Verify dtype consistency
    print(f"Model built - Input: {dummy_input.dtype}, Output: {output.dtype}")
    assert output.dtype == tf.float32, f"Expected float32 output, got {output.dtype}"

    # Count parameters efficiently
    total_params = count_parameters(model)
    print(f"Model created with {total_params:,} trainable parameters")

    return model

def train_model(model, train_dataset, test_dataset, config: Config, wandb_run, tokenizer, continue_training=False):
    """Train the model using training pipeline."""
    print("TRAINING")

    # Train the model using simplified training function
    model, history = train(
        model,
        train_dataset,
        test_dataset,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        wandb_run=wandb_run,
        checkpoint_dir=config.checkpoint_dir,
        continue_training=continue_training
    )

    # Create dummy trainer for compatibility
    trainer = None
    return history, trainer

def generate_sample_text(model, tokenizer, config: Config):
    """Generate and display sample text."""
    # Create TextGenerator
    generator = TextGenerator(model, tokenizer)
    
    # Mystery-themed prompts for detective stories
    sample_prompts = [
        "The detective examined",
        "In the dimly lit room",
        "The murder weapon was",
        "Holmes deduced that",
        ""  # Empty prompt for unconditional generation
    ]
    
    print("\n" + "=" * 60)
    print("SAMPLE TEXT GENERATION")
    print("=" * 60)
    
    for i, prompt in enumerate(sample_prompts):
        print(f"\nPrompt {i+1}: '{prompt}'" if prompt else f"\nPrompt {i+1}: [Random generation]")
        print("-" * 60)
        
        try:
            generated_text = generator.generate(
                prompt=prompt,
                max_length=config.generation_length,
                method="top_k",
                temperature=config.generation_temperature,
                top_k=config.generation_top_k,
                top_p=config.generation_top_p
            )
            print(generated_text)
        except Exception as e:
            print(f"Error generating text: {e}")
        
        print()
    
    print("=" * 60)

def interactive_generation(model, tokenizer, config: Config):
    """Interactive text generation session using new TextGenerator."""
    print("=" * 60)
    print("INTERACTIVE MYSTERY TEXT GENERATION")
    print("=" * 60)
    print("Commands:")
    print("  - Enter any text to use as prompt")
    print("  - 'random' for random generation (no prompt)")
    print("  - 'samples' for pre-defined mystery prompts")
    print("  - 'settings' to view current generation settings")
    print("  - 'quit' to exit")
    print()

    # Create TextGenerator
    generator = TextGenerator(model, tokenizer)
    
    # Set up mystery-themed prompts
    mystery_prompts = [
        "The detective examined",
        "In the dimly lit room",
        "The murder weapon was",
        "Holmes deduced that",
        "The evidence suggested",
        "At the crime scene",
        "The suspect claimed",
        "The mysterious letter read"
    ]
    
    # Generation settings
    gen_settings = {
        'max_length': config.generation_length,
        'temperature': config.generation_temperature,
        'top_k': config.generation_top_k,
        'top_p': config.generation_top_p,
        'method': 'top_k'
    }
    
    print(f"Current settings: {gen_settings}\n")
    
    # Handle user commands in interactive loop
    while True:
        try:
            user_input = input("Enter prompt (or command): ").strip()
            
            if not user_input:
                continue
            
            # Handle quit command
            if user_input.lower() == 'quit':
                print("Exiting interactive generation...")
                break
            
            # Handle settings command
            elif user_input.lower() == 'settings':
                print(f"\nCurrent settings: {gen_settings}\n")
                continue
            
            # Handle samples command - show predefined prompts
            elif user_input.lower() == 'samples':
                print("\n--- Pre-defined Mystery Prompts ---")
                for i, prompt in enumerate(mystery_prompts, 1):
                    print(f"{i}. \"{prompt}\"")
                print("\nGenerating samples...\n")
                
                for prompt in mystery_prompts[:3]:  # Generate first 3 samples
                    print(f"Prompt: '{prompt}'")
                    print("-" * 60)
                    try:
                        generated_text = generator.generate(prompt=prompt, **gen_settings)
                        print(generated_text)
                    except Exception as e:
                        print(f"Error: {e}")
                    print()
                continue
            
            # Handle random generation
            elif user_input.lower() == 'random':
                print("\n--- Random Generation (no prompt) ---")
                print("-" * 60)
                try:
                    generated_text = generator.generate(prompt="", **gen_settings)
                    print(generated_text)
                except Exception as e:
                    print(f"Error: {e}")
                print()
                continue
            
            # Generate text from user's custom prompt
            else:
                print(f"\nPrompt: '{user_input}'")
                print("-" * 60)
                try:
                    generated_text = generator.generate(prompt=user_input, **gen_settings)
                    print(generated_text)
                except Exception as e:
                    print(f"Error: {e}")
                print()
        
        except KeyboardInterrupt:
            print("\n\nExiting interactive generation...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

def main():
    """Main training function."""
    print("Starting...")
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Allow memory growth to prevent TensorFlow from allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU memory growth setup failed: {e}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Transformer Language Model')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--seq-length', type=int, default=None, help='Sequence length')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--simple', action='store_true', help='Use simple training mode')
    parser.add_argument('--interactive', action='store_true', help='Run interactive generation')
    parser.add_argument('--generate-only', action='store_true', help='Skip training, only generate')
    parser.add_argument('--continue-training', action='store_true', help='Continue from checkpoint')
    parser.add_argument('--force-fresh', action='store_true', help='Force fresh training')
    parser.add_argument('--model-type', type=str, choices=['transformer', 'vanilla_rnn', 'lstm'],
                       help='Type of model to train (transformer, vanilla_rnn, lstm)')
    parser.add_argument('--vocab-size', type=int, default=None,
                       help='Vocabulary size (default: 10000, try 2000-5000 for faster training)')
    parser.add_argument('--d-model', type=int, default=None,
                       help='Model dimension/hidden size (default: 512, try 128-256 for faster training)')

    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Override config with command line arguments
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.seq_length:
        config.seq_length = args.seq_length
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.model_type:
        config.model_type = args.model_type
    if args.vocab_size:
        config.vocab_size = args.vocab_size
    if args.d_model:
        config.d_model = args.d_model

    # Setup directories
    setup_directories(config)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("MYSTERY CORPUS TRAINING", flush=True)
    print(f"Timestamp: {timestamp}", flush=True)
    print(f"Data file: {config.data_path}", flush=True)

    # Check if pickle file exists
    if not os.path.exists(config.data_path):
        print(f"Error: {config.data_path} not found!")
        print("Please run the data preprocessing first:")
        print("  cd data")
        print("  python simple_process.py")
        return

    start_time = datetime.now()
    print("Starting data loading...", flush=True)

    if not args.generate_only:
        # Load preprocessed data
        train_dataset, test_dataset, tokenizer = load_mystery_data(config)

        # Create model
        model = create_model(config)

        # Save configuration
        config_path = os.path.join(config.logs_dir, f'config_{timestamp}.json')
        save_config(config, config_path)
        print(f"Configuration saved to: {config_path}")

        # Simple training mode
        print("Using simple training mode...")
        continue_training = False  # Set to True if you want to resume from checkpoint
        model, history = train(
            model, train_dataset, test_dataset,
            epochs=config.epochs, learning_rate=config.learning_rate,
            wandb_run=None, checkpoint_dir=config.checkpoint_dir,
            continue_training=continue_training
        )

        # Save model to both generic and model-specific paths
        generic_path = config.model_save_path + '.weights.h5'
        specific_path = f"{config.model_type}_model.weights.h5"
        
        model.save_weights(generic_path)
        model.save_weights(specific_path)
        
        print(f"Model saved to: {generic_path}")
        print(f"Model also saved to: {specific_path} (for submission)")
        
        # Save a model-specific config file for easy loading later
        model_config_path = f"{config.model_type}_config.json"
        save_config(config, model_config_path)
        print(f"Model config saved to: {model_config_path}")

        # Calculate total training time
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        print(f"Training completed in {total_time/60:.1f} minutes")

    else:
        # Generate-only mode
        print("Generate-only mode: Loading existing model...")
        
        # Try to load saved config if parameters not specified
        model_config_path = f"{config.model_type}_config.json"
        if os.path.exists(model_config_path):
            print(f"Found saved config: {model_config_path}")
            
            # Check which parameters were explicitly provided
            params_specified = {
                'vocab_size': args.vocab_size is not None,
                'd_model': args.d_model is not None,
                'seq_length': args.seq_length is not None
            }
            
            # Load config
            if load_config_from_file(model_config_path, config):
                print(f"Loaded model configuration:")
                print(f"  vocab_size: {config.vocab_size}")
                print(f"  d_model: {config.d_model}")
                print(f"  seq_length: {config.seq_length}")
                print(f"  model_type: {config.model_type}")
                
                # Override with explicitly provided parameters
                if params_specified['vocab_size']:
                    config.vocab_size = args.vocab_size
                    print(f"  (vocab_size overridden to {args.vocab_size})")
                if params_specified['d_model']:
                    config.d_model = args.d_model
                    print(f"  (d_model overridden to {args.d_model})")
                if params_specified['seq_length']:
                    config.seq_length = args.seq_length
                    print(f"  (seq_length overridden to {args.seq_length})")
        else:
            print(f"No saved config found at {model_config_path}")
            print("Using parameters from command line or defaults")
        
        # Load data (just for tokenizer)
        _, _, tokenizer = load_mystery_data(config)
        
        # Create and load model
        model = create_model(config)
        
        # Try to load model weights - first try model-specific name, then generic
        model_specific_path = f"{config.model_type}_model.weights.h5"
        generic_path = config.model_save_path + '.weights.h5'
        
        loaded = False
        load_errors = []
        for path in [model_specific_path, generic_path]:
            try:
                if os.path.exists(path):
                    model.load_weights(path)
                    print(f"Model loaded from: {path}")
                    loaded = True
                    break
            except Exception as e:
                load_errors.append(f"{path}: {str(e)}")
                continue
        
        if not loaded:
            print(f"Could not load {config.model_type} model weights!")
            print(f"Tried: {model_specific_path}, {generic_path}")
            if load_errors:
                print("\nErrors encountered:")
                for error in load_errors:
                    print(f"  - {error}")
            print("\nMake sure you:")
            print("  1. Have trained a model first")
            print("  2. Use the same --vocab-size, --d-model, and --seq-length as during training")
            return

    # Generate sample text
    print("Generating sample text...")
    generate_sample_text(model, tokenizer, config)

    # Interactive generation
    if args.interactive:
        interactive_generation(model, tokenizer, config)

    print("Mystery corpus training complete!")

if __name__ == "__main__":
    main()