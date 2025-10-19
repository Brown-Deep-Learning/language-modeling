import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="transformer")
class AttentionMatrix(keras.layers.Layer):
    """Compute attention matrix"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        Compute attention weights from K and Q matrices.

        Args:
            inputs: [K, Q] where K and Q are [batch_size, seq_length, embed_size]

        Returns:
            attention_weights: [batch_size, seq_length, seq_length]
        """
        K, Q = inputs

        # 1. Ensure consistent dtypes (cast to tf.float32)
        K = tf.cast(K, tf.float32)
        Q = tf.cast(Q, tf.float32)
        head_size = tf.cast(tf.shape(K)[-1], tf.float32)

        # TODO: Compute scaled dot-product attention scores and normalize
        # TODO: Apply softmax to get attention weights
        
        return NotImplementedError

    def get_config(self):
        config = super().get_config()
        return config

@keras.saving.register_keras_serializable(package="transformer")
class AttentionHead(keras.layers.Layer):
    """Single attention head"""

    def __init__(self, input_size, output_size, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size

        # TODO: Initialize linear projections for K, Q, V
        # TODO: Initialize attention matrix computation
        
        self.attention_matrix = AttentionMatrix()

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        Apply single attention head.

        Args:
            inputs_for_keys: [batch_size, seq_length, input_size]
            inputs_for_values: [batch_size, seq_length, input_size]
            inputs_for_queries: [batch_size, seq_length, input_size]

        Returns:
            output: [batch_size, seq_length, output_size]
        """
        # 1. Ensure consistent dtypes
        inputs_for_keys = tf.cast(inputs_for_keys, tf.float32)
        inputs_for_values = tf.cast(inputs_for_values, tf.float32)
        inputs_for_queries = tf.cast(inputs_for_queries, tf.float32)

        # TODO: Apply linear transformations to get K, Q, V
        # TODO: Compute attention weights
        # TODO: Apply attention to values

        return NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "output_size": self.output_size
        })
        return config

@keras.saving.register_keras_serializable(package="transformer")
class MultiHeadAttention(keras.layers.Layer):
    """Multi-head attention mechanism"""

    def __init__(self, embed_size, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads

        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"

        # TODO: Create attention heads
        # TODO: Initialize output projection (embed_size)

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        Apply multi-head attention.

        Args:
            inputs_for_keys: [batch_size, seq_length, embed_size]
            inputs_for_values: [batch_size, seq_length, embed_size]
            inputs_for_queries: [batch_size, seq_length, embed_size]

        Returns:
            output: [batch_size, seq_length, embed_size]
        """
        # 1. Ensure consistent dtypes
        inputs_for_keys = tf.cast(inputs_for_keys, tf.float32)
        inputs_for_values = tf.cast(inputs_for_values, tf.float32)
        inputs_for_queries = tf.cast(inputs_for_queries, tf.float32)

        # TODO: Apply each attention head
        # TODO: Concatenate head outputs
        # TODO: Apply output projection

        return NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_size": self.embed_size,
            "num_heads": self.num_heads
        })
        return config
    
@keras.saving.register_keras_serializable(package="transformer")
class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional encoding for transformer inputs.
    Uses sinusoidal position encodings as described in "Attention Is All You Need".
    """

    def __init__(self, d_model: int, max_seq_length: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Create positional encoding matrix
        pe = self.get_positional_encoding(max_seq_length, d_model)
        self.positional_encoding = tf.Variable(
            initial_value=pe, trainable=False, name='positional_encoding'
        )

    def get_positional_encoding(self, seq_length: int, d_model: int) -> tf.Tensor:
        """Generate sinusoidal positional encodings."""
        # TODO: Implement sinusoidal positional encodings
        # TODO: Use sine for even indices and cosine for odd indices (tf.sin, tf.cos)

        # TODO: Interleave sine and cosine: stack and reshape [seq_length, d_model]
        # TODO: Return tensor of shape [1, seq_length, d_model] (hint: use tf.expand_dims)
        
        return NotImplementedError

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            inputs with positional encodings added
        """
        seq_length = tf.shape(inputs)[1]

        # TODO: Extract appropriate slice of positional encodings

        # TODO: Add to inputs (residual connection)
        return NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "max_seq_length": self.max_seq_length
        })
        return config
    
@keras.saving.register_keras_serializable(package="transformer")
class LanguageTransformerBlock(keras.layers.Layer):
    """Single transformer block optimized for language modeling (no cross-attention)"""

    def __init__(self, embed_size, num_heads=8, ff_hidden_size=None, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.ff_hidden_size = ff_hidden_size or 4 * embed_size
        self.dropout_rate = dropout_rate

        # TODO: Initialize self-attention

        # TODO: Initialize feed-forward network (2 layers)
        # First layer: embed_size -> ff_hidden_size with activation
        # Second layer: ff_hidden_size -> embed_size

        # TODO: Initialize layer normalization layers

        # TODO: Initialize dropout layers

    def call(self, inputs, training=None):
        """
        Apply transformer block with residual connections and layer normalization.

        Args:
            inputs: [batch_size, seq_length, embed_size]
            training: Whether in training mode

        Returns:
            output: [batch_size, seq_length, embed_size]
        """
        # 1. Ensure consistent dtype
        inputs = tf.cast(inputs, tf.float32)

        # TODO: Self-attention with residual connection and layer norm

        # TODO: Feed-forward with residual connection and layer norm

        return NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_size": self.embed_size,
            "num_heads": self.num_heads,
            "ff_hidden_size": self.ff_hidden_size,
            "dropout_rate": self.dropout_rate
        })
        return config
    
@keras.saving.register_keras_serializable(package="transformer")
class TransformerLanguageModel(keras.Model):
    """
    Complete Transformer Language Model
    """

    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=None,
                 max_seq_length=512, dropout_rate=0.1, pad_token_id=0, **kwargs):
        super().__init__(**kwargs)

        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff or 4 * d_model
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        self.pad_token_id = pad_token_id

        # TODO: Initialize token embeddings

        # TODO: Create positional encodings (d_model, max_seq_length)

        # TODO: Initialize embedding dropout

        # TODO: Create transformer blocks (n_layers of LanguageTransformerBlock)

        # TODO: Initialize final layer normalization

        # TODO: Initialize transformer dropout

        # TODO: Initialize output projection to vocabulary


    def call(self, inputs, training=None):
        """
        Forward pass through the language model.

        Args:
            inputs: Token indices [batch_size, seq_length]
            training: Whether in training mode

        Returns:
            Logits over vocabulary [batch_size, seq_length, vocab_size]
        """
        # 1. Get token embeddings and scale by sqrt(d_model)
        embeddings = self.token_embedding(inputs)  # [batch_size, seq_length, d_model]
        embeddings = embeddings * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # TODO: Add positional encodings (remember to slice to seq_length)

        # TODO: Apply dropout to embeddings

        # TODO: Pass embeddings through transformer blocks using a loop

        # TODO: Apply final layer normalization

        # TODO: Project to vocabulary and return logits
        
        return NotImplementedError

    def get_config(self):
        """Get model configuration for saving."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'max_seq_length': self.max_seq_length,
            'dropout_rate': self.dropout_rate,
            'pad_token_id': self.pad_token_id
        }

def create_language_model(vocab_size: int, **kwargs) -> TransformerLanguageModel:
    """
    Factory function to create a language model with sensible defaults.

    Args:
        vocab_size: Size of the vocabulary
        **kwargs: Additional model parameters

    Returns:
        Initialized TransformerLanguageModel
    """
    default_config = {
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'max_seq_length': 256,
        'dropout_rate': 0.1,
        'pad_token_id': 0
    }

    # Update with provided kwargs
    config = {**default_config, **kwargs}
    config['vocab_size'] = vocab_size

    return TransformerLanguageModel(**config)