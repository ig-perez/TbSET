import numpy as np
import tensorflow as tf

# .....................................................................................................................


def positional_encoding(position, d_model):

  def _get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2*(i//2))/np.float32(d_model))  # TODO: Why i//2?

    return pos * angle_rates

  angle_rads = _get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(d_model, dff):
  """
  Point wise feed forward network consists of two fully-connected layers with a
  ReLU activation in between.
  """

  # TODO: Why we don't have a softmax?
  return tf.keras.Sequential([
                              tf.keras.layers.Dense(dff, activation="relu"),  # (b, n, dff)
                              tf.keras.layers.Dense(d_model)  # (b, n, d)
                              ])

# .....................................................................................................................


class MultiHeadAttention(tf.keras.layers.Layer):
  """
  Each multi-head attention block gets three inputs; Q (query), K (key),
  V (value). These are put through linear (Dense) layers (here we learn the
  weights W_Q, W_K, W_V) before the multi-head attention function. Instead of
  one single attention head, Q, K, and V are split into multiple heads because
  it allows the model to jointly attend to information from different
  representation subspaces at different positions. After the split each head
  has a reduced dimensionality, so the total computation cost is the same as a
  single head attention with full dimensionality.
  """

  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()

    self.num_heads = num_heads
    self.d_model = d_model

    # We want to dive the total dim. of the model into the available heads
    assert self.d_model % self.num_heads == 0

    # The number of dimensions each head will take care of
    self.depth = d_model // num_heads

    # With these layers we'll learn W_Q, W_K, W_V
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    # The last dense layer that receives the concat representation of the heads to produce final output
    self.dense = tf.keras.layers.Dense(d_model)

  def _split_heads(self, x, batch_size):
    """
    As x's shape is (b,n,d) we want that each head process depth = d//num_heads.
    This method split the last dimension (d or d_model).
    """

    # Split d_model dimension to consider "depth". 2nd dim is n or sequence len
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

    # Check the Transformer article on my site to understand the dimensions workout
    return tf.transpose(x, perm=[0, 2, 1, 3])  # (b, h, n, d//h)

  def _scaled_dot_product_attention(self, q, k, v, mask):
    """
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    # First, calculate QK^T
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (b, h, n_q, n_k)

    # Scale the previous calculation. d_k (depth of k) comes from k:(b, h, n_k, d_k//h)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (b, h, n_q, n_k)

    # TODO: Why? ... Adding the mask to the scaled tensor:: The mask is
    # multiplied with -1e9 (close to negative infinity). This is done because
    # the mask is summed with the scaled matrix multiplication of Q and K and is
    # applied immediately before a softmax. The goal is to zero out these cells,
    # and large negative inputs to softmax are near zero in the output.
    if mask is not None:  # TODO: HEY! if mask: ... will not work! check datatype
      scaled_attention_logits += (mask * -1e9)

    # Applying softmax. softmax is normalized on the last axis (n_k) so that the
    # scores add up to 1.

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (b, h, n_q, n_k)

    # Multiply with V to get the output
    output = tf.matmul(attention_weights, v)  # (b, h, n_q, d_v//h)

    return output, attention_weights

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]  # Or v or k

    q = self.wq(q)  # Learn W_Q ... (b, n, d)
    k = self.wk(k)  # Learn W_K ... (b, n, d)
    v = self.wv(v)  # Learn W_V ... (b, n, d)

    q = self._split_heads(q, batch_size)  # (b, h, n_q, d//h)
    k = self._split_heads(k, batch_size)  # (b, h, n_k, d//h)
    v = self._split_heads(v, batch_size)  # (b, h, n_v, d//h)

    # scaled_attention: (b, h, n_q, d//h)
    # attention_weights: (b, h, n_q, n_k)
    scaled_attention, attention_weights = self._scaled_dot_product_attention(q, k, v, mask)

    # Switch dimensions to undo split done by _split_heads
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (b, n_q, h, d//h)

    # Undo splitting: concat each head's representation into one
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (b, n_q, d)

    # Last layer processing before output
    output = self.dense(concat_attention)  # (b, n_q, d)

    return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
  """
  Each encoder layer consists of sublayers:
    - Multi-head attention (with padding mask)
    - Point wise feed forward networks.

  Each of these sublayers has a residual connection around it followed by a
  layer normalization. Residual connections help in avoiding the vanishing
  gradient problem in deep networks. The output of each sublayer is
  `LayerNorm(x + Sublayer(x))`. The normalization is done on the `d_model`
  (last) axis. There are N encoder layers in the transformer.

  Parameters
  ----------
  - `d_model`: The dimensionality of the internal vector representation
  - `dff`    : The dimensionality of the FFN
  """

  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # TODO: Research this layer type
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # TODO: Research this layer type

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):
    # Attention to the input x
    attn_output, _ = self.mha(x, x, x, mask)  # (b, n, d) TODO: Research how mask works/is implemented
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (b, n, d)

    # Attention output to the FFN
    ffn_output = self.ffn(out1)  # (b, n, d)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)

    return out2


class DecoderLayer(tf.keras.layers.Layer):
  """
  Each decoder layer consists of sublayers:
    - Masked multi-head attention (with look ahead mask and padding mask)
    - Multi-head attention (with padding mask). V (value) and K (key) receive
      the encoder output as inputs. Q (query) receives the output from the
      masked multi-head attention sublayer.
    - Point wise feed forward networks

  These sublayers has a residual connection like with the EncoderLayer. The
  normalization is done on the d_model (last) axis.

  There are N decoder layers in the transformer. As Q receives the output
  from decoder's first attention block, and K receives the encoder output,
  the attention weights represent the importance given to the decoder's input
  based on the encoder's output. In other words, the decoder predicts the
  next token by looking at the encoder output and self-attending to its own
  output.

  Parameters
  ----------
  - `d_model`: The dimensionality of the internal vector representation
  - `dff`    : The dimensionality of the FFN
  """

  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)  # For the target it receives as input
    self.mha2 = MultiHeadAttention(d_model, num_heads)  # For the output it receives from the encoder

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # After mha1
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # After mha2
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # After the FFN

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    attn1, att_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (b, n, d)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, att_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (b, n, d)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (b, n, d)

    ffn_output = self.ffn(out2)  # (b, n, d)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (b, n, d)

    # TODO: Research why do we need these att. weights?
    return out3, att_weights_block1, att_weights_block2


class Encoder(tf.keras.layers.Layer):
  """
  The Encoder consists of:
    - Input Embedding
    - Positional Encoding
    - N encoder layers

  The input is put through an embedding which is summed with the positional
  encoding. The output of this summation is the input to the encoder layers.
  The output of the encoder is the input to the decoder.

  Parameters
  ----------
  - `d_model`: The dimensionality of the internal vector representation
  """

  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    # An encoder is made of an Embedding layer:
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    # A positional encoding layer for the inputs:
    self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            self.d_model)  # maximum_position_encoding == pe_input from Transformer
    # Several Encoder layers (or blocks):
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for block in range(self.num_layers)]
    # Also a Dropout layer (it has a FNN inside):
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training: bool, mask):
    """
    A complete set of calculations for encoding the input sequence.

    Parameters
    ----------
    - `x`       : A ??? containing a batch of training examples as input
                  sequences. Its shape is (b, n, d) == (b, input_seq_len, d_model)
    - `training`: Indicates whether the layer should behave in training mode
                  (adding dropout) or in inference mode (doing nothing).
    """

    seq_len = tf.shape(x)[1]  # The number of words in all samples of current batch (padded?)

    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # BY_EXPERIENCE: A normalization term!
    x += self.pos_encoding[:, :seq_len,
         :]  # Remember (b, n, d). This applies only in the 2nd dimension. TODO: Understand what is being done here

    # The dropout before the Encoder blocks!
    x = self.dropout(x, training=training)

    # Now the calculations among the blocks
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)  # Parameters for EncoderLayer.call()

    return x  # This goes to the Decoder, (b, n, d)


class Decoder(tf.keras.layers.Layer):
  """
  The Decoder consists of:
    - Output Embedding
    - Positional Encoding
    - N decoder layers

  The target is put through an embedding which is summed with the positional
  encoding. The output of this summation is the input to the decoder layers.
  The output of the decoder is the input to the final linear layer.

  Parameters
  ----------
  - `d_model`: The dimensionality of the internal vector representation
  - `dff`    : The dimensionality of the FFN
  """

  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            d_model)  # maximum_position_encoding == pe_target from Transformer
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for block in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    seq_len = tf.shape(x)[1]  # TODO: Like with the Decoder, get an idea of how does this look like
    attention_weights = {}  # TODO: Why this?

    x = self.embedding(x)  # (b, n, d). Where n is the target seq_len
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # BY_EXPERIENCE: A normalization term!
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    # This call() calls the DecoderLayer's call() on each iteration
    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
      attention_weights[f"decoder_layer{i + 1}_block1"] = block1
      attention_weights[f"decoder_layer{i + 1}_block2"] = block2

    return x, attention_weights  # x's shape is (b, n, d)


class Transformer(tf.keras.Model):
  """
  The transformer model follows the same general pattern as a standard
  sequence to sequence with attention model.

  The input sentence is passed through N encoder layers that generates an
  output for each token in the sequence. The decoder attends to the encoder's
  output and its own input (self-attention) to predict the next word.

  Parameters
  ----------
  - `x`       : A ??? containing a batch of training examples as input
                sequences. Its shape is (b, n, d) == (b, input_seq_len, d_model)
  - `training`: Indicates whether the layer should behave in training mode
                (adding dropout) or in inference mode (doing nothing).
  """

  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    # A transformer is made of an Encoder:
    self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
    # A Decoder:
    self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
    # A linear layer:
    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def _create_masks(self, inp, tar):
    def _create_look_ahead_mask(size):
      """
      Mask the future tokens in a sequence within the Decoder. In other words,
      the mask indicates which entries should not be used. This means that to
      predict the third token, only the first and second token will be used.
      Similarly to predict the fourth token, only the first, second and the third
      tokens will be used and so on.

      Parameters
      ----------
      - `size` : Used to create a `size \times size` matrix where diagonal and
                 below values are zeros, and upper values are ones.
      """

      # -1 will preserve all items under the diagonal, 0 means none above diagonal will be kept
      mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

      # TODO: How this mask works? what is the meaning of each row?
      return mask

    def _create_padding_mask(seq):
      """
      The "[PAD]" tokens makes all input sequences the same size. But we don't
      want the Transformer to consider these tokens or learn about them. This
      function returns a tensor mask with 1 on the positions where a pad token is
      present for each sequence, and 0 otherwise.

      Parameters
      ----------
      - `seq` : A batch of input sequences made of integer values which are
                indexes of a dictionary. Its shape is (b, n)
      """

      # The pad token index is 0 in the vocabularies
      seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

      # TODO: Why do we need extra dims?. It says: add extra dimensions to add the padding to the attention logits.
      return seq[:, tf.newaxis, tf.newaxis, :]  # (b, 1, 1, n)

    # Padding mask for the inputs feeding the Encoder
    enc_padding_mask = _create_padding_mask(inp)  # (b, 1, 1, n)

    # Padding mask for the Encoder's output going to the Decoder
    dec_padding_mask = _create_padding_mask(inp)  # (b, 1, 1, n)

    # Look ahead mask for the target feed to the Decoder's 1st Attention block
    look_ahead_mask = _create_look_ahead_mask(tf.shape(tar)[1])  # Is tar (b,n,d)?
    dec_target_padding_mask = _create_padding_mask(tar)
    look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    # TODO: Why maximum!
    return enc_padding_mask, look_ahead_mask, dec_padding_mask

  # Keras expects all your inputs in the first argument
  def call(self, inputs, training):
    """
    Executes a complete calculation for the Transformer.

    Parameters
    ----------
    - `inputs`: A tuple of inputs and targets.
    """

    inp, tar = inputs  # es, en

    enc_padding_mask, look_ahead_mask, dec_padding_mask = self._create_masks(inp, tar)

    enc_output = self.encoder(inp, training, enc_padding_mask)  # (b, inp_n, d)
    dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)  # (b, tar_n, d)
    final_output = self.final_layer(dec_output)

    return final_output, attention_weights