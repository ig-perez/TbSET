import numpy as np
import tensorflow as tf

# TF type annotations not yet implemented as of sept. 21: https://github.com/tensorflow/tensorflow/issues/12345
from typing import Any

# TODO: REMOVE THIS BEFORE TRAINING!
# tf.config.run_functions_eagerly(True)

# .....................................................................................................................


def positional_encoding(position: int, dim_model: int) -> Any:
	"""
	This a helper function for the Decoder and Encoder. It calculates the positional encodings used when feeding the
	inputs to the Encoder/Decoder. The method returns a (1, max_position_enc, d_model) tensor. This means we have up to
	"max_position_enc" d_model-dimensional vectors that can be added to the word embeddings before input the sequences
	to the Transformer.

	:param position: The max. possible positions to consider in an input sequence.
	:param dim_model: Dimensionality of the internal vector representation of the Transformer (equals to the embeddings
					size).

	:return: A (1, max_position_enc, d_model) tensor containing positional encodings.
	"""

	def _get_angles(pos: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
		"""
		Calculates a d_model positional embedding where each dimension is represented with a different sinusoid. The
		wavelengths form a geometric progression from 2π to 10000⋅2π.

		:param pos: A column vector (max_position_enc, 1) representing the max. possible positions to consider in an
		input sequence.
		:param i: A row vector (1, d_model) representing the dimensionality of the token embeddings. Each dimension
		of the positional encoding corresponds to a sinusoid.
		:param d_model: Dimensionality of the internal vector representation of the Transformer. In this case
		corresponds to the Embedding size.

		:return: A (max_position_enc, d_model) matrix containing positional encodings.
		"""

		angle_rates = 1 / np.power(10000, (2*(i//2))/np.float32(d_model))  # TODO: Why i//2?

		return pos * angle_rates  # (max_position_enc, d_model) == (1000, 128)

	# Calculate all the positional encodings up to "position"
	angle_rads = _get_angles(np.arange(position)[:, np.newaxis], np.arange(dim_model)[np.newaxis, :], dim_model)

	# apply sin to even indices in the array; 2i -> (start:stop:step)
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
	# apply cos to odd indices in the array; 2i+1
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

	pos_encoding = angle_rads[np.newaxis, ...]

	return tf.cast(pos_encoding, dtype=tf.float32)  # (1, max_position_enc, d_model)


def point_wise_feed_forward_network(d_model: int, dff: int) -> tf.keras.Sequential:
	"""
	This a helper function for the Decoder and Encoder. Defines and returns a point wise feed forward network which
	consists of two fully-connected layers with a ReLU activation in between. These NN process the output of the
	MultiHeadAttention blocks.

	:param d_model: Dimensionality of the internal vector representation of the Transformer
	:param dff: Dimensionality of the NN's output space (number of units).

	:return: A sequential model composed by two densely connected layers to process the MultiHeadAttention blocks.
	"""

	return tf.keras.Sequential([
		tf.keras.layers.Dense(dff, activation="relu"),  # (b, n, dff)
		tf.keras.layers.Dense(d_model)  # (b, n, d)
	])

# .....................................................................................................................


class MultiHeadAttention(tf.keras.layers.Layer):

	def __init__(self, d_model: int, num_heads: int) -> None:
		"""
		Each multi-head attention block gets three inputs; Q (query), K (key), V (value). These are put through linear
		(Dense) layers to learn the weights W_Q, W_K, W_V. Instead of one single attention head, Q, K, and V are split
		into multiple heads because it allows the model to jointly attend to information from different representation
		subspaces at different positions. After the split, each head has a reduced dimensionality, so the total
		computation cost is the same as a single head attention with full dimensionality.

		:param d_model: Dimensionality of the internal vector representation of the Transformer
		:param num_heads: The number of Attention heads for the MultiHeadAttention module.

		:return: None.
		"""

		super(MultiHeadAttention, self).__init__()

		self.num_heads = num_heads
		self.d_model = d_model

		# We want to divide the total dim. of the model into the available heads
		assert self.d_model % self.num_heads == 0

		self.depth = d_model // num_heads  # depth  = d//h

		# Linear layers to learn W_Q, W_K, W_V
		self.wq = tf.keras.layers.Dense(d_model)
		self.wk = tf.keras.layers.Dense(d_model)
		self.wv = tf.keras.layers.Dense(d_model)

		# The last dense layer that receives the concat representation of the heads to produce final output
		self.dense = tf.keras.layers.Dense(d_model)

	def _split_heads(self, x: Any, batch_size: int) -> Any:
		"""
		As x's shape is (b,n,d) we want that each head process depth = d//num_heads. This method split the last
		dimension (d or d_model).

		:param x: A (b, n, d) tensor containing embeddings from the source or target datasets we want to pay
				attention to.
		:param batch_size: The number of training examples in current set.

		:return: A (b, h, n, d//h) tensor containing the dimension values of the current batch each head will pay
				attention to.
		"""

		# Split d_model dimension to consider "depth". 2nd dim is n or seq_len
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

		# Check https://ivanperez.pe/blog/nlp05-transformer-pretraining to understand the dimensions workout
		return tf.transpose(x, perm=[0, 2, 1, 3])  # (b, h, n, d//h)

	@staticmethod
	def _scaled_dot_product_attention(q: Any, k: Any, v: Any, mask: Any) -> tuple:
		"""
		Calculate the attention weights. q, k, v must have matching leading dimensions. k, v must have matching
		penultimate dimension, i.e.: n_k (or seq_len_k) = n_v. The mask has different shapes depending on its type
		(padding or look ahead) but it must be broadcastable for addition.

		:param q: query with shape (b, h, n_q, depth_q)
		:param k: key with shape (b, h, n_k, depth_k)
		:param v: value with shape (b, h, n_v, depth_v)
		:param mask: A float tensor with shape broadcastable to (b, h, n_q, n_k). Defaults to None. The Encoder uses a
					padding mask to avoid attending the PAD tokens. The Decoder uses a look_ahead mask for the first
					Attention block and a padding mask in the second Attention block (to process Encoder's output).

		:return: A tuple containing the result of the Attention calculations and the attention_weights tensor.
		"""

		# First, calculate QK^T
		matmul_qk = tf.matmul(q, k, transpose_b=True)  # (b, h, n_q, n_k)

		# Scale the previous calculation. d_k (depth of k) comes from k:(b, h, n_k, d_k=depth_k=d//h)
		dk = tf.cast(tf.shape(k)[-1], tf.float32)
		scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (b, h, n_q, n_k) sqrt needed to have stable gradients

		# Adding the mask to the scaled tensor:: The mask is multiplied with -1e9 (close to negative infinity). This is
		# done because the mask is summed with the scaled matrix multiplication of Q and K and is applied immediately
		# before a softmax. The goal is to zero out these cells, and large negative inputs to softmax are near zero in
		# the output.

		# Apply the mask (padding or look_ahead). Look_ahead used in decoder to only attend prev/current tokens when 
		# predicting. Both masks uses 1 for values we don't want to consider like the PAD tokens or the "future" tokens
		# we don't want to take into account when predicting. 1 will be turn into -inf, so after softmax will be zero.
		if mask is not None:  # if mask: will not work
			scaled_attention_logits += (mask * -1e9)

		# Softmax is normalized on the last axis (n_k) so that the scores add up to 1.
		attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (b, h, n_q, n_k)

		# Multiply with V to get the output
		output = tf.matmul(attention_weights, v)  # (b, h, n_q, d_v//h)

		return output, attention_weights

	def call(self, v: Any, k: Any, q: Any, mask: Any) -> tuple:
		"""
		Calculate the attention weights for each head and the merging the results into one tensor.

		:param q: query with shape (b, n_q, d)
		:param k: key with shape (b, n_k, d)
		:param v: value with shape (b, n_v, d)
		:param mask: Float tensor with shape broadcastable to (b, h, n_q, n_k). Fx to create masks returns (b, 1, 1, n)

		:return: A tuple containing the result of the Attention calculations and the attention_weights tensor.
		"""

		batch_size = tf.shape(q)[0]  # Or v or k

		q = self.wq(q)  # Learn W_Q ... (b, n, d)
		k = self.wk(k)  # Learn W_K ... (b, n, d)
		v = self.wv(v)  # Learn W_V ... (b, n, d)

		q = self._split_heads(q, batch_size)  # (b, h, n_q, d//h)
		k = self._split_heads(k, batch_size)  # (b, h, n_k, d//h)
		v = self._split_heads(v, batch_size)  # (b, h, n_v, d//h)

		# scaled_attention: (b, h, n_q, d//h) | attention_weights: (b, h, n_q, n_k)
		scaled_attention, attention_weights = self._scaled_dot_product_attention(q, k, v, mask)

		# Switch dimensions to undo split done by _split_heads
		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (b, n_q, h, d//h)

		# Undo splitting: concat each head's representation into one
		concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (b, n_q, d)

		# Last layer processing before output
		output = self.dense(concat_attention)  # (b, n_q, d)

		return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):

	def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1) -> None:
		"""
		Each encoder layer consists of two sublayers:
			- Multi-head attention (with padding mask)
			- Point wise feed forward networks

		Each of these sublayers has a residual connection around it followed by a layer normalization. Residual
		connections help in avoiding the vanishing gradient problem in deep networks. The output of each sublayer is
		`LayerNorm(x + Sublayer(x))`. The normalization is done on the `d_model` (last) axis. There are N encoder
		layers in the transformer.

		:param d_model: Dimensionality of the internal vector representation of the Transformer
		:param num_heads: The number of Attention heads for the MultiHeadAttention module.
		:param dff: Dimensionality of the NN's output space (number of units).
		:param rate: The percentaje of units to turn off during training.

		:return: None.
		"""

		super(EncoderLayer, self).__init__()

		# First sub-block
		self.mha = MultiHeadAttention(d_model, num_heads)
		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # Add & Norm 1

		# Second sub-block
		self.ffn = point_wise_feed_forward_network(d_model, dff)
		self.dropout2 = tf.keras.layers.Dropout(rate)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # Add & Norm

	def call(self, x: Any, training: bool, mask: Any) -> Any:
		"""
		A complete set of calculations to apply Attention to the batch of input sequences.

		:param x: A (b, n_inp, d) tensor containing training examples to be used as inputs. Where `b` is the
				`batch_size`, `n_inp` is the length of the input padded examples, and `d` is the embeddings
				dimensionality.
		:param training: Indicates how the layer should behave in training mode (adding dropout) or in inference mode
						(doing nothing).
		:param mask: A (b, 1, 1, n_inp) tensor containing the padding mask for the Encoder.

		:return: A (b, n_inp, d) tensor containing the representation of the inputs batch after the current Encoder
				layer.
		"""

		# Attention to the input x
		attn_output, _ = self.mha(x, x, x, mask)  # (b, n_inp, d)
		attn_output = self.dropout1(attn_output, training=training)
		out1 = self.layernorm1(x + attn_output)  # (b, n_inp, d)

		# Attention output to the FFN
		ffn_output = self.ffn(out1)  # (b, n_inp, d)
		ffn_output = self.dropout2(ffn_output, training=training)
		out2 = self.layernorm2(out1 + ffn_output)

		return out2  # (b, n_inp, d)


class DecoderLayer(tf.keras.layers.Layer):

	def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1) -> None:
		"""
		Each decoder layer consists of sublayers:
			- Masked multi-head attention (with look ahead mask and padding mask)
			- Multi-head attention (with padding mask). V (value) and K (key) receive the encoder output as inputs. Q
			(query) receives the output from the masked multi-head attention sublayer.
			- Point wise feed forward networks

		These sublayers has a residual connection like with the EncoderLayer. The normalization is done on the d_model
		(last) axis.

		There are N decoder layers in the transformer. As Q receives the output from decoder's first attention block,
		and K receives the encoder output, the attention weights represent the importance given to the decoder's input
		based on the encoder's output. In other words, the decoder predicts the next token by looking at the encoder
		output and self-attending to its own output.

		:param d_model: Dimensionality of the internal vector representation of the Transformer
		:param num_heads: The number of Attention heads for the MultiHeadAttention module.
		:param dff: Dimensionality of the NN's output space (number of units).
		:param rate: The percentaje of units to turn off during training.

		:return: None.
		"""

		super(DecoderLayer, self).__init__()

		# First Decoder sub-block
		self.mha1 = MultiHeadAttention(d_model, num_heads)  # For the target it receives as input
		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # After mha1

		# Second Decoder sub-block
		self.mha2 = MultiHeadAttention(d_model, num_heads)  # For the output it receives from the encoder
		self.dropout2 = tf.keras.layers.Dropout(rate)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # After mha2

		# Third Decoder sub-block
		self.ffn = point_wise_feed_forward_network(d_model, dff)
		self.dropout3 = tf.keras.layers.Dropout(rate)
		self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # After the FFN

	def call(self, x: Any, enc_output: Any, training: bool, look_ahead_mask: Any, padding_mask: Any) -> Any:
		"""
		A complete set of calculations to apply Attention to the batch of input sequences.

		:param x: A (b, n_tar, d) tensor containing a batch of corresponding translations for current input sentences.
				These sentences are shifted right one position so the Decoder don't learn to copy the word it should
				predict. `b` is the `batch_size`, `n_tar` is the length of the input padded examples, and `d` is the
				embeddings dimensionality.
		:param enc_output: A (b, n_inp, d) tensor containing the Encoder's representation of the current training batch
		:param training: Indicates how the layer should behave in training mode (adding dropout) or in inference mode
						(doing nothing).
		:param look_ahead_mask: A (b, 1, n_tar, n_tar) tensor containing the look_ahead mask for the Decoder.
		:param padding_mask: A (b, 1, 1, n_inp) tensor containing the padding mask for the 2nd MHA in the Decoder.

		:return: A tuple of three elements. The current Decoder layer output (b, n_tar, d), the Attention weights from
				the Masked MultiHeadAttention block (b, h, n_tar, n_tar), and the Attention weights from the
				MultiHeadAttention block that receives the Encoder's output (b, h, n_tar, n_inp).
		"""

		attn1, att_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (b, n_tar, d)
		attn1 = self.dropout1(attn1, training=training)
		out1 = self.layernorm1(attn1 + x)

		attn2, att_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (b, n_tar, d)
		attn2 = self.dropout2(attn2, training=training)
		out2 = self.layernorm2(attn2 + out1)  # (b, n_tar, d)

		ffn_output = self.ffn(out2)  # (b, n_tar, d)
		ffn_output = self.dropout3(ffn_output, training=training)
		out3 = self.layernorm3(ffn_output + out2)  # (b, n_tar, d)

		# att_weights_block1 is (b, h, n_tar, n_tar), att_weights_block2 is (b, h, n_tar, n_inp)
		return out3, att_weights_block1, att_weights_block2  # out3 is (b, n_tar, d)


class Encoder(tf.keras.layers.Layer):

	def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding,
				 rate=0.1) -> None:
		"""
		The Encoder consists of:
			- Input Embedding
			- Positional Encoding
			- N Encoder Layers

		The input is put through an embedding which is summed with the positional encoding. The output of this
		summation is the input to the encoder layers. The output of the encoder is the input to the decoder.

		:param num_layers: Number of Encoder Layers to create in the Encoder block.
		:param d_model: Dimensionality of the internal vector representation of the Transformer (equals to the
						embeddings size in this case).
		:param num_heads: The number of Attention heads for the MultiHeadAttention module.
		:param dff: Dimensionality of the NN's output space (number of units).
		:param input_vocab_size: The top number of tokens to consider in the source vocabulary.
		:param maximum_position_encoding: The number of positional encodings to generate.
		:param rate: The percentaje of units to turn off during training.

		:return: None.
		"""

		super(Encoder, self).__init__()

		self.d_model = d_model
		self.num_layers = num_layers

		self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
		# maximum_position_encoding == pe_input from Transformer
		self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)  # (1000, 128)
		self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(self.num_layers)]  # _ == block
		self.dropout = tf.keras.layers.Dropout(rate)

	def call(self, x: Any, training: bool, mask: Any) -> Any:
		"""
		A complete set of calculations for encoding the input sequence.

		:param x: A (b, n_inp) tensor containing training examples to be used as inputs. Where `b` is the `batch_size`,
				and `n_inp` is the length of the input padded examples.
		:param training: Indicates how the layer should behave in training mode (adding dropout) or in inference mode
						(doing nothing).
		:param mask: A (b, 1, 1, n_inp) tensor containing the padding mask for the Encoder.

		:return: A (b, n_inp, d) tensor containing the representation of each training example in current batch after
				applying the Attention calculations.
		"""

		seq_len = tf.shape(x)[1]  # It is likely that the Target batch have a different sequence length

		x = self.embedding(x)  # (b, n_inp = seq_len, d)
		x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # BY_EXPERIENCE: A normalization term!
		x += self.pos_encoding[:, :seq_len, :]  # Adds the position encoding to sequences (seq_len) in current batch
		x = self.dropout(x, training=training)  # (b, n_inp, d)

		for i in range(self.num_layers):
			x = self.enc_layers[i](x, training, mask)  # Attention calculations and more

		return x  # This goes to the Decoder for current batch: (b, n_inp, d)


class Decoder(tf.keras.layers.Layer):

	def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, target_vocab_size: int,
				 maximum_position_encoding: int, rate: float = 0.1) -> None:
		"""
		The Decoder consists of:
			- Output Embedding
			- Positional Encoding
			- N decoder layers

		The target is put through an embedding which is summed with the positional encoding. The output of this summation
		is the input to the decoder layers. The output of the decoder is the input to the final linear layer.

		:param num_layers: Number of Decoder Layers to create in the Encoder block.
		:param d_model: Dimensionality of the internal vector representation of the Transformer (equals to the
						embeddings size in this case).
		:param num_heads: The number of Attention heads for the MultiHeadAttention module.
		:param dff: Dimensionality of the NN's output space (number of units).
		:param target_vocab_size: The top number of tokens to consider in the target vocabulary.
		:param maximum_position_encoding: The number of positional encodings to generate.
		:param rate: The percentaje of units to turn off during training.

		:return: None.
		"""

		super(Decoder, self).__init__()

		self.d_model = d_model
		self.num_layers = num_layers

		self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
		self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
		self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]  # _ ~ block
		self.dropout = tf.keras.layers.Dropout(rate)

	def call(self, x: Any, enc_output: Any, training: bool, look_ahead_mask: Any, padding_mask: Any) -> Any:
		"""
		A complete set of calculations for decoding the input sequence and making a prediction.

		:param x: A (b, n_tar) tensor containing training examples to be used as inputs. Where `b` is the `batch_size`,
				and `n_inp` is the length of the tokenized padded examples.
		:param enc_output: A (b, n_inp, d) tensor containing the representation build by the Encoder block.
		:param training: Indicates how the layer should behave in training mode (adding dropout) or in inference mode
						(doing nothing).
		:param look_ahead_mask: A (b, 1, n_tar, n_tar) tensor containing the look_ahead mask for the Decoder.
		:param padding_mask: A (b, 1, 1, n_inp) tensor containing the padding mask to process the 2nd MHA with the
							Encoder's output + the 1st Decoder's MHA output.

		:return: A tuple consisting of a (b, n_tar, d) tensor with the decoded sequence (logits) and a dictionary with
				the Attention weights for each of the Attention heads. The weight of each head is a (b, h, n_tar, n_tar
				or n_inp) tensor.
		"""

		seq_len = tf.shape(x)[1]  # n_tar
		attention_weights = {}  # Each head's Attention weights will be stored here

		x = self.embedding(x)  # (b, n_tar, d)
		x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # BY_EXPERIENCE: A normalization term!
		x += self.pos_encoding[:, :seq_len, :]  # Adding the positional encodings to the input sequences
		x = self.dropout(x, training=training)

		for i in range(self.num_layers):
			x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
			attention_weights[f"decoder_layer{i + 1}_block1"] = block1
			attention_weights[f"decoder_layer{i + 1}_block2"] = block2

		return x, attention_weights  # (b, n_tar, d)


class Transformer(tf.keras.Model):

	def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, input_vocab_size: int,
				 target_vocab_size: int, pe_input: int, pe_target: int, rate: float = 0.1) -> None:
		"""
		The transformer model follows the same general pattern as a standard sequence to sequence with attention model.

		The input sentence is passed through N encoder layers that generates an output for each token in the sequence.
		The decoder attends to the encoder's output and its own input (self-attention) to predict the next word.

		:param num_layers: The number of Encoder/Decoder blocks in the Transformer.
		:param d_model: Dimensionality of the internal vector representation of the Transformer.
		:param num_heads: The number of Attention heads for the MultiHeadAttention module.
		:param dff: Dimensionality of the NN's output space (number of units) inside the Encoder/Decoder blocks.

		:param input_vocab_size: The number of tokens (subwords included) to consider when creating the vocabulary for
				the input language (Spanish).
		:param target_vocab_size: The number of tokens (subwords included) to consider when creating the vocabulary for
				the target language (English).
		:param pe_input: The maxinum number of positional encodings to generate for the input sequences.
		:param pe_target: The maxinum number of positional encodings to generate for the target sequences.
		:param rate: The percentaje of units to turn off during training.

		:return: None
		"""

		super(Transformer, self).__init__()

		# A transformer is made of an Encoder:
		self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
		# A Decoder:
		self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
		# A linear layer with vocab_size units to predict the translated token:
		self.final_layer = tf.keras.layers.Dense(target_vocab_size)

	@staticmethod
	def _create_masks(inp: Any, tar: Any) -> tuple:
		"""
		Create padding and "look ahead" masks for the input sequences of the source and target training examples. Both
		masks uses 1 for values we don't want to consider like the PAD tokens or the "future" tokens we don't want to
		take into account when predicting. 1 will be turn into -inf, so after softmax will be zero.

		:param inp: A (b, n_inp) tensor containing a batch of source input sequences (Spanish).
		:param tar: A (b, n_tar) tensor containing a batch of target input sequences (English).

		:return: A tuple with the padding mask to use for the Encoder inputs, look-ahead mask for the Decoder, and
				a padding mask for the Decoder when receives its inputs.
		"""

		def _create_look_ahead_mask(size: int) -> Any:
			"""
			Mask the future tokens in a sequence within the Decoder. In other words, the mask indicates which entries
			should not be used. This means that to predict the third token, only the first and second token will be
			used. Similarly to predict the fourth token, only the first, second and the third tokens will be used and
			so on.

			:param size: The target sequence length of current batch.

			:return: A `size x size` matrix where diagonal and below values are zeros, and upper values are ones.
			"""

			# -1 will preserve all items under the diagonal, 0 means none above diagonal will be kept
			mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

			# Mask will be summed with the scaled_attention_logits ~(QK^T) inside the _scaled_dot_product_attention
			# method. Those values we don't care about will be nullified by the Softmax function: Softmax(-inf) = 0
			return mask

		def _create_padding_mask(seq: Any) -> Any:
			"""
			The "[PAD]" tokens makes all input sequences the same size. But we don't want the Transformer to consider
			these tokens or learn about them.

			:param seq: A batch of input sequences with shape (b, n) where `b` is the batch size and `n` is the
						sequence length.

			:return: A tensor mask with 1 on the positions where a pad token is present for each sequence, and 0
					otherwise.
			"""

			# The pad token index is 0 in the vocabularies
			seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

			# We add the mask to the scaled_attention_logits in the _scaled_dot_product_attention method
			return seq[:, tf.newaxis, tf.newaxis, :]  # (b, n) -> (b, 1, 1, n) so we can sum it with (b, h, n_q, n_k)

		# Padding mask for the inputs feeding the Encoder
		enc_padding_mask = _create_padding_mask(inp)  # (b, 1, 1, n_inp)

		# This one masks the Encoder's output when goes to the Decoder in the 2nd MHA
		dec_padding_mask = _create_padding_mask(inp)  # (b, 1, 1, n_inp)

		# Mask for the target feed to the Decoder's 1st MHA
		look_ahead_mask = _create_look_ahead_mask(tf.shape(tar)[1])  # tar is (b,n_tar) then the mask is (n_tar, n_tar)
		dec_target_padding_mask = _create_padding_mask(tar)  # (b, 1, 1, n_tar)
		look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)  # Merge padding and look_ahead for tar

		return enc_padding_mask, look_ahead_mask, dec_padding_mask

	# Keras expects all your inputs in the first argument
	def call(self, inputs: tuple, training: bool) -> tuple:
		"""
		Executes a complete set of calculations to process a batch of inputs and its corresponding outputs as part of
		each training step.

		:param inputs: A tuple of inputs and targets. Each one containing batch_size padded sentences.
		:param training: Indicates how the layer should behave in training mode (adding dropout) or in inference mode
						(doing nothing).

		:return: A tuple containing the transformer final output for the current batch (b, n_tar, d), and the
				corresponding attention weights.
		"""

		inp, tar = inputs  # es, en. tar lacks the last token from the original batch

		enc_padding_mask, look_ahead_mask, dec_padding_mask = self._create_masks(inp, tar)

		# (b, n_inp, d)
		enc_output = self.encoder(inp, training, enc_padding_mask)

		# dec_output is (b, n_tar, d)
		dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

		final_output = self.final_layer(dec_output)

		# final_output is (b, n_tar, d)
		return final_output, attention_weights
