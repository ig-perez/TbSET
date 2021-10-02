import time
import logging
import tensorflow as tf

from typing import Any
from tbset.src.data import Dataset
from tbset.src.model import Transformer
from tbset.src.optimization import CustomSchedule, loss_function, accuracy_function

tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(3)

# .....................................................................................................................


class Trainer:

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, dropout_rate: float, ckpt_path: str,
                 save_path: str, epochs: int, dwn_destination: str, vocab_path: str, buffer_size: int, batch_size: int,
                 vocab_size: int, num_examples: int) -> None:
        """
        An object that trains and makes inference with a transformer model for Spanish-English translation.

        :param num_layers: The number of Encoder/Decoder blocks in the Transformer.
        :param d_model: Dimensionality of the internal vector representation of the Transformer.
        :param num_heads: The number of Attention heads for the MultiHeadAttention module.
        :param dff: Dimensionality of the NN's output space (number of units) inside the Encoder/Decoder blocks.
        :param dropout_rate: The percentaje of units to turn off during training.
        :param ckpt_path: The path where to store/search partial parameters from training.
        :param save_path: The path where to save/load the model after training.
        :param epochs: The number of epochs we'll train the model with the training batches.
        :param dwn_destination: The path where to download the training dataset.
        :param vocab_path: The path where to save/load the source/target vocabularies.
        :param buffer_size: The number of items to consider when shuffling the training data.
        :param vocab_size: The top number of tokens to include in the source/target vocabulary.
        :param num_examples: The number of examples to obtain from the training corpus.

        :return: None
        """

        super(Trainer, self).__init__()

        # Hyperparameters for training
        self.num_layers = int(num_layers)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dff = int(dff)
        self.dropout_rate = float(dropout_rate)
        self.ckpt_path = ckpt_path
        self.save_path = save_path
        self.epochs = int(epochs)

        # Hyperparameters for dataset workout
        self.dwn_destination = dwn_destination
        self.vocab_path = vocab_path
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.vocab_size = int(vocab_size)
        self.num_examples = int(num_examples)

        # Optimization initialization: Concrete values in point 3. Optimization, loss, and accuracy
        self.train_loss = None
        self.train_accuracy = None

    def train(self) -> None:
        """
        Training process of a Spanish-English neural translator using a Transformer model. Spanish is used as the input
        language and English as the target language.

        :return: None
        """

        # .............................................................................................................

        @tf.function(input_signature=[  # Define input signatures to avoid retracing
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64)])
        def _train_step(inp_batch: Any, tar_batch: Any) -> None:
            """
            An internal method to process each training step. The translated sentence (tar) is divided into tar_inp and
            tar_real. tar_inp is passed as input to the decoder. tar_real is that same input shifted by 1: At each
            location in tar_inp, tar_real contains the next token that should be predicted.

            :param inp_batch: A tokenized eager (b, n_int) tensor where `n_int` is the longest sentence length in the
                                batch. All shorter items have been padded with the PAD token (0).
            :param tar_batch: A tokenized eager (b, n_tar) tensor where `n_tar` is the longest sentence length in the
                                batch. All shorter items have been padded with the PAD token (0). In the 1st train step
                                the tar_inp token is <START>. The Decoder should predict the first token in inp_real.
                                The last token in tar_inp should be used to predict the last token in inp_real which is
                                <END>. This is why we don need to have the <END> token at the end of tar_inp.

            :return: None
            """

            tar_inp = tar_batch[:, :-1]  # Decoder's input. We don't need the <END> tokens here
            tar_real = tar_batch[:, 1:]  # Used to force teach and to calculate loss in prediction

            with tf.GradientTape() as tape:
                predictions, _ = transformer([inp_batch, tar_inp], training=True)  # inp -> Encoder, tar -> Decoder
                loss = loss_function(tar_real, predictions)

                # Calculate the gradients of the loss function w.r.t. its weights
                gradients = tape.gradient(target=loss, sources=transformer.trainable_variables)
                optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

                # Store the values
                self.train_loss(loss)  # Avg. loss of current batch
                self.train_accuracy(accuracy_function(tar_real, predictions))  # Avg. accuracy of current batch

        # .............................................................................................................

        # 1. Obtain batched training dataset
        print("INFO: Preparing training data ...")
        dataset_object = Dataset(
            dwn_destination=self.dwn_destination,
            vocab_path=self.vocab_path,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            vocab_size=self.vocab_size,
            num_examples=self.num_examples)

        train_batches, tst_batches, val_batches = dataset_object.workout_datasets()

        # 2. Instantiate the model
        print("INFO: Instantiating the model ...")
        transformer = Transformer(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            input_vocab_size=dataset_object.tokenizers.es.get_vocab_size().numpy(),  # input == ES
            target_vocab_size=dataset_object.tokenizers.en.get_vocab_size().numpy(),
            pe_input=1000,  # Max. Positional encoding for input ES
            pe_target=1000,  # Max. Positional encoding for target EN
            rate=self.dropout_rate)

        # 3. Optimization, loss, and accuracy
        print("INFO: Preparing optimizer, loss, and accuracy functions ...")
        learning_rate = CustomSchedule(self.d_model)  # Adaptive LR
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)  # As in paper
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

        # 4. Setup checkpoints
        ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, self.ckpt_path, max_to_keep=5)

        # Restore previous ckpt if exists and continue from there
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("INFO: Latest checkpoint restored.")

        # 5. Run the training step for each epoch-batch to process all training dataset
        print("INFO: Starting training ...")
        for epoch in range(self.epochs):
            start = time.time()

            self.train_loss.reset_states()  # Resets all of the metric state variables for the epoch
            self.train_accuracy.reset_states()

            # inp (b, n_inp): Tokenized Spanish sentences. All padded with zeros up to the longest sentence in batch
            # tar (b, n_tar): Tokenized English sentences. All padded with zeros up to the longest sentence in batch
            # Training dataset contains train_batches = (train_examples/batch_size)
            for (batch, (inp, tar)) in enumerate(train_batches):
                _train_step(inp, tar)

                if batch % 50 == 0:
                    print(
                        f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} '
                        f'Accuracy {self.train_accuracy.result():.4f}'
                    )

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

            print(f'Epoch {epoch + 1} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')
            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

        # 6. Save trained model for later use
        print("INFO: Saving trained model ...")
        translator = Translator(dataset_object.tokenizers, transformer)  # The inference model
        exporter = Exporter(translator)  # Wraps the inference model with a @tf.function decorator
        tf.saved_model.save(exporter, export_dir=self.save_path)


class Exporter(tf.Module):

    def __init__(self, translator: tf.Module) -> None:
        """
        A wrapper to efficiently save the inference model.

        :param translator: An inference model to be used when loading the saved trained model.

        :return: None
        """
        super(Exporter, self).__init__()
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence: Any) -> Any:
        """
        Inference method to translate a sentence from Spanish to English. When you save a tf.Module, any tf.Variable
        attributes, tf.function-decorated methods, and tf.Modules found via recursive traversal are saved. However,
        any Python attributes, functions, and data are lost. This means that when a tf.function is saved, no Python
        code is saved. The tf.function decorator works by tracing the Python code to generate a ConcreteFunction
        (a callable wrapper around tf.Graph). When saving a tf.function, you're really saving the tf.function's cache
        of ConcreteFunctions.

        The resulting SavedModel will only be servable with an input (sentence) with dtype string. When invoking a
        signature in an exported SavedModel, Tensor arguments are identified by name. These names will come from the
        Python function's argument names by default (in this case sentence).

        :param sentence: A sentence in Spanish to be translated into English with an inference model.

        :return: A tuple containing the prediction for the translated sentence, the tokens that make up the prediction,
                and the attention_weights from the prediction.
        """
        (result, tokens, attention_weights) = self.translator(sentence, max_length=100)

        # Only output sentence is returned. tf.function's non-strict execution never computes any unnecessary values
        return result


class Translator(tf.Module):

    def __init__(self, tokenizers: tf.Module, transformer: tf.keras.Model) -> None:
        """
        An object to perform inference from a trained model. The following steps are done for inference:
            1. Encode the input sentence using the Spanish tokenizer (tokenizers.es). This is the encoder input.
            2. The decoder input is initialized to the [START] token.
            3. Calculate the padding masks and the look ahead masks.
            4. The decoder outputs the predictions by looking at the encoder output and its own output (self-attention).
            5. Concatenate the predicted token to the decoder input and pass it to the decoder.
            6. In this approach, the decoder predicts the next token based on the previous tokens it predicted.

        :param tokenizers: Bert tokenizers for English and Spanish text.
        :param transformer: A Transformer model trained for Spanish-English translation.

        :return: None
        """

        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(self, sentence: Any, max_length: int = 20) -> tuple:
        """
        A callable that implements the full logic needed to make inference with a given trained model.

        :param sentence: A tensor containing strings in the source language (Spanish) to be used as input for the
                        Encoder for translation.
        :param max_length: The maximum length of the translated sentence. The prediction loop will break if the END
                        token is found before the max. length has been reached.

        :return: A tuple containing the prediction for the translated sentence, the tokens that make up the prediction,
                and the attention_weights from the prediction.
        """

        # Input sentence is Spanish, hence adding the START and END token
        assert isinstance(sentence, tf.Tensor)

        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.es.tokenize(sentence).to_tensor()

        encoder_input = sentence

        # As the target is english, the first token to the transformer should be the english start token.
        start_end = self.tokenizers.en.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a python list) so that the dynamic-loop can be traced by
        # `tf.function`
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())

            # More efficient for inference if only calculate last prediction when running in inference mode
            predictions, _ = self.transformer([encoder_input, output], training=False)

            # select the last token from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)  # Argmax over the vocab

            # concatentate the predicted_id to the output which is given to the decoder as its input.
            output_array = output_array.write(i+1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())  # (1, num_tokens_prediction)
        text = self.tokenizers.en.detokenize(output)[0]  # Shape: ()
        tokens = self.tokenizers.en.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were calculated on the last iteration of the
        # loop. So recalculate them outside the loop.
        _, attention_weights = self.transformer([encoder_input, output[:, :-1]], training=False)

        return text, tokens, attention_weights
