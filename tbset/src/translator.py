import time
import tensorflow as tf

from tbset.src.data import Dataset
from tbset.src.model import Transformer
from tbset.src.optimization import CustomSchedule, loss_function, accuracy_function


class TranslatorExporter(tf.Module):

    def __init__(self, translator):
        super(TranslatorExporter, self).__init__()
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        (result, tokens, attention_weights) = self.translator(sentence, max_length=100)

        return result

class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(self, sentence, max_length=20):
        # input sentence is spanish, hence adding the start and end token
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
          sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.es.tokenize(sentence).to_tensor()

        encoder_input = sentence

        # as the target is english, the first token to the transformer should be the
        # english start token.
        start_end = self.tokenizers.en.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
          output = tf.transpose(output_array.stack())
          predictions, _ = self.transformer([encoder_input, output], training=False)

          # select the last token from the seq_len dimension
          predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

          predicted_id = tf.argmax(predictions, axis=-1)

          # concatentate the predicted_id to the output which is given to the decoder
          # as its input.
          output_array = output_array.write(i+1, predicted_id[0])

          if predicted_id == end:
            break

        output = tf.transpose(output_array.stack())
        # output.shape (1, tokens)
        text = self.tokenizers.en.detokenize(output)[0]  # shape: ()

        tokens = self.tokenizers.en.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        _, attention_weights = self.transformer([encoder_input, output[:,:-1]], training=False)

        return text, tokens, attention_weights

def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')

class Translator:

    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate, ckpt_path, save_path, dwn_destination, vocab_path, buffer_size, batch_size):
        super(Translator, self).__init__()

        # Hyperparameters for training
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.ckpt_path = ckpt_path
        self.save_path = save_path
        # Hyperparameters for dataset workout
        self.dwn_destination = dwn_destination
        self.vocab_path = vocab_path
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def train(self):
        """
        Spanish is used as the input language and English is the target language.

        :return:
        """

        # .............................................................................................................

        # Signatures for each training step. The idea is to avoid retracing
        @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64)])
        def _train_step(inp, tar):
            """
            The translated sentence (tar) is divided into tar_inp and tar_real. tar_inp
            is passed as an input to the decoder. tar_real is that same input shifted by
            1: At each location in tar_inp, tar_real contains the next token that should
            be predicted.

            Parameters
            ----------
            - `tar`: Shape (b, n)
            """

            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            with tf.GradientTape() as tape:
                predictions, _ = transformer([inp, tar_inp], training=True)
                loss = loss_function(tar_real, predictions)

                gradients = tape.gradient(loss, transformer.trainable_variables)
                optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

                train_loss(loss)
                train_accuracy(accuracy_function(tar_real, predictions))

        # .............................................................................................................

        # 1. Obtain batched training dataset
        dataset_object = Dataset(dwn_destination=self.dwn_destination, vocab_path=self.vocab_path, buffer_size=self.buffer_size, batch_size=self.batch_size)
        train_batches, tst_batches, val_batches = dataset_object.workout_datasets()

        # 2. Instantiate the model
        transformer = Transformer(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            input_vocab_size=dataset_object.tokenizers.es.get_vocab_size().numpy(),  # Ojo
            target_vocab_size=dataset_object.tokenizers.en.get_vocab_size().numpy(),
            pe_input=1000,  # TODO: What is this?
            pe_target=1000,  # TODO: What is this?
            rate=self.dropout_rate
        )

        # 3. Optimization, loss, and accuracy
        learning_rate = CustomSchedule(self.d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

        # 4. Setup checkpoints
        ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, self.ckpt_path, max_to_keep=5)

        # Restore previous ckpt if exists and continue from there
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("INFO: Latest checkpoint restored.")

        # 5. Run the training step for each epoch to process all training dataset
        EPOCHS = 20

        for epoch in range(EPOCHS):
            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            # inp: Spanish, tar:English
            for (batch, (inp, tar)) in enumerate(train_batches):
                _train_step(inp, tar)

                if batch % 50 == 0:
                    print(
                        f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

            # After processing 5 epochs, save the checkpoint
            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

            print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

        # 6. Save trained model for later use
        translator = Translator(dataset_object.tokenizers, transformer)
        exported_translator = TranslatorExporter(translator)
        tf.saved_model.save(exported_translator, export_dir=self.save_path)

    def translate(self, oracion):
        reloaded = tf.saved_model.load(self.save_path)
        translation = reloaded(oracion).numpy()

        return translation

