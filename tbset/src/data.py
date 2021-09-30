import os
import re
import tensorflow as tf
import tensorflow_text as text
import tensorflow_datasets as tfds

from typing import Any
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab_maker

# .....................................................................................................................


class EnEsTokenizer(tf.Module):

    def __init__(self, reserved_tokens: list, vocab_path: str) -> None:
        """
        An object to perform tokenization, detokenization, return the tokens from a sentence, and other helper
        functions for source and targe languages.

        :param reserved_tokens: A list of special tokens including the START, END and PADDING tokens.
        :param vocab_path: The folder where to store/read-from the vocabulary files for source/target languages.

        :return: None
        """

        super(EnEsTokenizer, self).__init__()

        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)

        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)  # Tell TF to include vocab file when tf.saved_model.save()
        self._start = tf.argmax(tf.constant(reserved_tokens) == "[START]")  # index of START
        self._end = tf.argmax(tf.constant(reserved_tokens) == "[END]")  # index of END

        # rstrip will remove \n on each line
        with open(vocab_path, "r") as file:
            vocab = [token.rstrip() for token in file.readlines()]
        self.vocab = tf.Variable(vocab)

        # Define graph templates (signatures) for below methods. For tokenize on decorator
        self.detokenize.get_concrete_function(tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    def _add_start_end(self, ragged_tensor: Any) -> Any:
        """
        Adds the START and END tokens at the beginning and end of provided tensor.

        :param ragged_tensor: A tensor with variable shape where we want to add the START and END tokens.

        :return: A ragged tensor containing the START and END token. These tokens are added before padding the
                sequences.
        """

        count = ragged_tensor.bounding_shape()[0]  # Number of rows
        starts = tf.fill([count, 1], self._start)  # col vector: count-times the START index
        ends = tf.fill([count, 1], self._end)

        # Add START and END as column value on each item
        return tf.concat([starts, ragged_tensor, ends], axis=1)

    @staticmethod
    def _cleanup_text(reserved_words: list, words: Any) -> Any:
        """
        Used during detokenization. This method removes reserved tokens and join tensor of words into sentences.

        :param reserved_words:
        """

        # Remove special tokens except "[UNK]"
        bad_tokens = [re.escape(token) for token in reserved_words if token != "[UNK]"]
        regex = "|".join(bad_tokens)
        bad_cells = tf.strings.regex_full_match(words, regex)
        result = tf.ragged.boolean_mask(words, ~bad_cells)

        # Join clean words into sentences
        result = tf.strings.reduce_join(result, separator=" ", axis=-1)

        return result

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def tokenize(self, strings: Any) -> Any:
        """
        Uses the text.BertTokenizer method to convert string inputs into numeric representations. For example:

        'but what if it were active ?' -> [2, 87, 90, 107, 76, 129, 1852, 30, 3]

        The START token has index 2 in the vocabularies. The END token, 3.

        :param strings: A Tensor or RaggedTensor of untokenized UTF-8 strings.
        :return: A RaggedTensor with words and wordpieces following the WordPiece algorithm. The START and END tokens
                have been added.
        """
        results = self.tokenizer.tokenize(strings)  # BertTokenizer returns (batch, word, word-piece)
        results = results.merge_dims(-2, -1)  # (batch, word, word-piece) -> (batch, tokens)
        results = self._add_start_end(results)

        return results

    @tf.function
    def detokenize(self, tokenized: Any) -> Any:
        """
        Converts the numeric representation of a batch of examples into a batch of strings.

        :param tokenized: A tensor containing a batch of numeric representation of examples.

        :return: A tensor containing the equivalent words and wordpieces of numeric values.
        """

        words = self.tokenizer.detokenize(tokenized)

        return self._cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids: Any) -> Any:
        """
        Use the vocab to return strings (wordpieces included) from token ids.

        :param token_ids: Numeric representation of words as indexes of the correpondind vocabulary.

        :return: A tensor containing the corresponding words or wordpieces from the vocabulary.
        """

        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self) -> Any:
        """
        Obtains the size of given vocabulary.

        :return: The size of current vocabulary.
        """

        return tf.shape(self.vocab)[0]  # Resist the temptation of using len!

    @tf.function
    def get_vocab_path(self) -> Any:
        """
        Obtains the path where current vocabulary is stored.

        :return: The path to current vocabulary file.
        """

        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self) -> Any:
        """
        Obtains the list of reserved tokens used during the tokenization process.

        :return: The list of reserved tokens used for tokenization.
        """
        return tf.constant(self._reserved_tokens)


class Dataset:

    def __init__(self, dwn_destination: str, vocab_path: str, buffer_size: int, batch_size: int, vocab_size: int,
                 num_examples: int) -> None:
        """
        An object to load, prepare, and provide Dataset objects for training. It specifically uses the OPUS corpora
        to retrieve a fixed number of English/Spanish training examples. The method `workout_dataset`:
        - Shuffles each dataset
        - Creates batches of examples to be consumed at each training step
        - Tokenizes each batch
        - Prefetchs the batches

        :param dwn_destination: The folder where to save the training examples.
        :param vocab_path: The path where to store the generated vocabulary files.
        :param buffer_size: The number of items to consider when shuffling the training data.
        :param batch_size: The number of training example to include on each training set.
        :param vocab_size: The top tokens to consider when performing tokenization.
        :param num_examples: The number of training examples to retrieve from the OPUS corpora.
        """

        super(Dataset, self).__init__()

        self.dwn_destination = dwn_destination
        self.vocab_path = vocab_path

        # Dataset hyperparameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_examples = num_examples

        self.tokenizers = tf.Module()

    @staticmethod
    def _get_opus_datasets(dwn_path: str, num_items: int):
        """
        A function to download an ES/EN dataset and return non-shuffled training, test, and validation Datasets objects

        :param dwn_path: The folder where to save the training examples.
        :param num_items: The number of training examples to retrieve from the OPUS corpora.

        :return: A tuple containing the training dataset splitted into training, test, and validations subsets.
        """

        build_config = tfds.translate.opus.OpusConfig(
            version=tfds.core.Version("0.1.0"),
            language_pair=("es", "en"),
            subsets=["OpenSubtitles"]  # TODO: Add Books dataset to improve performance
        )
        # OPUS only provides one split: "train"
        builder = tfds.builder("opus", config=build_config)

        # By default stored in /root/tensorflow_datasets
        dwn_config = tfds.download.DownloadConfig(
            extract_dir=dwn_path,  # store extracted files here
            max_examples_per_split=num_items  # "train" split only
        )

        builder.download_and_prepare(
            download_dir=dwn_path,  # Use the files from here instead downloading again
            download_config=dwn_config
        )

        # Construct the DS with all downloaded data: 80-10-10
        raw_trn_ds, raw_tst_ds, raw_val_ds = builder.as_dataset(split=["train[:80%]", "train[80%:90%]", "train[90%:]"])

        return raw_trn_ds, raw_tst_ds, raw_val_ds

    def _tokenize_pairs(self, pair: dict) -> tuple:
        """
        Uses the Bert tokenizer to convert a batch of strings (Spanish and English) into numerical equivalences. The
        batches after tokenization are ragged tensors (variable size). After applying the `to_tensor` method all items
        in the batch will be padded with 0 (PAD token) until they have the same length of the longest sentence in the
        batch. This type of tokenization considers subword tokens.

        :param pair: Dictionary containing a batch-size set of sentences in Spanish and their equivalent translation in
                    English. After the batch has been tokenized it is converted to a dense tensor with the `to_tensor`
                    method. This method pads the items to the longest item length in the batch.

        :return: A tuple of dense tensors containing the Spanish and English tokenized sentences in the "pair" batch.
        """

        en = pair["en"]
        es = pair["es"]

        es = self.tokenizers.es.tokenize(es)
        en = self.tokenizers.en.tokenize(en)

        # From ragged to dense padding with zeros up to the longest item length of the batch
        return es.to_tensor(), en.to_tensor()

    def workout_datasets(self):
        """
        A dataset pipeline that load the training dataset, create the vocabularies (if not exist) for the source and
        target languages, create the tokenizers for both languages, and prepare the datasets for consumption in the
        training process. This involves next actions:
        - Shuffle each dataset
        - Create batches of examples to be consumed at each training step
        - Tokenize each batch
        - Prefetch the batches

        :return: A tuple containing three prefetch Dataset objetcs to be used as training, test and validation sets.
                Each dataset contains batches of training examples. Each item in the datasets is a tuple of input and
                target batches of sentences.
        """

        # 1. Load the datasets. Each of these datasets contain strings
        (raw_trn_ds,
         raw_tst_ds,
         raw_val_ds) = self._get_opus_datasets(self.dwn_destination, self.num_examples)

        # 2. Create the tokenizers starting with the vocabularies
        reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

        if os.path.isfile(f"{self.vocab_path}/es_vocab.txt") and os.path.isfile(f"{self.vocab_path}/en_vocab.txt"):
            pass
        else:  # No vocab files found -> create them
            bert_tokenizer_params = dict(lower_case=True)  # Otherwise Que != que

            bert_vocab_maker_args = dict(
                vocab_size=self.vocab_size,
                reserved_tokens=reserved_tokens,
                bert_tokenizer_params=bert_tokenizer_params,
                learn_params={}
            )

            en_trn_ds = raw_trn_ds.map(lambda x: x["en"])
            es_trn_ds = raw_trn_ds.map(lambda x: x["es"])

            # bert_vocab_from_dataset follows the Wordpiece algorithm
            es_vocab = bert_vocab_maker.bert_vocab_from_dataset(es_trn_ds.batch(1000).prefetch(2),
                                                                **bert_vocab_maker_args)
            en_vocab = bert_vocab_maker.bert_vocab_from_dataset(en_trn_ds.batch(1000).prefetch(2),
                                                                **bert_vocab_maker_args)

            with open(f"{self.vocab_path}/es_vocab.txt", "w") as file:
                file.writelines("%s\n" % token for token in es_vocab)

            with open(f"{self.vocab_path}/en_vocab.txt", "w") as file:
                file.writelines("%s\n" % token for token in en_vocab)

        # Create tokenizers
        self.tokenizers.es = EnEsTokenizer(reserved_tokens, f"{self.vocab_path}/es_vocab.txt")
        self.tokenizers.en = EnEsTokenizer(reserved_tokens, f"{self.vocab_path}/en_vocab.txt")

        # 3. Prepare datasets for training. map function will tokenize all the batches
        train_batches = raw_trn_ds.cache()\
            .shuffle(self.buffer_size)\
            .batch(self.batch_size)\
            .map(self._tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)\
            .prefetch(tf.data.AUTOTUNE)

        tst_batches = raw_tst_ds.cache()\
            .shuffle(self.buffer_size)\
            .batch(self.batch_size)\
            .map(self._tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)\
            .prefetch(tf.data.AUTOTUNE)

        val_batches = raw_val_ds.cache()\
            .shuffle(self.buffer_size)\
            .batch(self.batch_size)\
            .map(self._tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)\
            .prefetch(tf.data.AUTOTUNE)

        # I'm not using other datasets yet
        return train_batches, None, None
