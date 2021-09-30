import tensorflow as tf

from typing import Any

# .....................................................................................................................

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")


def loss_function(real: Any, pred: Any) -> Any:
    """
    Since the target sequences are padded, it is important to apply a padding mask when calculating the loss.

    :param real: A tensor batch of (b, n_tar) containing the ground truths for each token in the translated sentence.
    :param pred: A tensor batch of (b, n_tar, vocab_size) containing the predictions for each token in the dictionary
                correponding to each token in the source sentence.
    """

    mask = tf.math.logical_not(tf.math.equal(real, 0))  # (b, n_tar)
    loss_ = loss_object(real, pred)  # (b, n_tar)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask  # Avoid the PAD tokens

    # Average loss of current batch
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real: Any, pred: Any) -> Any:
    """
    :param real: A tensor batch of (b, n_tar) containing the ground truths for each token in the translated sentence.
    :param pred: A tensor batch of (b, n_tar, vocab_size) containing the predictions for each token in the dictionary
                correponding to each token in the source sentence.
    """

    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

# Adam with custom LR schedule according to the Transformer paper
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model: int, warmup_steps: int = 4000) -> None:
        """
        A custom LR scheduler following the 'Attention Is All You Need' paper's instructions for the optimizer setup.
        After increasing the LR linearly for warmup_steps, it is decreased proportionally to the inverse square root of
        the step number.

        :param d_model: Dimensionality of the internal vector representation of the Transformer.
        :param warmup_steps: The first training steps where the LR is linearly increased.
        """
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step: Any) -> Any:
        """
        Implementation of the schedule object to modulare LR changes during training.
        :param step: A scalar integer tensor, the current training step count.

        :return: A tensor containing the LR to use.
        """
        arg1 = tf.math.rsqrt(step)  # a^-0.5 == 1/sqrt(a) == rsqrt(a)
        arg2 = step * self.warmup_steps**-1.5

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)