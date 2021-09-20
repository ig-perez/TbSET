import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

# TODO: Review this
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  # Cast will produce 1 for those values different than zero
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


# TODO: Review this
def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)

  return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

# Adam with custom LR schedule according to the Transformer paper
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)  # a^-0.5 == 1/sqrt(a) == rsqrt(a)
    arg2 = step * self.warmup_steps**-1.5

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)