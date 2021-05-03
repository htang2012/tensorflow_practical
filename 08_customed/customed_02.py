#https://jhui.github.io/2017/03/14/TensorFlow-Estimator/

import numpy as np
import tensorflow as tf



def cnn_model_fn(features, labels, mode):
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  conv1 = tf.compat.v1.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
           eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  mnist = tf.keras.datasets.mnist.load_data()
  (x_train,y_train),(x_test,y_test) = mnist
  train_data = x_train
  train_labels = np.asarray(y_train, dtype=np.int32)

  eval_data = x_test
  eval_labels = np.asarray(y_test, dtype=np.int32)

  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, 
      model_dir="/tmp/mnist_convnet_model")

  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.compat.v1.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"x": train_data}, 
        y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
  mnist_classifier.train(input_fn=train_input_fn, 
        steps=20000, hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, 
        y=eval_labels, num_epochs=1, shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  tf.compat.v1.app.run()
