import tensorflow as tf
import numpy as np


x_feature = tf.feature_column.numeric_column('f1')

# Training
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x = {"f1": np.array([1., 2., 3., 4.])},      # Input features
      y = np.array([1.5, 3.5, 5.5, 7.5]),         # true labels
      batch_size=2,
      num_epochs=None,                            # Supply unlimited epochs of data
      shuffle=True)

# Testing
test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x = {"f1": np.array([5., 6., 7.])},
      y = np.array([9.5, 11.5, 13.5]),
      num_epochs=1,
      shuffle=False)

# Prediction
samples = np.array([8., 9.])
predict_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={"f1": samples},
      num_epochs=1,
      shuffle=False)

regressor = tf.estimator.LinearRegressor(
    feature_columns=[x_feature],
    model_dir='./output'
)

regressor.train(input_fn=train_input_fn, steps=2500)

average_loss = regressor.evaluate(input_fn=test_input_fn)["average_loss"]
print(f"Average loss in testing: {average_loss:.4f}")
# Average loss in testing: 0.0000

predictions = list(regressor.predict(input_fn=predict_input_fn))

for input, p in zip(samples, predictions):
    v  = p["predictions"][0]
    print(f"{input} -> {v:.4f}")
# 8.0 -> 15.4991
# 9.0 -> 17.4990
