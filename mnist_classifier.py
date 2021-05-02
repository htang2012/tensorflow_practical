import tensorflow as tf
import numpy as np
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

train, test = tf.keras.datasets.mnist.load_data()

print(train)
print(test)


def input(dataset):
    return dataset.images, dataset.labels.astype(np.int32)

# Specify feature
feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]

# Build 2 layer DNN classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[256, 32],
    optimizer=tf.optimizers.Adam(1e-4),
    n_classes=10,
    dropout=0.1,
    model_dir="./tmp/mnist_model"
)

# Define the training inputs
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={"x": train[0]},
        y=train[1].astype(np.int32),
    num_epochs=None,
    batch_size=50,
    shuffle=True
)

classifier.train(input_fn=train_input_fn, steps=100000)

# Define the test inputs
test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": test[0]},
    y=test[1].astype(np.int32),
    num_epochs=1,
    shuffle=False
)

# Evaluate accuracy
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
