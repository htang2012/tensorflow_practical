#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Example of DNNClassifier for Iris plant dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pandas as pd
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=5000, type=int,
                    help='number of training steps')

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Sentosa', 'Versicolor', 'Virginica']

def load_data(label_name='Species'):
    """Parses the csv file in TRAIN_URL and TEST_URL."""

    # Create a local copy of the training set.
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],
                                         origin=TRAIN_URL)
    # train_path now holds the pathname: ~/.keras/datasets/iris_training.csv

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0  # ignore the first row of the CSV file.
                       )
    # train now holds a pandas DataFrame, which is data structure
    # analogous to a table.

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop(label_name)

    # Apply the preceding logic to the test set.
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)



# Call load_data() to parse the CSV file.
(train_feature, train_label), (test_feature, test_label) = load_data()
print(train_feature, train_label)
print(test_feature, test_label)

feature_columns = [ ]
for key in train_feature.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))


def train_input_fn(features, labels=None, training = True, batch_size=32):
    """An input function for evaluation or prediction"""
    # Convert inputs to a tf.dataset object.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)



def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'])
    for units in params.get('hidden_units', [10, 20, 10]):
        net = tf.compat.v1.layers.dense(net, units=units, activation=tf.nn.relu)
        net = tf.compat.v1.layers.dropout(net, rate=0.1,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    # Compute logits (1 per class).
    logits = tf.compat.v1.layers.dense(net, params['n_classes'], activation=None)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Convert the labels to a one-hot tensor of shape (length of features, 3)
    # and with a on-value of 1 for each one-hot vector of length 3.
    onehot_labels = tf.one_hot(labels, 3, 1, 0)
    # Compute loss.
    loss = tf.compat.v1.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.compat.v1.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.1)

    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = load_data()
    train_x = dict(train_x)
    test_x = dict(test_x)

    # Feature columns describe the input: all columns are numeric.
    feature_columns = [tf.feature_column.numeric_column(col_name)
                       for col_name in CSV_COLUMN_NAMES[:-1]]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': feature_columns,
            'hidden_units': [10, 20, 10],
            'n_classes': 3,
        },
        model_dir = "./convent_model")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.compat.v1.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    
    classifier.train(input_fn=lambda:train_input_fn(train_feature, train_label, training=True), hooks=[logging_hook], steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(input_fn=lambda:train_input_fn(test_feature, test_label, training=False))
    
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }


    def input_fn(features, batch_size=256):
       """An input function for prediction."""
       # Convert the inputs to a Dataset without labels.
       return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    predictions = classifier.predict(
    input_fn=lambda: input_fn(predict_x))
    for pred_dict, expec in zip(predictions, expected):
         class_id = pred_dict['class_ids'][0]
         probability = pred_dict['probabilities'][class_id]

         print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
             SPECIES[class_id], 100 * probability, expec))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


 

if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    tf.compat.v1.app.run(main)
