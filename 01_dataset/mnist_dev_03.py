#https://developers.googleblog.com/2017/12/creating-custom-estimators-in-tensorflow.html
#https://docs.w3cub.com/tensorflow~guide/get_started/datasets_quickstart

import tensorflow as tf
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

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
# test
#

# Call load_data() to parse the CSV file.
(train_feature, train_label), (test_feature, test_label) = load_data()
print(train_feature, train_label)
print(test_feature, test_label)

feature_columns = []
for key in train_feature.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

print(str(feature_columns))



sys.exit("done")





