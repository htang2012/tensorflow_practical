#https://developers.googleblog.com/2017/12/creating-custom-estimators-in-tensorflow.html
#https://docs.w3cub.com/tensorflow~guide/get_started/datasets_quickstart

import tensorflow as tf
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

# dataset source
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_path = "/home/ytang/workspace/IRIS.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_path), 
                                            origin=train_dataset_url)
# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','species']
feature_names = column_names[:-1]
label_name = column_names[-1]

batch_size = 32 

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

features,labels = next(iter(train_dataset))

#display plot
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()
print(features, labels)


def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

train_dataset = train_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))

print(features, labels)



sys.exit("done")





