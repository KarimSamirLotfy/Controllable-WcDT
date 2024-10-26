
# Imports
import os
import tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm

from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2

from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.utils.sim_agents import test_utils as sim_agents_test_utils
from waymo_open_dataset.utils.sim_agents import visualizations
from waymo_open_dataset.utils import trajectory_utils

# Please edit.

# Replace this path with your own tfrecords./
# This tutorial is based on using data in the Scenario proto format directly,
# so choose the correct dataset version.
DATASET_FOLDER = '/home/data/womd/scenario/training'

TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training.*')
# Define the dataset from the TFRecords.
filenames = tf.io.matching_files(TRAIN_FILES)
dataset = tf.data.TFRecordDataset(filenames)
# Since these are raw Scenario protos, we need to parse them in eager mode.
dataset_iterator = dataset.as_numpy_iterator()

# get the number of scenarios in the dataset
num_scenarios = 0
for scenario in dataset_iterator:
    num_scenarios += 1
print(f'Number of scenarios in the dataset: {num_scenarios}')