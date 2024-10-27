
# Load model
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: show_result_task.py
@Author: YangChen
@Date: 2024/1/6
"""
from ast import List
import json
import os.path
import re
import shutil
from typing import Any, Dict
from unittest import result

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.math_ops import Sum
import torch
from matplotlib import animation
from matplotlib import patches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos.scenario_pb2 import Scenario
from waymo_open_dataset.utils.sim_agents import visualizations, submission_specs
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils import trajectory_utils
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

from tasks.show_result_task import ShowResultsTask
from utils.data_utils import DataUtil
from utils.eval_utils import EvalUtil
from utils.map_utils import MapUtil
from common import TaskType, LoadConfigResultDate
from net_works import BackBone
from tasks import BaseTask
from utils import DataUtil, MathUtil, MapUtil
from PIL import Image, ImageSequence
from torch.utils.tensorboard import SummaryWriter
COND_RESULT_DIR = r"/home/k.lotfy/WcDT/cond_show_results"
DATA_SET_PATH = r"/home/k.lotfy/data/womd-mini/waymo-micro/training/training.tfrecord-00000-of-01000"
VALIDATION_DATA_SET_PATH = r"/home/k.lotfy/data/womd-mini/waymo-micro/validation/validation.tfrecord-00000-of-00150"
MODEL_PATH = r"/home/k.lotfy/WcDT/investigate/before_metrics/output_no_teacher_forcing_10x_run/model/20241021-16-04-08_75742/epoch_90_batch_num_0_model.pth"
MODELS_DIR = r"/home/k.lotfy/WcDT/output/model/20241018-00-09-19_60149" # This does evaluation on differetn iterations of the smae model at different epochs.

class ConditionedShowResultsTask(BaseTask):
    TASK_TYPE = TaskType.SHOW_RESULTS
    CMAP = LinearSegmentedColormap.from_list(
        'my_cmap',
        [np.array([0., 232., 157.]) / 255, np.array([0., 120., 255.]) / 255],
        100
    )
    COLOR_DICT = {
        0: np.array([0., 120., 255.]) / 255, # blue
        1: np.array([0., 232., 157.]) / 255, # Green-blueish
        2: np.array([255., 205., 85.]) / 255, # Orange
        3: np.array([244., 175., 145.]) / 255, # Redish orange
        4: np.array([145., 80., 200.]) / 255, # Pruple
        5: np.array([0., 51., 102.]) / 255, # Dark blue
        6: np.array([1, 0, 0]), # RED
        7: np.array([0, 1, 0]), # GREEN
    }


    def __init__(self):
        self.task_type = self.TASK_TYPE
        self.cmap = self.CMAP
        self.color_dict = self.COLOR_DICT
    @staticmethod
    def load_pretrain_model(result_info: LoadConfigResultDate) -> BackBone:
        betas = MathUtil.generate_linear_schedule(result_info.train_model_config.time_steps)

        model = BackBone(betas, 
                         diffusion_type=result_info.train_model_config.diffusion_type, 
                         teacher_forcing=False # To test must use previous model output
                 ).eval()
        device = torch.device("cpu")
        pretrained_dict = torch.load(MODEL_PATH, map_location=device)
        model_dict = model.state_dict()
        # 模型参数赋值
        new_model_dict = dict()
        for key in model_dict.keys():
            if ("module." + key) in pretrained_dict:
                new_model_dict[key] = pretrained_dict["module." + key]
            elif key in pretrained_dict:
                new_model_dict[key] = pretrained_dict[key]
            else:
                print("key: ", key, ", not in pretrained")
        model.load_state_dict(new_model_dict)
        print("load_pretrain_model success")
        return model

    def show_result(self, result_info: LoadConfigResultDate, max_dataset=100):
        model: BackBone = self.load_pretrain_model(result_info)
        match_filenames = tf.io.matching_files([DATA_SET_PATH])
        dataset = tf.data.TFRecordDataset(match_filenames, name="train_data").take(max_dataset)
        dataset_iterator = dataset.as_numpy_iterator()
        for index, scenario_bytes in enumerate(dataset_iterator):
            scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
            data_dict = DataUtil.transform_data_to_input(scenario, result_info, evaluation_data=True)
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(torch.float32).unsqueeze(dim=0)
                
            
            def f_x(model_output):
                # Make the second Predicted_agent at timestep t=90 be at 17, -75
                value = model_output[1][79][:2]
                return torch.nn.L1Loss()(value, torch.tensor([17, -75], dtype=torch.float32))
            # predict_traj, _ = model.sample_conditioned(data_dict, f_x)
            for idx, (predict_traj, _) in enumerate(model.sample_conditioned(data_dict, f_x)):
                predicted_traj_mask = data_dict['predicted_traj_mask'][0]
                predicted_future_traj = data_dict['predicted_future_traj'][0]
                predicted_his_traj = data_dict['predicted_his_traj'][0]
                predicted_num = 0
                for i in range(predicted_traj_mask.shape[0]):
                    if int(predicted_traj_mask[i]) == 1:
                        predicted_num += 1
                generate_traj = predict_traj[:predicted_num]
                predicted_future_traj = predicted_future_traj[:predicted_num]
                predicted_his_traj = predicted_his_traj[:predicted_num]
                real_traj = torch.cat((predicted_his_traj, predicted_future_traj), dim=1)[:, :, :2].detach().numpy()
                real_yaw = torch.cat((predicted_his_traj, predicted_future_traj), dim=1)[:, :, 2].detach().numpy()
                model_output = torch.cat((predicted_his_traj, generate_traj), dim=1)[:, :, :2].detach().numpy()
                model_yaw = torch.cat((predicted_his_traj, generate_traj), dim=1)[:, :, 2].detach().numpy()

                image_path = os.path.join(COND_RESULT_DIR, f"{idx}-{index}_model_output.gif")
                ShowResultsTask.draw_gif(predicted_num, model_output, model_yaw, data_dict, scenario, image_path)




    def execute(self, result_info: LoadConfigResultDate):
        if os.path.exists(COND_RESULT_DIR):
            shutil.rmtree(COND_RESULT_DIR)
        os.makedirs(COND_RESULT_DIR, exist_ok=True)
        NUM_OF_SAMPLES = 100
        self.show_result(result_info, max_dataset=NUM_OF_SAMPLES)

# Load 1 example
# Create sampling that goes thorugh the entire process. then 