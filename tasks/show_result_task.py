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

from utils.data_utils import DataUtil
from utils.eval_utils import EvalUtil
from utils.map_utils import MapUtil
from common import TaskType, LoadConfigResultDate
from net_works import BackBone
from tasks import BaseTask
from utils import DataUtil, MathUtil, MapUtil
from PIL import Image, ImageSequence
from torch.utils.tensorboard import SummaryWriter
RESULT_DIR = r"/home/k.lotfy/WcDT/show_results"
DATA_SET_PATH = r"/home/k.lotfy/data/womd-mini/waymo-micro/training/training.tfrecord-00000-of-01000"
VALIDATION_DATA_SET_PATH = r"/home/k.lotfy/data/womd-mini/waymo-micro/validation/validation.tfrecord-00000-of-00150"
MODEL_PATH = r"/home/k.lotfy/WcDT/investigate/baseline-10x-run/output_big_run_10x/model/20241018-00-09-19_60149/epoch_180_batch_num_0_model.pth"
MODELS_DIR = r"/home/k.lotfy/WcDT/output/model/20241018-00-09-19_60149" # This does evaluation on differetn iterations of the smae model at different epochs.

class ShowResultsTask(BaseTask):
    TASK_TYPE = TaskType.SHOW_RESULTS
    CMAP = LinearSegmentedColormap.from_list(
        'my_cmap',
        [np.array([0., 232., 157.]) / 255, np.array([0., 120., 255.]) / 255],
        100
    )
    COLOR_DICT = {
        0: np.array([0., 120., 255.]) / 255,
        1: np.array([0., 232., 157.]) / 255,
        2: np.array([255., 205., 85.]) / 255,
        3: np.array([244., 175., 145.]) / 255,
        4: np.array([145., 80., 200.]) / 255,
        5: np.array([0., 51., 102.]) / 255,
        6: np.array([1, 0, 0]),
        7: np.array([0, 1, 0]),
    }

    def __init__(self):
        self.task_type = self.TASK_TYPE
        self.cmap = self.CMAP
        self.color_dict = self.COLOR_DICT

    def execute(self, result_info: LoadConfigResultDate):
        if os.path.exists(RESULT_DIR):
            shutil.rmtree(RESULT_DIR)
        os.makedirs(RESULT_DIR, exist_ok=True)
        self.show_result(result_info)
        # Show Metrics for Model progress
        # ShowResultsTask.evaluate_all_models(MODELS_DIR, result_info)

    @staticmethod
    def evaluate_all_models(model_dir, result_info):
        epoch_pattern = re.compile(r'epoch_(\d+)_batch_num_0_model\.pth')
        for file_name in os.listdir(model_dir):
            match = epoch_pattern.match(file_name)
            if match:
                epoch_num = int(match.group(1))
                model_path = os.path.join(model_dir, file_name)
                MODEL_PATH = model_path
                model = ShowResultsTask.load_pretrain_model(result_info)
                evaluation_result = ShowResultsTask.evaluate_metrics_validation(model, result_info, epoch_num, number_of_scenarios=3, print_verbose_comments=False)
                print(f"Evaluation result for {file_name} (Epoch {epoch_num}): \n{evaluation_result}")
                # in showresults folder. create a json file for each and dump it
                with open(os.path.join(RESULT_DIR, f"{epoch_num}_evaluation.json"), 'w') as f:
                    f.write(json.dumps(evaluation_result, indent=4))


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

    def show_result(self, result_info: LoadConfigResultDate):
        model = self.load_pretrain_model(result_info)
        match_filenames = tf.io.matching_files([DATA_SET_PATH])
        dataset = tf.data.TFRecordDataset(match_filenames, name="train_data").take(100)
        dataset_iterator = dataset.as_numpy_iterator()
        for index, scenario_bytes in enumerate(dataset_iterator):
            scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
            data_dict = DataUtil.transform_data_to_input(scenario, result_info)
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(torch.float32).unsqueeze(dim=0)

            predict_traj = model(data_dict)[-1]
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
            # 可视化输入
            image_path = os.path.join(RESULT_DIR, f"{index}_input.png")
            self.draw_input(scenario, image_path)
            # 可视化ground truth
            # image_path = os.path.join(RESULT_DIR, f"{index}_ground_truth.png")
            # self.draw_scene(predicted_num, real_traj, data_dict, scenario, image_path)
            # # 可视化model output
            # image_path = os.path.join(RESULT_DIR, f"{index}_model_output.png")
            # self.draw_scene(predicted_num, model_output, data_dict, scenario, image_path)
            # 可视化ground truth
            # image_path = os.path.join(RESULT_DIR, f"{index}_ground_truth.png")
            # self.draw_scene(predicted_num, real_traj, data_dict, scenario, image_path)
            # # 可视化model output
            # image_path = os.path.join(RESULT_DIR, f"{index}_model_output.png")
            # self.draw_scene(predicted_num, model_output, data_dict, scenario, image_path)
            # GIFS
            image_path = os.path.join(RESULT_DIR, f"{index}_ground_truth.gif")
            self.draw_gif(predicted_num, real_traj, real_yaw, data_dict, scenario, image_path)
            image_path = os.path.join(RESULT_DIR, f"{index}_model_output.gif")
            self.draw_gif(predicted_num, model_output, model_yaw, data_dict, scenario, image_path)
            image_path = os.path.join(RESULT_DIR, f"{index}_scenario.gif")
            self.draw_gif_from_scenario(predicted_num,scenario, submission_specs, image_path)

            

    @staticmethod
    def evaluate_metrics_validation(model, result_info, epoch_num, number_of_scenarios=3, print_verbose_comments=True): 
        vprint = print if print_verbose_comments else lambda arg: None
        # get the device to use from result_info
        device = next(model.parameters()).device

        ### READ VALIDATION DATA
        match_filenames = tf.io.matching_files([VALIDATION_DATA_SET_PATH])
        dataset = tf.data.TFRecordDataset(match_filenames, name="train_data")
        dataset = dataset.shuffle(buffer_size=number_of_scenarios*5)  # Shuffle with a buffer size of 1000
        dataset = dataset.take(number_of_scenarios)  # Then take the first 100 elements

        dataset_iterator = dataset.as_numpy_iterator()
        for index, scenario_bytes in enumerate(dataset_iterator):
            scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
            data_dict = DataUtil.transform_data_to_input(scenario, result_info)
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(torch.float32).unsqueeze(dim=0).to(device)

            ### Data prepratation done

            ### Create logged trajectories ###
            # To load the data, we create a simple tensorized version of the object tracks.
            logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(scenario)
            # Using `ObjectTrajectories` we can select just the objects that we need to
            # simulate and remove the "future" part of the Scenario.
            vprint(f'Original shape of tensors inside trajectories: {logged_trajectories.valid.shape} (n_objects, n_steps)')
            logged_trajectories = logged_trajectories.gather_objects_by_id(
                tf.convert_to_tensor(submission_specs.get_sim_agent_ids(scenario)))
            logged_trajectories = logged_trajectories.slice_time(
                start_index=0, end_index=submission_specs.CURRENT_TIME_INDEX + 1)
            vprint(f'Modified shape of tensors inside trajectories: {logged_trajectories.valid.shape} (n_objects, n_steps)')
            # We can verify that all of these objects are valid at the last step.
            vprint(f'Are all agents valid: {tf.reduce_all(logged_trajectories.valid[:, -1]).numpy()}')

            all_logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(scenario)
            all_logged_trajectories = all_logged_trajectories.slice_time(
                start_index=0, end_index=submission_specs.N_FULL_SCENARIO_STEPS + 1)
            predicted_obs_id = submission_specs.get_evaluation_sim_agent_ids(scenario)

            ### MODEL INFERENCE ###
            predicted_obs_traj, _confidence = model.predict(data_dict)
            predicted_obs_traj = predicted_obs_traj.cpu().detach().numpy()

            ### PUT into siumlation format of (x, y, z, heading) ### Do this via extrapolation
            predicted_obs_id_traj = {obs_id: predicted_obs_traj[index] for index, obs_id in enumerate(data_dict['predicted_obs_index'])}
            # 自车在当前时刻的位置 The position of the vehicle at the current moment
            curr_loc = data_dict['curr_loc']
            simulated_states = list()
            for index, obs_id in enumerate(submission_specs.get_sim_agent_ids(scenario)):
                if obs_id not in predicted_obs_id_traj.keys(): # if it is not one of the agents to be predicted. then ignore it. 
                    simulated_states.append(np.zeros(shape=(80, 4)))
                else: # otherwise, simulate it
                    one_predicted_obs_traj = predicted_obs_id_traj[obs_id]
                    one_predicted_obs_x = one_predicted_obs_traj[:, 0]
                    one_predicted_obs_y = one_predicted_obs_traj[:, 1]
                    one_predicted_obs_z = np.array([float(logged_trajectories.z[:, -1][index])] * 80)
                    one_predicted_obs_x, one_predicted_obs_y = MapUtil.local_to_global(curr_loc[2], one_predicted_obs_x,
                                                                            one_predicted_obs_y, curr_loc[0], curr_loc[1])
                    one_predicted_obs_heading = MapUtil.theta_local_to_global(curr_loc[2], one_predicted_obs_traj[:, 2])
                    one_simulated_state = np.stack((one_predicted_obs_x, one_predicted_obs_y,
                                                    one_predicted_obs_z, one_predicted_obs_heading), axis=-1)
                    simulated_states.append(one_simulated_state)
            simulated_states = np.stack(simulated_states, axis=0)
            simulated_states = np.stack([simulated_states] * submission_specs.N_ROLLOUTS, axis=0)
            simulated_states = tf.convert_to_tensor(simulated_states)


            joint_scene = EvalUtil.joint_scene_from_states(simulated_states[0, :, :, :],
                                                logged_trajectories.object_id)
            # Validate the joint scene. Should raise an exception if it's invalid.
            submission_specs.validate_joint_scene(joint_scene, scenario)
            scenario_rollouts = EvalUtil.scenario_rollouts_from_states(
                scenario, simulated_states, logged_trajectories.object_id)
            # As before, we can validate the message we just generate.
            submission_specs.validate_scenario_rollouts(scenario_rollouts, scenario)
            # Compute the features for a single JointScene.
            # single_scene_features = metric_features.compute_metric_features(
            #     scenario, joint_scene)

            ### Compute the metrics for the scenario rollouts ###
            config = metrics.load_metrics_config_2()
            scenario_metrics = metrics.compute_scenario_metrics_for_bundle(
                config, scenario, scenario_rollouts)
            vprint(scenario_metrics)
            logs = {
                'metametric': scenario_metrics.metametric,
                'linear_acceleration_likelihood': scenario_metrics.linear_acceleration_likelihood,
                'time_to_collision_likelihood': scenario_metrics.time_to_collision_likelihood,
                'offroad_indication_likelihood': scenario_metrics.offroad_indication_likelihood,
                'min_average_displacement_error': scenario_metrics.min_average_displacement_error,
                'linear_speed_likelihood': scenario_metrics.linear_speed_likelihood,
                'distance_to_road_edge_likelihood': scenario_metrics.distance_to_road_edge_likelihood,
                'distance_to_nearest_object_likelihood': scenario_metrics.distance_to_nearest_object_likelihood,
                'collision_indication_likelihood': scenario_metrics.collision_indication_likelihood,
                'average_displacement_error': scenario_metrics.average_displacement_error,
                'angular_speed_likelihood': scenario_metrics.angular_speed_likelihood,
                'angular_acceleration_likelihood': scenario_metrics.angular_acceleration_likelihood
            }

            if result_info.train_model_config.writer is None:
                vprint("No tensorboard writer found. creating new writer")
                # result_info.train_model_config.writer = SummaryWriter(result_info.train_model_config.log_dir)
                return logs
            writer = result_info.train_model_config.writer
            for key, value in logs.items():
                writer.add_scalar(f'metrics/{key}', value, epoch_num*number_of_scenarios+index)

            # flush the writer
            writer.flush()
            vprint(f"Scenario {index} metrics: {logs}")




    @staticmethod
    def show_results_validation(model, result_info: LoadConfigResultDate, save_dir: str, epoch_num:int,number_of_scenarios: int=10):
        match_filenames = tf.io.matching_files([VALIDATION_DATA_SET_PATH])
        dataset = tf.data.TFRecordDataset(match_filenames, name="train_data")
        dataset = dataset.shuffle(buffer_size=number_of_scenarios*5)  # Shuffle with a buffer size of 1000
        dataset = dataset.take(number_of_scenarios)  # Then take the first 100 elements

        dataset_iterator = dataset.as_numpy_iterator()
        for index, scenario_bytes in enumerate(dataset_iterator):
            scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
            data_dict = DataUtil.transform_data_to_input(scenario, result_info)
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(torch.float32).unsqueeze(dim=0)

            predict_traj = model(data_dict)[-1].cpu().detach()
            predicted_traj_mask = data_dict['predicted_traj_mask'][0].cpu().detach()
            predicted_future_traj = data_dict['predicted_future_traj'][0].cpu().detach()
            predicted_his_traj = data_dict['predicted_his_traj'][0].cpu().detach()
            predicted_num = 0
            for i in range(predicted_traj_mask.shape[0]):
                if int(predicted_traj_mask[i]) == 1:
                    predicted_num += 1
            generate_traj = predict_traj[:predicted_num]
            predicted_future_traj = predicted_future_traj[:predicted_num]
            predicted_his_traj = predicted_his_traj[:predicted_num]
            real_traj = torch.cat((predicted_his_traj, predicted_future_traj), dim=1)[:, :, :2].cpu().detach().numpy()
            real_yaw = torch.cat((predicted_his_traj, predicted_future_traj), dim=1)[:, :, 2].cpu().detach().numpy()
            model_output = torch.cat((predicted_his_traj, generate_traj), dim=1)[:, :, :2].cpu().detach().numpy()
            model_yaw = torch.cat((predicted_his_traj, generate_traj), dim=1)[:, :, 2].cpu().detach().numpy()
            # image_path = os.path.join(save_dir, f"{index}_ground_truth.gif")
            # ShowResultsTask.draw_gif(predicted_num, real_traj, real_yaw, data_dict, scenario, image_path)
            image_path = os.path.join(save_dir, f"{index}_model_output.gif")
            ShowResultsTask.draw_gif(predicted_num, model_output, model_yaw, data_dict, scenario, image_path)

            # validation images in writer 
            # fig = ShowResultsTask.draw_scene(predicted_num, real_traj, data_dict, scenario, os.path.join(save_dir, f"{index}_ground_truth.png"), return_fig=True)
            # result_info.train_model_config.writer.add_figure(f'validation/ground_truth', fig, epoch_num)
            fig = ShowResultsTask.draw_scene(predicted_num, model_output, data_dict, scenario, os.path.join(save_dir, f"{index}_model_output.png"), return_fig=True)
            result_info.train_model_config.writer.add_figure(f'validation/model_output', fig, epoch_num)

    @staticmethod
    def draw_input(scenario: Scenario, image_path: str):
        fig, axis = plt.subplots(1, 1, figsize=(10, 10))
        visualizations.add_map(axis, scenario)
        predicted_obs_ids = submission_specs.get_evaluation_sim_agent_ids(scenario)
        current_time_index = scenario.current_time_index
        for track in scenario.tracks:
            if track.id not in predicted_obs_ids:
                continue
            param_dict = {
                "x": track.states[current_time_index].center_x,
                "y": track.states[current_time_index].center_y,
                "bbox_yaw": track.states[current_time_index].heading,
                "length": track.states[current_time_index].length,
                "width": track.states[current_time_index].width,
            }
            rect = visualizations.get_bbox_patch(**param_dict)
            axis.add_patch(rect)
        plt.savefig(image_path)
        plt.close('all')  # 避免内存泄漏

    @staticmethod
    def draw_scene(
            predicted_num: int, traj: np.ndarray,
            data_dict: Dict[str, Any], scenario: Scenario, image_path: str, return_fig=False
    ):
        fig, axis = plt.subplots(1, 1, figsize=(10, 10))
        visualizations.add_map(axis, scenario)
        # visualizations.get_bbox_patch()
        # axis.axis('equal')  # 横纵坐标比例相等
        curr_x, curr_y, curr_heading, _ = data_dict['curr_loc']
        for i in range(predicted_num):
            real_traj_x, real_traj_y = MapUtil.local_to_global(curr_heading, traj[i, :, 0],
                                                               traj[i, :, 1], curr_x, curr_y)
            num = np.linspace(0, 1, len(real_traj_x))
            for j in range(2, len(real_traj_x)):
                axis.plot(
                    real_traj_x[j - 2:j],
                    real_traj_y[j - 2:j],
                    linewidth=5,
                    color=ShowResultsTask.CMAP(num[j]),
                )
        axis.set_xticks([])
        axis.set_yticks([])
        # plt.show()
        if return_fig:
            return fig
        plt.savefig(image_path)
        plt.close('all')  # 避免内存泄漏
        print(f"{image_path} save success")

    @staticmethod
    def draw_gif(
            predicted_num: int, traj: np.ndarray, real_yaw: np.ndarray,
            data_dict: Dict[str, Any], scenario: Scenario, image_path: str, return_animations=False
    ):
        fig, axis = plt.subplots(1, 1, figsize=(10, 10))
        visualizations.add_map(axis, scenario)
        # visualizations.get_bbox_patch()
        # axis.axis('equal')  # 横纵坐标比例相等
        curr_x, curr_y, curr_heading, _ = data_dict['curr_loc']
        x_list = list()
        y_list = list()
        yaw_list = list()
        for i in range(predicted_num):
            real_traj_x, real_traj_y = MapUtil.local_to_global(curr_heading, traj[i, :, 0],
                                                               traj[i, :, 1], curr_x, curr_y)
            real_traj_yaw = MapUtil.theta_local_to_global(curr_heading, real_yaw[i])
            x_list.append(real_traj_x)
            y_list.append(real_traj_y)
            yaw_list.append(real_traj_yaw)
        # [num, step]
        x_list = np.stack(x_list, axis=0)
        y_list = np.stack(y_list, axis=0)
        yaw_list = np.stack(yaw_list, axis=0)
        predicted_feature = data_dict['predicted_feature'].squeeze()[:, :2]

        def animate(t: int) -> list[patches.Rectangle]:
            # At each animation step, we need to remove the existing patches. This can
            # only be done using the `pop()` operation.
            for _ in range(len(axis.patches)):
                axis.patches.pop()
            bboxes = list()
            for j in range(x_list.shape[0]):
                bboxes.append(axis.add_patch(
                    ShowResultsTask.get_bbox_patch(
                        x_list[:, t][j], y_list[:, t][j], yaw_list[:, t][j],
                        predicted_feature[j, 1], predicted_feature[j, 0], ShowResultsTask.COLOR_DICT[j]
                    )
                ))
            return bboxes

        animations = animation.FuncAnimation(
            fig, animate, frames=x_list.shape[1], interval=100,
            blit=True)
        axis.set_xticks([])
        axis.set_yticks([])
        # plt.show()
        if return_animations:
            return animations
        animations.save(image_path, writer='ffmpeg', fps=30)
        plt.close('all')  # 避免内存泄漏
        print(f"{image_path} save success")
        return animations, fig

    @staticmethod
    def get_bbox_patch(
            x: float, y: float, bbox_yaw: float, length: float, width: float,
            color: np.ndarray
    ) -> patches.Rectangle:
        left_rear_object = np.array([-length / 2, -width / 2])

        rotation_matrix = np.array([[np.cos(bbox_yaw), -np.sin(bbox_yaw)],
                                    [np.sin(bbox_yaw), np.cos(bbox_yaw)]])
        left_rear_rotated = rotation_matrix.dot(left_rear_object)
        left_rear_global = np.array([x, y]) + left_rear_rotated
        color = list(color) + [0.5]
        rect = patches.Rectangle(
            left_rear_global, length, width, angle=np.rad2deg(bbox_yaw), color=color)
        return rect

    @staticmethod
    def draw_gif_from_scenario(
            predicted_num,scenario: Scenario, submission_specs, image_path: str, return_animations=False
        ):
        fig, axis = plt.subplots(1, 1, figsize=(10, 10))
        
        # Add map visualization
        visualizations.add_map(axis, scenario)
        
        # Collect all the tracks that we want to visualize
        tracks = [track for track in scenario.tracks if track.id in submission_specs.get_sim_agent_ids(scenario)]
        
        # Store trajectory points
        x_list = []
        y_list = []
        yaw_list = []
        width_list = []
        length_list = []
        for idx, track in enumerate(tracks):
            if track.id in submission_specs.get_sim_agent_ids(scenario):
            # if idx < predicted_num:
                valids = np.array([state.valid for state in track.states])
                if np.all(valids):
                    x = np.array([state.center_x for i, state in enumerate(track.states)])
                    y = np.array([state.center_y for i, state in enumerate(track.states)])
                    yaw = np.array([state.heading for i, state in enumerate(track.states)])
                    width = np.array([state.width for i, state in enumerate(track.states)])
                    length = np.array([state.length for i, state in enumerate(track.states)])
                    x_list.append(x)
                    y_list.append(y)
                    yaw_list.append(yaw)
                    width_list.append(width)
                    length_list.append(length)
        
        # # Stack the x and y coordinates
        x_list = np.stack(x_list, axis=0)
        y_list = np.stack(y_list, axis=0)
        yaw_list = np.stack(yaw_list, axis=0)
        width_list = np.stack(width_list, axis=0)
        length_list = np.stack(length_list, axis=0)
        # Function to animate the plotting of the tracks
        # Adapted function using the better bounding box implementation
        def animate(t: int) -> list[patches.Rectangle]:
            # Clear previous patches
            for _ in range(len(axis.patches)):
                axis.patches.pop()
            
            bboxes = []
            for j in range(len(x_list)):
                # Use the get_bbox_patch method for better bounding boxes
                bboxes.append(axis.add_patch(
                    ShowResultsTask.get_bbox_patch(
                        x=x_list[j, t], 
                        y=y_list[j, t], 
                        bbox_yaw=yaw_list[j, t],  # Assuming yaw_list contains the orientation for each object
                        length=length_list[j, t], 
                        width=width_list[j, t], 
                        color=ShowResultsTask.COLOR_DICT[j % len(ShowResultsTask.COLOR_DICT)]
                    )
                ))
            return bboxes


        # Create the animation
        animations = animation.FuncAnimation(
            fig, animate, frames=x_list.shape[1], interval=100, blit=True
        )

        axis.set_xticks([])
        axis.set_yticks([])

        if return_animations:
            return animations
        
        # Save the GIF
        animations.save(image_path, writer='ffmpeg', fps=30)
        plt.close('all')  # Avoid memory leak
        print(f"{image_path} saved successfully")

        return animations, fig

    @staticmethod
    def draw_gif_from_animated_states(
        fig: plt.Figure, axis: plt.Axes, scenario: scenario_pb2.Scenario,
        x: tf.Tensor, y: tf.Tensor, yaw: tf.Tensor, length: tf.Tensor,
        width: tf.Tensor, color_idx: tf.Tensor, image_path: str, return_animations=False
    ) -> animation.FuncAnimation:
        """
        Animates the states in a pyplot figure and saves it as a GIF.

        Args:
        fig: The pyplot figure to animate.
        axis: The pyplot axis to which bounding boxes and map are added.
        scenario: The Scenario proto from which the map is extracted.
        x: Array of shape (num_objects, num_steps) of x-coordinates.
        y: Array of shape (num_objects, num_steps) of y-coordinates.
        yaw: Array of shape (num_objects, num_steps) of bounding box yaws.
        length: Array of shape (num_objects, num_steps) of object lengths.
        width: Array of shape (num_objects, num_steps) of object width.
        color_idx: Array of shape (num_objects, num_steps) of color indices, picked
            from the `WAYMO_COLORS` palette.
        image_path: The file path to save the GIF.
        return_animations: If True, return the animation object.

        Returns:
        An animation object if `return_animations` is True, otherwise saves a GIF.
        """
        # To avoid a double figure (one static and one animated), we need to first
        # close the existing pyplot figure.
        plt.close(fig)

        # Add the static map features to the animation.
        visualizations.add_map(axis, scenario)

        def animate(t: int) -> list[patches.Rectangle]:
            # At each animation step, we need to remove the existing patches.
            for _ in range(len(axis.patches)):
                axis.patches.pop()
            
            # Add bounding boxes of objects in the current time step
            bboxes = visualizations.add_all_current_objects(
                axis=axis, x=x[:, t], y=y[:, t], yaw=yaw[:, t], 
                length=length[:, t], width=width[:, t], color_idx=color_idx[:, t]
            )
            return bboxes

        # Create the animation
        animations = animation.FuncAnimation(
            fig, animate, frames=x.shape[1], interval=_ANIMATION_INTERVAL_MS, blit=True
        )

        axis.set_xticks([])
        axis.set_yticks([])

        # Save the animation as a GIF
        if not return_animations:
            animations.save(image_path, writer='ffmpeg', fps=30)
            plt.close('all')  # Avoid memory leak
            print(f"{image_path} saved successfully")
        else:
            return animations, fig


if __name__ == "__main__":
    # show_result()
    pass
