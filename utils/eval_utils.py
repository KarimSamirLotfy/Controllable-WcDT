import json
import math
import os
import tarfile
from typing import Tuple, Dict, Any, List, Callable

import cv2
import numpy as np
import tensorflow as tf
import torch
import tqdm
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils import trajectory_utils
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

from utils.data_utils import DataUtil
from utils.map_utils import MapUtil

class EvalUtil:
    @staticmethod
    def joint_scene_from_states(
            states: tf.Tensor, object_ids: tf.Tensor
    ) -> sim_agents_submission_pb2.JointScene:
        """Packages a simulated state trajectory into a JointScene proto message. 
        Args:
            states: A tensor of shape (num_objects, num_steps, 4) representing the simulated trajectory.
            object_ids: A tensor of shape (num_objects,) representing the object IDs."""
        # States shape: (num_objects, num_steps, 4).
        # Objects IDs shape: (num_objects,).
        if isinstance(states, tf.Tensor) or isinstance(states, torch.Tensor):
            states = states.numpy()
        simulated_trajectories = []
        for i_object in range(len(object_ids)):
            simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
                center_x=states[i_object, :, 0], center_y=states[i_object, :, 1],
                center_z=states[i_object, :, 2], heading=states[i_object, :, 3],
                object_id=object_ids[i_object]
            ))
        return sim_agents_submission_pb2.JointScene(
            simulated_trajectories=simulated_trajectories)
    @staticmethod
    # Now we can replicate this strategy to export all the parallel simulations.
    def scenario_rollouts_from_states(
            scenario: scenario_pb2.Scenario,
            states: tf.Tensor, object_ids: tf.Tensor
    ) -> sim_agents_submission_pb2.ScenarioRollouts:
        # States shape: (num_rollouts, num_objects, num_steps, 4).
        # Objects IDs shape: (num_objects,).
        joint_scenes = []
        for i_rollout in range(states.shape[0]):
            joint_scenes.append(EvalUtil.joint_scene_from_states(states[i_rollout], object_ids))
        return sim_agents_submission_pb2.ScenarioRollouts(
            # Note: remember to include the Scenario ID in the proto message.
            joint_scenes=joint_scenes, scenario_id=scenario.scenario_id)
    
    @staticmethod
    def to_sumulated_states(curr_x, curr_y, curr_z, curr_heading, traj, real_yaw, predicted_num):
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

        x_list = np.stack(x_list, axis=0)
        y_list = np.stack(y_list, axis=0)
        yaw_list = np.stack(yaw_list, axis=0)
        z_list = np.full_like(x_list, float(curr_z)) # We don't ac
        return np.stack((x_list, y_list, z_list, yaw_list), axis=-1)
        