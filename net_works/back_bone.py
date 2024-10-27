#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: back_bone.py
@Author: YangChen
@Date: 2023/12/27
"""
from typing import Dict

import numpy as np
import torch
from torch import nn

from net_works.diffusion import GaussianDiffusion
from net_works.scene_encoder import SceneEncoder
from net_works.traj_decoder import TrajDecoder
from utils import MathUtil


class MultiModalLoss(nn.Module):
    def __init__(self):
        super(MultiModalLoss, self).__init__()
        self.huber_loss = nn.HuberLoss(reduction="none")
        self.confidence_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, traj, confidence, predicted_future_traj, predicted_traj_mask):
        batch = traj.shape[0]
        obs_num = traj.shape[1]
        multimodal = traj.shape[2]
        future_step = traj.shape[3]
        output_dim = traj.shape[4]
        predicted_future_extend = predicted_future_traj.unsqueeze(dim=-3)
        loss = self.huber_loss(traj, predicted_future_extend)
        loss = loss.view(batch, obs_num, multimodal, -1)
        loss = torch.mean(loss, dim=-1)
        min_loss, _ = torch.min(loss, dim=-1)
        min_loss_modal = torch.argmin(loss, dim=-1)
        min_loss_index = min_loss_modal.view(batch, obs_num, 1, 1, 1).repeat(
            (1, 1, 1, future_step, output_dim)
        )
        min_loss_traj = torch.gather(traj, -3, min_loss_index).squeeze()
        confidence = confidence.view(-1, multimodal)
        min_loss_modal = min_loss_modal.view(-1)
        confidence_loss = self.confidence_loss(confidence, min_loss_modal).view(batch, obs_num)
        traj_loss = torch.sum(min_loss * predicted_traj_mask) / (torch.sum(predicted_traj_mask) + 0.00001)
        confidence_loss = torch.sum(confidence_loss * predicted_traj_mask) / (torch.sum(predicted_traj_mask) + 0.00001)
        return traj_loss, confidence_loss, min_loss_traj


class BackBone(nn.Module):
    def __init__(self, betas: np.ndarray, diffusion_type: str, teacher_forcing: bool):
        super(BackBone, self).__init__()
        self.diffusion = GaussianDiffusion(betas=betas, diffusion_type=diffusion_type)
        self.scene_encoder = SceneEncoder()
        self.traj_decoder = TrajDecoder()
        self.multi_modal_loss = MultiModalLoss()

        self.teacher_forcing = teacher_forcing

    def forward(self, data: Dict):
        # batch, other_obs(10), 40, 7
        predicted_feature = data['predicted_feature'] # 4, 8, 7
        # batch, other_obs(10), 40, 5
        other_his_pos = data['other_his_pos'] # 4, 6, 2
        other_his_traj_delt = data['other_his_traj_delt'] # 4, 6, 10, 5
        other_feature = data['other_feature'] # 4, 6, 7
        other_traj_mask = data['other_traj_mask']
        # batch, pred_obs(15), 40, 5
        predicted_his_pos = data['predicted_his_pos'] # 4, 8, 2
        predicted_his_traj_delt = data['predicted_his_traj_delt'] # 4, 8, 10, 5
        predicted_his_traj = data['predicted_his_traj']
        # batch, pred_obs(15), 50, 5
        predicted_future_traj = data['predicted_future_traj']
        predicted_traj_mask = data['predicted_traj_mask']
        # batch, tl_num(10), 2
        traffic_light = data['traffic_light']
        traffic_light_pos = data['traffic_light_pos']
        # batch, num_lane(32), num_point(128), 2
        lane_list = data['lane_list']
        # diffusion训练
        noise = torch.randn_like(predicted_his_traj_delt)
        diffusion_loss = self.diffusion(data)

        # HEre we use a sampled path from noise
        if self.teacher_forcing:
            noise = predicted_his_traj_delt
        else:
            noise = self.diffusion.sample(noise, predicted_his_traj) # 64 seconds batch 4
        # scene encoder
        scene_feature = self.scene_encoder(
            noise, lane_list,
            other_his_traj_delt, other_his_pos, other_feature,
            predicted_his_traj_delt, predicted_his_pos, predicted_feature,
            traffic_light, traffic_light_pos
        )
        # traj_decoder
        traj, confidence = self.traj_decoder(scene_feature)
        traj = MathUtil.post_process_output(traj, predicted_his_traj)
        traj_loss, confidence_loss, min_loss_traj = self.multi_modal_loss(traj, confidence, predicted_future_traj,
                                                                          predicted_traj_mask)
        return diffusion_loss, traj_loss, confidence_loss, min_loss_traj

    def predict(self, data: Dict): 
        # batch, other_obs(10), 40, 7
        predicted_feature = data['predicted_feature']
        # batch, other_obs(10), 40, 5
        other_his_pos = data['other_his_pos']
        other_his_traj_delt = data['other_his_traj_delt']
        other_feature = data['other_feature']
        other_traj_mask = data['other_traj_mask']
        # batch, pred_obs(15), 40, 5
        predicted_his_pos = data['predicted_his_pos']
        predicted_his_traj_delt = data['predicted_his_traj_delt']
        predicted_his_traj = data['predicted_his_traj']
        predicted_traj_mask = data['predicted_traj_mask']
        # batch, pred_obs(15), 50, 5
        predicted_future_traj = data['predicted_future_traj']
        predicted_traj_mask = data['predicted_traj_mask']
        # batch, tl_num(10), 2
        traffic_light = data['traffic_light']
        traffic_light_pos = data['traffic_light_pos']
        # batch, num_lane(32), num_point(128), 2
        lane_list = data['lane_list']
        # diffusion训练
        noise = torch.randn_like(predicted_his_traj_delt)
        noise = self.diffusion.sample(noise, predicted_his_traj) # 64 seconds batch 4
        # scene encoder
        scene_feature = self.scene_encoder(
            noise, lane_list,
            other_his_traj_delt, other_his_pos, other_feature,
            predicted_his_traj_delt, predicted_his_pos, predicted_feature,
            traffic_light, traffic_light_pos
        )
        # traj_decoder
        traj, confidence = self.traj_decoder(scene_feature)
        traj = MathUtil.post_process_output(traj, predicted_his_traj)
        traj_loss, confidence_loss, min_loss_traj = self.multi_modal_loss(traj, confidence, predicted_future_traj,
                                                                          predicted_traj_mask)                    
        return min_loss_traj, confidence
    
    def sample_conditioned(self, data: Dict, f_x):
        # batch, other_obs(10), 40, 7
        predicted_feature = data['predicted_feature']
        # batch, other_obs(10), 40, 5
        other_his_pos = data['other_his_pos']
        other_his_traj_delt = data['other_his_traj_delt']
        other_feature = data['other_feature']
        other_traj_mask = data['other_traj_mask']
        # predicted_his_pos is the starting position that the model will be conditoned on.
        predicted_his_pos = data['predicted_his_pos'] # shape = batch, max_pred_num, 2
        predicted_his_traj_delt = data['predicted_his_traj_delt'] # shape = batch, max_pred_num, timesteps, 5
        predicted_his_traj = data['predicted_his_traj']
        predicted_traj_mask = data['predicted_traj_mask']
        # batch, pred_obs(15), 50, 5
        predicted_future_traj = data['predicted_future_traj']
        predicted_traj_mask = data['predicted_traj_mask']
        # batch, tl_num(10), 2
        traffic_light = data['traffic_light']
        traffic_light_pos = data['traffic_light_pos']
        # batch, num_lane(32), num_point(128), 2
        lane_list = data['lane_list']
        # diffusion训练
        noise = torch.randn_like(predicted_his_traj_delt)
        print(f'noise shape: {noise.shape}')
        ### HERE WE START SAMPLING AND PROPAGATING THE NOISE
        for t in range(self.diffusion.num_time_steps - 1, -1, -1):
            condition = predicted_his_traj
            behavior = self.diffusion.sample_step(noise, t, condition)

            # scene encoder
            scene_feature = self.scene_encoder(
                behavior, lane_list,
                other_his_traj_delt, other_his_pos, other_feature,
                predicted_his_traj_delt, predicted_his_pos, predicted_feature,
                traffic_light, traffic_light_pos
            )
            # traj_decoder
            # shape: batch, max_pred_num, Multi-modal(AKA anchors), timesteps, 3
            # confidence.shape = batch, max_pred_num, Multi-modal
            traj, confidence = self.traj_decoder(scene_feature) 

            # shape_post_processed = batch, max_pred_num, multi-modal, timesteps, 5
            traj = MathUtil.post_process_output(traj, predicted_his_traj)

            # min_loss_traj.shape = batch, max_pred_num, timesteps, 5
            traj_loss, confidence_loss, min_loss_traj = self.multi_modal_loss(traj, confidence, predicted_future_traj,
                                                                            predicted_traj_mask)                    

            unguided_min_traj_loss = min_loss_traj
            score = f_x(unguided_min_traj_loss)
            gradient_of_score = torch.autograd.grad(score, behavior, create_graph=True)[0]
            behavior = behavior + 3*gradient_of_score
            noise = behavior
            yield min_loss_traj, confidence