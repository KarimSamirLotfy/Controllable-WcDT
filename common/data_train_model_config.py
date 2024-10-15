#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: train_model_config.py
@Author: YangChen
@Date: 2023/12/27
"""
from typing import List

from common.data import BaseConfig


class TrainModelConfig(BaseConfig):
    use_gpu: bool = True
    gpu_ids: List[int] = [1]
    batch_size: int = 8
    num_works: int = 20
    his_step: int = 11
    max_pred_num: int = 8
    max_other_num: int = 6
    max_traffic_light: int = 8
    max_lane_num: int = 32
    max_point_num: int = 128
    num_head: int = 8
    attention_dim: int = 128
    multimodal: int = 10
    time_steps: int = 100
    schedule: str = "cosine"
    num_epoch: int = 0
    init_lr: float = 0.00001
    diffusion_type: str = "dit" # dit, unet, none (None for )

    teacher_forcing: bool = True
    # saving model 
    save_model: bool = False
    save_interval_epoch: int = 10
