"""
This script is used to visualize the dataset and merge RGB, LiDAR and BEV views as a sanity check.
"""

from copy import deepcopy
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import torch
from config import GlobalConfig
import transfuser_utils as t_u
from torch.utils.data import DataLoader
import math


def visualize_model(
        config,
        step,
        rgb,
        lidar_bev,
        target_point,
        pred_wp,
        target_point_next=None,
        pred_semantic=None,
        pred_bev_semantic=None,
        pred_depth=None,
        pred_checkpoint=None,
        pred_speed=None,
        pred_target_speed_scalar=None,
        pred_bb=None,
        gt_wp=None,
        gt_checkpoints=None,
        gt_bbs=None,
        gt_speed=None,
        gt_bev_semantic=None,
        wp_selected=None):
    # 0 Car, 1 Pedestrian, 2 Red light, 3 Stop sign, 4 emergency vehicle, 5 other
    # color_classes = [
    #   np.array([255, 165, 0]),
    #   np.array([0, 255, 0]),
    #   np.array([255, 0, 0]),
    #   np.array([250, 160, 160]),
    #   np.array([16, 133, 133]),
    #   np.array([128, 0, 128])
    # ]
    color_classes = [
        np.array([51, 172, 255]),  # Car
        np.array([148, 148, 183]),  # Pedestrian
        np.array([255, 0, 255]),  # Red light
        np.array([255, 77, 77]),  # Stop sign
        np.array([51, 172, 255]),  # Emergency vehicle
        np.array([179, 36, 0])  # Other
    ]

    size_width = int((config.max_y - config.min_y) * config.pixels_per_meter)
    size_height = int((config.max_x - config.min_x) * config.pixels_per_meter)

    scale_factor = 4
    origin_x_ratio = config.max_x / (
            config.max_x -
            config.min_x) if config.crop_bev and config.crop_bev_height_only_from_behind else 1
    origin = ((size_width * scale_factor) // 2, (origin_x_ratio * size_height * scale_factor) // 2)
    loc_pixels_per_meter = config.pixels_per_meter * scale_factor

    if pred_bev_semantic is not None:
        bev_semantic_indices = np.argmax(pred_bev_semantic[0], axis=0)
        converter = np.array(config.bev_classes_list)
        converter[1][0:3] = 40
        bev_semantic_image = converter[bev_semantic_indices, ...].astype('uint8')
        alpha = np.ones_like(bev_semantic_indices) * 0.33
        alpha = alpha.astype(np.float32)
        alpha[bev_semantic_indices == 0] = 0.0
        alpha[bev_semantic_indices == 1] = 0.1

        alpha = cv2.resize(alpha,
                           dsize=(alpha.shape[1] * scale_factor, alpha.shape[0] * scale_factor),
                           interpolation=cv2.INTER_NEAREST)
        alpha = np.expand_dims(alpha, 2)
        bev_semantic_image = cv2.resize(bev_semantic_image,
                                        dsize=(bev_semantic_image.shape[1] * scale_factor,
                                               bev_semantic_image.shape[0] * scale_factor),
                                        interpolation=cv2.INTER_NEAREST)

        images_lidar = bev_semantic_image * alpha

    if gt_bev_semantic is not None:
        bev_semantic_indices = gt_bev_semantic
        bev_semantic_indices = bev_semantic_indices[64:192, 64:192].repeat(2, axis=0).repeat(2, axis=1)
        converter = np.array(config.bev_classes_list)
        # converter[1][0:3] = 40
        bev_semantic_image = converter[bev_semantic_indices, ...].astype('uint8')
        alpha = np.ones_like(bev_semantic_indices) * 0.33
        alpha = alpha.astype(np.float32)
        alpha[bev_semantic_indices == 0] = 0.0
        alpha[bev_semantic_indices == 1] = 0.1

        alpha = cv2.resize(alpha,
                           dsize=(alpha.shape[1] * scale_factor, alpha.shape[0] * scale_factor),
                           interpolation=cv2.INTER_NEAREST)
        alpha = np.expand_dims(alpha, 2)
        bev_semantic_image = cv2.resize(bev_semantic_image,
                                        dsize=(bev_semantic_image.shape[1] * scale_factor,
                                               bev_semantic_image.shape[0] * scale_factor),
                                        interpolation=cv2.INTER_NEAREST)
        images_lidar = bev_semantic_image
        images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)

    # Draw wps
    # Red ground truth
    if gt_wp is not None:
        gt_wp_color = (255, 255, 0)
        for wp in gt_wp:
            wp_x = wp[0] * loc_pixels_per_meter + origin[0]
            wp_y = wp[1] * loc_pixels_per_meter + origin[1]
            cv2.circle(images_lidar, (int(wp_x), int(wp_y)), radius=10, color=gt_wp_color, thickness=-1)

    # Orange ground truth checkpoint
    if gt_checkpoints is not None:
        for wp in gt_checkpoints[0]:
            wp_x = wp[0] * loc_pixels_per_meter + origin[0]  # this is where the minus comes from ^
            wp_y = wp[1] * loc_pixels_per_meter + origin[1]
            cv2.circle(images_lidar, (int(wp_x), int(wp_y)), radius=8, lineType=cv2.LINE_AA, color=(0, 0, 0),
                       thickness=-1)

    # Green predicted checkpoint
    if pred_checkpoint is not None:
        for wp in pred_checkpoint[0]:
            wp_x = wp[0] * loc_pixels_per_meter + origin[0]
            wp_y = wp[1] * loc_pixels_per_meter + origin[1]
            cv2.circle(images_lidar, (int(wp_x), int(wp_y)),
                       radius=8,
                       lineType=cv2.LINE_AA,
                       color=(0, 128, 255),
                       thickness=-1)

    # Blue predicted wp
    if pred_wp is not None:
        pred_wps = pred_wp[:, [1, 0]] 
        num_wp = len(pred_wps)
        for idx, wp in enumerate(pred_wps):
            color_weight = 0.5 + 0.5 * float(idx) / num_wp
            wp_x = float(wp[0] * loc_pixels_per_meter + origin[0])
            wp_y = float(wp[1] * loc_pixels_per_meter + origin[1])
            if np.isfinite(wp_x) and np.isfinite(wp_y):
                center = (int(np.round(wp_x)), int(np.round(wp_y)))
                try:
                    cv2.circle(images_lidar, center,
                            radius=8,
                            lineType=cv2.LINE_AA,
                            color=(0, 0, int(color_weight * 255)),
                            thickness=-1)
                except Exception as e:
                    print(f"[Error] {e}")

    # Draw target points
    if config.use_tp:
        x_tp = target_point[1] * loc_pixels_per_meter + origin[0]
        y_tp = target_point[0] * loc_pixels_per_meter + origin[1]
        cv2.circle(images_lidar, (int(origin[0]), int(origin[1])), 5, (0, 255, 0), -1)
        cv2.circle(images_lidar, (int(x_tp), int(y_tp)), radius=12, lineType=cv2.LINE_AA, color=(255, 0, 0),
                   thickness=-1)

        # draw next tp too
        if config.two_tp_input and target_point_next is not None:
            x_tpn = target_point_next[0] * loc_pixels_per_meter + origin[0]
            y_tpn = target_point_next[0] * loc_pixels_per_meter + origin[1]
            cv2.circle(images_lidar, (int(x_tpn), int(y_tpn)),
                       radius=12,
                       lineType=cv2.LINE_AA,
                       color=(255, 0, 0),
                       thickness=-1)

    # draw ego
    sample_box = np.array([
        int(images_lidar.shape[0] / 2),
        int(origin_x_ratio * images_lidar.shape[1] / 2), config.ego_extent_x * loc_pixels_per_meter,
                                                         config.ego_extent_y * loc_pixels_per_meter,
        np.deg2rad(90.0), 0.0
    ])
    images_lidar = t_u.draw_box(images_lidar, sample_box, color=(0, 200, 0), pixel_per_meter=16, thickness=4)

    if pred_bb is not None:
        for box in pred_bb:
            inv_brake = 1.0 - box[6]
            color_box = deepcopy(color_classes[int(box[7])])
            color_box[1] = color_box[1] * inv_brake
            box = t_u.bb_vehicle_to_image_system(box, loc_pixels_per_meter, config.min_x, config.min_y)
            images_lidar = t_u.draw_box(images_lidar, box, color=color_box, pixel_per_meter=loc_pixels_per_meter)

    if gt_bbs is not None:
        gt_bbs = np.array(gt_bbs)
        real_boxes = gt_bbs.sum(axis=-1) != 0.
        gt_bbs = gt_bbs[real_boxes]
        for box in gt_bbs:
            inv_brake = 1.0 - box[6]
            # car: box[7] == 0, walker: box[7] == 1, traffic_light: box[7] == 2, stop_sign: box[7] == 3
            color_box = deepcopy(color_classes[int(box[7])])
            color_box[1] = color_box[1] * inv_brake
            box[:4] = box[:4] * scale_factor
            images_lidar = t_u.draw_box(images_lidar, box, color=color_box, pixel_per_meter=loc_pixels_per_meter,
                                        thickness=4)

    images_lidar = np.rot90(images_lidar, k=1)
    images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)

    rgb_image = rgb

    if wp_selected is not None:
        colors_name = ['blue', 'yellow']
        colors_idx = [(0, 0, 255), (255, 255, 0)]
        cv2.putText(images_lidar, 'Selected: ', (700, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(images_lidar, f'{colors_name[wp_selected]}', (850, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    colors_idx[wp_selected], 2, cv2.LINE_AA)

    if pred_speed is not None:
        pred_speed = pred_speed[0]
        t_u.draw_probability_boxes(images_lidar, pred_speed, config.target_speeds)

    if gt_speed is not None:
        gt_speed_float = gt_speed
        cv2.putText(images_lidar, f'Speed: {gt_speed_float:.2f}', (10, 690), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1,
                    cv2.LINE_AA)

    if pred_target_speed_scalar is not None:
        cv2.putText(images_lidar, f'Pred TS: {pred_target_speed_scalar:.2f}', (10, 660), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 1, cv2.LINE_AA)

    all_images = np.concatenate((rgb_image, images_lidar), axis=0)
    return all_images.astype(np.uint8)
