import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict
import torch
import numpy as np
from PIL import Image
from scipy.integrate import RK45
from torchvision import transforms as T
from Bench2DriveZoo.team_code.pid_controller import PIDController
from Bench2DriveZoo.team_code.planner import RoutePlanner
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from longitudinal_controller import LongitudinalLinearRegressionController
from agents.navigation.local_planner import RoadOption
from lateral_controller import LateralPIDController
from Bench2DriveZoo.team_code.kinematic_bicycle_model import KinematicBicycleModel
from leaderboard.autoagents import autonomous_agent
from Bench2DriveZoo.team_code.planner_mr import RoutePlannerMR
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import (get_dist_info, init_dist,
                        load_checkpoint, wrap_fp16_model)
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type
from pyquaternion import Quaternion
from scipy.optimize import fsolve

from vad_b2d_agent import VadAgent
from shapely.geometry import Polygon
import transfuser_utils as t_u
from birds_eye_view.chauffeurnet import ObsManager
from birds_eye_view.run_stop_sign import RunStopSign
from config import GlobalConfig
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from agents.tools.misc import (
    is_within_distance, get_trafficlight_trigger_location, compute_distance)
from visualize_dataset import visualize_model

import threading
import queue
from threading import Lock

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)


def get_entry_point():
    return 'VadAgentMRP'


class VadAgentMRP(VadAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self.track = autonomous_agent.Track.MAP
        self.config = GlobalConfig()
        self.last_lidar = None
        self.last_ego_transform = None
        self.tmp_visu = True

        # KBM
        self.ego_model = KinematicBicycleModel(self.config)
        self.vehicle_model = KinematicBicycleModel(self.config)

        # Takeover Gate
        self.takeover_buffers = {
            'collision': [0] * 5,
            'stop_signal': [0] * 5,
            'stalling': [0] * 200,
        }
        self.recovery_buffer = [0] * 20
        self.buffer_indices = {
            'collision': 0,
            'stop_signal': 0,
            'stalling': 0,
            'recovery': 0
        }
        self.collision_time_buffer = [float('inf')] * 5
        self.buffer_indices['collision_time'] = 0

        self.hazard_monitor_queue = queue.Queue(maxsize=10)
        self.hazard_monitor_lock = Lock()
        self.hazard_monitor_thread = None
        self.monitor_running = False

        self.current_hazards = {
            'collision': False,
            'stop_signal': False,
            'stalling': False
        }

        self.takeover_active = False
        self.mitigation_type = None
        self.pending_stop_signs = {}
        self.cleared_stop_sign = False
        self.list_traffic_lights = []
        self.reroute = True
        self.last_static_bounding_boxes = None
        self.last_collision_time = None

        self.world_map = CarlaDataProvider.get_map()

        if SAVE_PATH is not None:
            # (self.save_path / 'lidar').mkdir()
            # (self.save_path / 'rgb').mkdir()
            # (self.save_path / 'semantics').mkdir()
            # (self.save_path / 'bev_semantics').mkdir()
            # (self.save_path / 'boxes').mkdir()
            (self.save_path / 'privilege_bev').mkdir()

    def _init(self):
        super()._init()

        print("Sparse Waypoints:", len(self._global_plan))
        print("Dense Waypoints:", len(self.org_dense_route_world_coord))

        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()

        obs_config = {
            'width_in_pixels': self.config.lidar_resolution_width,  # 256
            'pixels_ev_to_bottom': self.config.lidar_resolution_height / 2.0,  # 256/2 = 128
            'pixels_per_meter': self.config.pixels_per_meter_collection,  # 2.0
            'history_idx': [-1],
            'scale_bbox': True,
            'scale_mask_col': 1.0,
            'map_folder': 'maps_2ppm_cv'
        }

        self.stop_sign_criteria = RunStopSign(self._world)
        self.ss_bev_manager = ObsManager(obs_config, self.config)
        self.ss_bev_manager.attach_ego_vehicle(
            self._vehicle, criteria_stop=self.stop_sign_criteria)

        distance_to_road = self.org_dense_route_world_coord[0][0].location.distance(
            self._vehicle.get_location())
        starts_with_parking_exit = distance_to_road > 2

        # Set up the route planner and extrapolation
        self._waypoint_planner = RoutePlannerMR(self.config)
        self._waypoint_planner.setup_route(self.org_dense_route_world_coord, self._world, self.world_map,
                                           starts_with_parking_exit, self._vehicle.get_location())
        self._waypoint_planner.save()

        # Set up the longitudinal controller and command planner
        self._longitudinal_controller = LongitudinalLinearRegressionController(
            self.config)
        self._turn_controller = LateralPIDController(self.config)

        all_actors = self._world.get_actors()
        for actor in all_actors:
            if "traffic_light" in actor.type_id:
                center, waypoints = t_u.get_traffic_light_waypoints(
                    actor, self.world_map)
                self.list_traffic_lights.append((actor, center, waypoints))
            if "vehicle" in actor.type_id:
                extent = actor.bounding_box.extent
                if extent.x < 0.001 or extent.y < 0.001 or extent.z < 0.001:
                    actor.destroy()
        self._start_hazard_monitor()

    def sensors(self):
        result = super().sensors()
        if self.save_path is not None:
            result.append({
                'type': 'sensor.camera.rgb',
                'x': self.config.camera_pos[0],
                'y': self.config.camera_pos[1],
                'z': self.config.camera_pos[2],
                'roll': self.config.camera_rot_0[0],
                'pitch': self.config.camera_rot_0[1],
                'yaw': self.config.camera_rot_0[2],
                'width': self.config.camera_width,
                'height': self.config.camera_height,
                'fov': self.config.camera_fov,
                'id': 'rgb'
            })
            result.append({
                'type': 'sensor.lidar.ray_cast',
                'x': self.config.lidar_pos[0],
                'y': self.config.lidar_pos[1],
                'z': self.config.lidar_pos[2],
                'roll': self.config.lidar_rot[0],
                'pitch': self.config.lidar_rot[1],
                'yaw': self.config.lidar_rot[2],
                'rotation_frequency': self.config.lidar_rotation_frequency,
                'points_per_second': self.config.lidar_points_per_second,
                'id': 'lidar'
            })
        # print(result, flush=True)
        return result

    def tick(self, input_data):
        result = super().tick(input_data)

        rgb = input_data['rgb'][1][:, :, :3]
        # semantics = input_data['semantics'][1][:, :, 2]

        # The 10 Hz LiDAR only delivers half a sweep each time step at 20 Hz.
        # Here we combine the 2 sweeps into the same coordinate system
        if self.last_lidar is not None:
            ego_transform = self._vehicle.get_transform()
            ego_location = ego_transform.location
            last_ego_location = self.last_ego_transform.location
            relative_translation = np.array([
                ego_location.x - last_ego_location.x, ego_location.y - last_ego_location.y,
                ego_location.z - last_ego_location.z
            ])

            ego_yaw = ego_transform.rotation.yaw
            last_ego_yaw = self.last_ego_transform.rotation.yaw
            relative_rotation = np.deg2rad(
                t_u.normalize_angle_degree(ego_yaw - last_ego_yaw))

            orientation_target = np.deg2rad(ego_yaw)
            # Rotate difference vector from global to local coordinate system.
            rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                        [np.sin(orientation_target),
                                         np.cos(orientation_target), 0.0], [0.0, 0.0, 1.0]])
            relative_translation = rotation_matrix.T @ relative_translation

            lidar_last = t_u.algin_lidar(
                self.last_lidar, relative_translation, relative_rotation)
            # Combine back and front half of LiDAR
            lidar_360 = np.concatenate(
                (input_data['lidar'], lidar_last), axis=0)
        else:
            lidar_360 = input_data['lidar']  # The first frame only has 1 half

        bounding_boxes = self.get_bounding_boxes(lidar=lidar_360)

        self.stop_sign_criteria.tick(self._vehicle)

        bev_semantics = self.ss_bev_manager.get_observation([])

        result.update({
            'lidar': lidar_360,
            'rgb': rgb,
            'bev_semantics': bev_semantics['bev_semantic_classes'],
            'bounding_boxes': bounding_boxes,
        })
        return result

        def _start_hazard_monitor(self):
        self.monitor_running = True
        self.hazard_monitor_thread = threading.Thread(
            target=self._hazard_monitor_worker, daemon=True)
        self.hazard_monitor_thread.start()
        print("[ARGUS] Hazard Monitor thread started", flush=True)

    def _hazard_monitor_worker(self):
        while self.monitor_running:
            try:
                monitor_data = self.hazard_monitor_queue.get(timeout=0.1)
                hazard_results = self._evaluate_hazards(monitor_data)
                with self.hazard_monitor_lock:
                    self._update_shared_buffers(hazard_results)

                self.hazard_monitor_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ARGUS] Hazard Monitor error: {e}", flush=True)

    def _evaluate_hazards(self, monitor_data):
        pred_wp = monitor_data['pred_wp']
        bounding_boxes = monitor_data['bounding_boxes']
        control = monitor_data['control']
        ego_speed = monitor_data['ego_speed']
        ego_position = monitor_data['ego_position']

        hazard_results = {
            'collision': False,
            'stop_signal': False,
            'stalling': False,
            'recovery_safe': True
        }

        try:
            static_bounding_boxes, collision_time, collision_bbox, collision_actors = self.kbm_based_collision_risk(
                pred_wp=pred_wp,
                pred_bb=bounding_boxes,
                control=control,
            )
            self.last_static_bounding_boxes = static_bounding_boxes
            idx = self.buffer_indices['collision_time']
            previous_time = self.collision_time_buffer[idx]
            if collision_time is not None and (collision_time <= previous_time or collision_time <= self.config.dangerous_time):
                hazard_results['collision'] = True
                hazard_results['recovery_safe'] = False
                self.buffer_indices['collision_time'] = (
                    idx + 1) % len(self.collision_time_buffer)
                self.collision_time_buffer[idx] = collision_time
            else:
                self.collision_time_buffer[idx] = float('inf')
        except Exception as e:
            print(f"[ARGUS] Collision risk evaluation error: {e}", flush=True)

        try:
            stop_signal_violation = self._evaluate_stop_signal_hazard(
                ego_position)
            if stop_signal_violation:
                hazard_results['stop_signal'] = True
                hazard_results['recovery_safe'] = False
        except Exception as e:
            print(f"[ARGUS] Stop signal evaluation error: {e}", flush=True)

        try:
            if ego_speed < self.config.STOP_THRESHOLD_SPEED:
                hazard_results['stalling'] = True
                hazard_results['recovery_safe'] = False
        except Exception as e:
            print(f"[ARGUS] Stalling evaluation error: {e}", flush=True)

        return hazard_results

    def _evaluate_stop_signal_hazard(self, ego_position):
        for traffic_light, center, waypoints in self.list_traffic_lights:
            if traffic_light.state == carla.TrafficLightState.Red:
                distance_to_light = ego_position.distance(center)
                if distance_to_light < 5.0: 
                    return {'type': 'red_light', 'distance': distance_to_light, 'actor': traffic_light}
        if hasattr(self.stop_sign_criteria, 'test') and self.stop_sign_criteria.test():
            return {'type': 'stop_sign', 'criteria': True}
        return None

    def _update_shared_buffers(self, hazard_results):
        for hazard_type in self.takeover_buffers.keys():
            idx = self.buffer_indices[hazard_type]
            if hazard_results[hazard_type]:
                self.takeover_buffers[hazard_type][idx] = 1
            else:
                self.takeover_buffers[hazard_type][idx] = 0
            self.buffer_indices[hazard_type] = (
                idx + 1) % len(self.takeover_buffers[hazard_type])
        recovery_idx = self.buffer_indices['recovery']
        if hazard_results['recovery_safe']:
            self.recovery_buffer[recovery_idx] = 1
        else:
            self.recovery_buffer[recovery_idx] = 0
        self.buffer_indices['recovery'] = (
            recovery_idx + 1) % len(self.recovery_buffer)

    def _takeover_gate(self, ads_control):
        with self.hazard_monitor_lock:
            takeover_needed = False
            active_hazards = []

            for hazard_type, buffer in self.takeover_buffers.items():
                if hazard_type == 'stalling':
                    if sum(buffer) == 200:
                        takeover_needed = True
                        active_hazards.append(hazard_type)
                        self.current_hazards[hazard_type] = True
                else:
                    if sum(buffer) >= 3:
                        takeover_needed = True
                        active_hazards.append(hazard_type)
                        self.current_hazards[hazard_type] = True
            recovery_ready = sum(self.recovery_buffer) == 20

        if takeover_needed:
            self.takeover_active = True
            self.mitigation_type = active_hazards
            print(
                f"[ARGUS] Takeover activated for hazards: {active_hazards}", flush=True)
            return self._idm_hazard_mitigator()
        elif self.takeover_active and recovery_ready:
            self.takeover_active = False
            self.mitigation_type = None
            self._reset_hazard_states()
            print("[ARGUS] Control returned to ADS", flush=True)
            return ads_control
        elif self.takeover_active:
            return self._idm_hazard_mitigator()
        else:
            return ads_control

    def _idm_hazard_mitigator(self):
        position = self._vehicle.get_location()
        ego_position_carla = np.array([position.x, position.y, position.z])

        if 'collision' in self.mitigation_type:
            control = self._get_control_idm(ego_position_carla)
            print(f"[ARGUS] Collision mitigation active", flush=True)
        if 'stop_signal' in self.mitigation_type:
            control = self._get_control_idm(ego_position_carla)
            print(f"[ARGUS] Stop signal mitigation active", flush=True)
        if 'stalling' in self.mitigation_type:
            if self.reroute:
                self._waypoint_planner.replan_with_dstar_lite(
                    ego_position_carla, self.last_static_bounding_boxes)
                self.reroute = False
            control = self._get_control_idm(ego_position_carla)
            print(f"[ARGUS] Stalling mitigation active", flush=True)

        else:
            control = self._get_control_idm(ego_position_carla)
        return control

    def _reset_hazard_states(self):
        for hazard_type in self.current_hazards:
            self.current_hazards[hazard_type] = False

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        input_data['lidar'] = t_u.lidar_to_ego_coordinate(
            self.config, input_data['lidar'])

        ads_control = super().run_step(input_data, timestamp)

        lidar_bev_image = self.lidar_to_histogram_features(
            self.tick_data['lidar'])
        theta_to_lidar = self.tick_data['compass']
        command_near_xy = np.array(
            [self.tick_data['command_near_xy'][0] - self.tick_data['pos'][0],
             -self.tick_data['command_near_xy'][1] + self.tick_data['pos'][1]])
        rotation_matrix = np.array(
            [[np.cos(theta_to_lidar), -np.sin(theta_to_lidar)], [np.sin(theta_to_lidar), np.cos(theta_to_lidar)]])
        local_command_xy = rotation_matrix @ command_near_xy

        bounding_boxes, draw_bounding_boxes = t_u.parse_bounding_boxes(
            self.config, self.tick_data['bounding_boxes'])
        gt_bev_semantic = self.tick_data['bev_semantics']
        ego_speed = self.tick_data['speed']

        print(
            f"[DEBUG] Time {self.step // 10} p's {self.step % 10} frame", flush=True)

        position = self._vehicle.get_location()
        ego_position_carla = np.array([position.x, position.y, position.z])

        print(
            f"[ARGUS] Step {self.step}, Takeover: {self.takeover_active}, Type: {self.mitigation_type}", flush=True)
        monitor_data = {
            'pred_wp': self.pred_wp,
            'bounding_boxes': bounding_boxes,
            'control': ads_control,
            'ego_speed': ego_speed,
            'ego_position': position
        }

        try:
            self.hazard_monitor_queue.put(monitor_data, block=False)
        except queue.Full:
            try:
                self.hazard_monitor_queue.get(block=False)
                self.hazard_monitor_queue.put(monitor_data, block=False)
            except queue.Empty:
                pass
        final_control = self._takeover_gate(ads_control)
        if not self.takeover_active:
            self._waypoint_planner.run_step(ego_position_carla)

        self.last_lidar = input_data['lidar']
        self.last_ego_transform = self._vehicle.get_transform()

        idm_wp = self.get_future_route_local_coords(self._waypoint_planner,
                                                    np.array(self._vehicle.get_transform().get_matrix()))
        self.tick_data["privileged_bev"] = visualize_model(
            config=self.config,
            step=self.step,
            rgb=self.tick_data['rgb'],
            lidar_bev=lidar_bev_image,
            target_point=local_command_xy,
            pred_wp=self.pred_wp,
            target_point_next=None,
            pred_semantic=None,
            pred_bev_semantic=None,
            pred_depth=None,
            pred_checkpoint=None,
            pred_speed=None,  # prob_target_speed
            pred_target_speed_scalar=None,  # pred_target_speed_scalar
            pred_bb=None,
            gt_wp=idm_wp,  # IDM wp
            gt_checkpoints=None,
            gt_bbs=draw_bounding_boxes,
            gt_speed=ego_speed,
            gt_bev_semantic=gt_bev_semantic,
            wp_selected=None,
        )
        if SAVE_PATH is not None:
            self.save(self.tick_data)

        return final_control

    def save(self, tick_data):
        super().save(tick_data)
        frame = self.step // 10
        Image.fromarray(tick_data["privileged_bev"]).save(
            self.save_path / 'privilege_bev' / ('%04d.png' % frame))

    def destroy(self):
        self.monitor_running = False
        if self.hazard_monitor_thread and self.hazard_monitor_thread.is_alive():
            self.hazard_monitor_thread.join(timeout=1.0)
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * \
            (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * \
            math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])

    def get_future_route_local_coords(self, waypoint_planner, ego_matrix, max_distance=32.0, sample_distance=0.5):
        points_per_meter = waypoint_planner.points_per_meter
        route_points = waypoint_planner.route_points
        current_index = waypoint_planner.route_index

        max_points = int(points_per_meter * max_distance)
        total_points = route_points.shape[0]
        to_index = min(current_index + max_points, total_points)

        sample_step = max(int(points_per_meter * sample_distance), 1)
        future_points_world = route_points[current_index:to_index:sample_step]

        ego_translation = ego_matrix[:3, 3]
        ego_rotation = ego_matrix[:3, :3].T

        local_coords = []
        for point in future_points_world:
            relative_pos = point - ego_translation
            local_pos = ego_rotation @ relative_pos
            local_coords.append(local_pos[:2])

        return np.vstack(local_coords)

    def get_bounding_boxes(self, lidar=None):
        results = []

        ego_transform = self._vehicle.get_transform()
        ego_control = self._vehicle.get_control()
        ego_velocity = self._vehicle.get_velocity()
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_rotation = ego_transform.rotation
        ego_extent = self._vehicle.bounding_box.extent
        ego_speed = self._get_forward_speed(
            transform=ego_transform, velocity=ego_velocity)
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z])
        ego_yaw = np.deg2rad(ego_rotation.yaw)
        ego_brake = ego_control.brake

        relative_yaw = 0.0
        relative_pos = t_u.get_relative_transform(ego_matrix, ego_matrix)

        self._actors = self._world.get_actors()
        vehicle_list = self._actors.filter('*vehicle*')

        result = {
            'class': 'ego_car',
            'extent': [ego_dx[0], ego_dx[1], ego_dx[2]],
            'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
            'yaw': relative_yaw,
            'num_points': -1,
            'distance': -1,
            'speed': ego_speed,
            'brake': ego_brake,
            'id': int(self._vehicle.id),
            'matrix': ego_transform.get_matrix()
        }
        results.append(result)

        for vehicle in vehicle_list:
            if vehicle.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
                if vehicle.id != self._vehicle.id:
                    vehicle_transform = vehicle.get_transform()
                    vehicle_rotation = vehicle_transform.rotation
                    vehicle_matrix = np.array(vehicle_transform.get_matrix())
                    vehicle_control = vehicle.get_control()
                    vehicle_velocity = vehicle.get_velocity()
                    vehicle_extent = vehicle.bounding_box.extent
                    vehicle_id = vehicle.id

                    vehicle_extent_list = [
                        vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]
                    yaw = np.deg2rad(vehicle_rotation.yaw)

                    relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
                    relative_pos = t_u.get_relative_transform(
                        ego_matrix, vehicle_matrix)
                    vehicle_speed = self._get_forward_speed(
                        transform=vehicle_transform, velocity=vehicle_velocity)
                    vehicle_brake = vehicle_control.brake
                    vehicle_steer = vehicle_control.steer
                    vehicle_throttle = vehicle_control.throttle

                    # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
                    if not lidar is None:
                        num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, vehicle_extent_list,
                                                                     lidar)
                    else:
                        num_in_bbox_points = -1

                    distance = np.linalg.norm(relative_pos)

                    result = {
                        'class': 'car',
                        'extent': vehicle_extent_list,
                        'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                        'yaw': relative_yaw,
                        'num_points': int(num_in_bbox_points),
                        'distance': distance,
                        'speed': vehicle_speed,
                        'brake': vehicle_brake,
                        'steer': vehicle_steer,
                        'throttle': vehicle_throttle,
                        'id': int(vehicle_id),
                        'role_name': vehicle.attributes['role_name'],
                        'type_id': vehicle.type_id,
                        'matrix': vehicle_transform.get_matrix()
                    }
                    results.append(result)

        walkers = self._actors.filter('*walker*')
        for walker in walkers:
            if walker.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
                walker_transform = walker.get_transform()
                walker_velocity = walker.get_velocity()
                walker_rotation = walker.get_transform().rotation
                walker_matrix = np.array(walker_transform.get_matrix())
                walker_id = walker.id
                walker_extent = walker.bounding_box.extent
                walker_extent = [walker_extent.x,
                                 walker_extent.y, walker_extent.z]
                yaw = np.deg2rad(walker_rotation.yaw)

                relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
                relative_pos = t_u.get_relative_transform(
                    ego_matrix, walker_matrix)

                walker_speed = self._get_forward_speed(
                    transform=walker_transform, velocity=walker_velocity)

                # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
                if not lidar is None:
                    num_in_bbox_points = self.get_points_in_bbox(
                        relative_pos, relative_yaw, walker_extent, lidar)
                else:
                    num_in_bbox_points = -1

                distance = np.linalg.norm(relative_pos)

                result = {
                    'class': 'walker',
                    'extent': walker_extent,
                    'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                    'yaw': relative_yaw,
                    'num_points': int(num_in_bbox_points),
                    'distance': distance,
                    'speed': walker_speed,
                    'id': int(walker_id),
                    'matrix': walker_transform.get_matrix()
                }
                results.append(result)

        # Note this only saves static actors, which does not include static background objects
        static_list = self._actors.filter('*static*')
        for static in static_list:
            if static.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
                static_transform = static.get_transform()
                static_velocity = static.get_velocity()
                static_rotation = static.get_transform().rotation
                static_matrix = np.array(static_transform.get_matrix())
                static_id = static.id
                static_extent = static.bounding_box.extent
                static_extent = [static_extent.x,
                                 static_extent.y, static_extent.z]
                yaw = np.deg2rad(static_rotation.yaw)

                relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
                relative_pos = t_u.get_relative_transform(
                    ego_matrix, static_matrix)

                static_speed = self._get_forward_speed(
                    transform=static_transform, velocity=static_velocity)

                # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
                if not lidar is None:
                    num_in_bbox_points = self.get_points_in_bbox(
                        relative_pos, relative_yaw, static_extent, lidar)
                else:
                    num_in_bbox_points = -1

                distance = np.linalg.norm(relative_pos)

                result = {
                    'class': 'static',
                    'extent': static_extent,
                    'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                    'yaw': relative_yaw,
                    'num_points': int(num_in_bbox_points),
                    'distance': distance,
                    'speed': static_speed,
                    'id': int(static_id),
                    'matrix': static_transform.get_matrix(),
                    'type_id': static.type_id,
                    'mesh_path': static.attributes['mesh_path'] if 'mesh_path' in static.attributes else None
                }
                results.append(result)

        stop_signs = self._actors.filter('*traffic.stop*')
        for stop_sign in stop_signs:
            if stop_sign.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
                stop_sign_extent = carla.Vector3D(1.5, 1.5, 0.5)
                stop_sign_transform = stop_sign.get_transform()
                stop_sign_rotation = stop_sign_transform.rotation
                stop_sign_matrix = np.array(stop_sign_transform.get_matrix())
                yaw = np.deg2rad(stop_sign_rotation.yaw)

                relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
                relative_pos = t_u.get_relative_transform(
                    ego_matrix, stop_sign_matrix)
                distance = np.linalg.norm(relative_pos)

                result = {
                    'class': 'stop_sign',
                    'extent': [stop_sign_extent.x, stop_sign_extent.y, stop_sign_extent.z],
                    'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                    'yaw': relative_yaw,
                    'distance': distance,
                    'id': int(stop_sign.id),
                    'affects_ego': True,
                    'matrix': stop_sign_transform.get_matrix()
                }
                results.append(result)

        return results

    def _get_forward_speed(self, transform=None, velocity=None):
        """
            Calculate the forward speed of the vehicle based on its transform and velocity.

            Args:
                transform (carla.Transform, optional): The transform of the vehicle. If not provided, it will be obtained from the vehicle.
                velocity (carla.Vector3D, optional): The velocity of the vehicle. If not provided, it will be obtained from the vehicle.

            Returns:
                float: The forward speed of the vehicle in m/s.
        """
        if not velocity:
            velocity = self._vehicle.get_velocity()

        if not transform:
            transform = self._vehicle.get_transform()

        # Convert the velocity vector to a NumPy array
        velocity_np = np.array([velocity.x, velocity.y, velocity.z])

        # Convert rotation angles from degrees to radians
        pitch_rad = np.deg2rad(transform.rotation.pitch)
        yaw_rad = np.deg2rad(transform.rotation.yaw)

        # Calculate the orientation vector based on pitch and yaw angles
        orientation_vector = np.array(
            [np.cos(pitch_rad) * np.cos(yaw_rad),
             np.cos(pitch_rad) * np.sin(yaw_rad),
             np.sin(pitch_rad)])

        # Calculate the forward speed by taking the dot product of velocity and orientation vectors
        forward_speed = np.dot(velocity_np, orientation_vector)

        return forward_speed

    def _vehicle_obstacle_detected(self,
                                   vehicle_list=None,
                                   max_distance=None,
                                   up_angle_th=90,
                                   low_angle_th=0,
                                   lane_offset=0):
        self._use_bbs_detection = False
        self._offset = 0

        def get_route_polygon():
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + \
                carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + \
                carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + \
                    carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                p2 = wp.transform.location + \
                    carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self.world_map.get_waypoint(
            ego_location, lane_type=carla.libcarla.LaneType.Any)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(self._vehicle.bounding_box.extent.x *
                                                       ego_transform.get_forward_vector())

        opposite_invasion = abs(
            self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self.world_map.get_waypoint(
                target_transform.location, lane_type=carla.LaneType.Any)

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(
                    target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return (True, target_vehicle.id, compute_distance(target_vehicle.get_location(), ego_location))

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[
                        0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance,
                                      [low_angle_th, up_angle_th]):
                    return (True, target_vehicle.id,
                            compute_distance(target_transform.location, ego_transform.location))

        return (False, None, -1)

    def get_points_in_bbox(self, vehicle_pos, vehicle_yaw, extent, lidar):
        """
            Checks for a given vehicle in ego coordinate system, how many LiDAR hit there are in its bounding box.
            :param vehicle_pos: Relative position of the vehicle w.r.t. the ego
            :param vehicle_yaw: Relative orientation of the vehicle w.r.t. the ego
            :param extent: List, Extent of the bounding box
            :param lidar: LiDAR point cloud
            :return: Returns the number of LiDAR hits within the bounding box of the
            vehicle
            """
        rotation_matrix = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0.0],
                                    [np.sin(vehicle_yaw), np.cos(vehicle_yaw), 0.0], [0.0, 0.0, 1.0]])

        # LiDAR in the with the vehicle as origin
        vehicle_lidar = (rotation_matrix.T @ (lidar - vehicle_pos).T).T

        # check points in bbox
        x, y, z = extent[0], extent[1], extent[2]
        num_points = ((vehicle_lidar[:, 0] < x) & (vehicle_lidar[:, 0] > -x) & (vehicle_lidar[:, 1] < y) &
                      (vehicle_lidar[:, 1] > -y) & (vehicle_lidar[:, 2] < z) & (vehicle_lidar[:, 2] > -z)).sum()
        return num_points

    def lidar_to_histogram_features(self, lidar, use_ground_plane=False):
        """
        Convert LiDAR point cloud into 2-bin histogram over a fixed size grid
        :param lidar: (N,3) numpy, LiDAR point cloud
        :param use_ground_plane, whether to use the ground plane
        :return: (2, H, W) numpy, LiDAR as sparse image
        """

        def splat_points(point_cloud):
            xbins = np.linspace(self.config.min_x, self.config.max_x,
                                (self.config.max_x - self.config.min_x) * int(self.config.pixels_per_meter) + 1)
            ybins = np.linspace(self.config.min_y, self.config.max_y,
                                (self.config.max_y - self.config.min_y) * int(self.config.pixels_per_meter) + 1)
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self.config.hist_max_per_pixel] = self.config.hist_max_per_pixel
            overhead_splat = hist / self.config.hist_max_per_pixel
            # The transpose here is an efficient axis swap.
            # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
            # (x height channel, y width channel)
            return overhead_splat.T

        # Remove points above the vehicle
        lidar = lidar[lidar[..., 2] < self.config.max_height_lidar]
        below = lidar[lidar[..., 2] <= self.config.lidar_split_height]
        above = lidar[lidar[..., 2] > self.config.lidar_split_height]
        below_features = splat_points(below)
        above_features = splat_points(above)
        if use_ground_plane:
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        return features

    def forecast_ego_agent(self, current_ego_transform, current_ego_speed, num_future_frames, ego_control):
        """
            Forecast the future states of the ego agent using the kinematic bicycle model and assume their is no hazard to
            check subsequently whether the ego vehicle would collide.
            Args:
                current_ego_transform (carla.Transform): The current transform of the ego vehicle.
                current_ego_speed (float): The current speed of the ego vehicle in m/s.
                num_future_frames (int): The number of future frames to forecast.
                ego_control(carla.VehicleControl): The control of ego car.

            Returns:
                list: A list of bounding boxes representing the future states of the ego vehicle.
            """
        location = np.array(
            [current_ego_transform.location.x, current_ego_transform.location.y, current_ego_transform.location.z])
        heading_angle = np.array(
            [np.deg2rad(current_ego_transform.rotation.yaw)])
        speed = np.array([current_ego_speed])
        action = np.array(
            [ego_control.steer, ego_control.throttle, ego_control.brake])

        future_bounding_boxes = []
        for _ in range(num_future_frames):
            location, heading_angle, speed = self.ego_model.forecast_ego_vehicle(
                location, heading_angle, speed, action)
            heading_deg = np.rad2deg(heading_angle).item()
            extent = self._vehicle.bounding_box.extent
            extent = carla.Vector3D(extent.x, extent.y, extent.z)
            if speed.item() < self.config.extent_ego_bbs_speed_threshold:
                extent.x *= self.config.slow_speed_extent_factor_ego
                extent.y *= self.config.slow_speed_extent_factor_ego
            else:
                extent.x *= self.config.high_speed_extent_factor_ego_x
                extent.y *= self.config.high_speed_extent_factor_ego_y

            transform = carla.Transform(
                carla.Location(x=location[0].item(), y=location[1].item(), z=location[2].item()))
            bbox = carla.BoundingBox(transform.location, extent)
            bbox.rotation = carla.Rotation(pitch=0, yaw=heading_deg, roll=0)
            future_bounding_boxes.append(bbox)
        return future_bounding_boxes

    def predict_other_actors_bounding_boxes_from_bev(self, pred_bb, ego_matrix, ego_yaw, num_future_frames,
                                                     near_lane_change):
        """
            Predict future bounding boxes of nearby vehicles based on BEV detection (in ego coords) and convert to world coords.
            IMPORTANT: Assumes pred_bb is in ego-relative coordinates and converts to world coordinates using ego_matrix.
            Args:
                pred_bb (np.ndarray): Detected bounding boxes from BEV model. Shape: [N, 8]
                ego_matrix (np.ndarray): 4x4 ego transformation matrix in world coordinates.
                num_future_frames (int): Number of frames to predict.

            Returns:
                Dict[int, List[carla.BoundingBox]]: Predicted bounding boxes per pseudo-vehicle ID.
            """
        predicted_bounding_boxes = {}
        static_bounding_boxes = []
        rear_vehicle_ids = set()
        nearby_actors = {}
        if pred_bb is None or len(pred_bb) == 0:
            return predicted_bounding_boxes, static_bounding_boxes, rear_vehicle_ids, nearby_actors
        # Filter vehicles only and within radius
        detection_radius = self.config.detection_radius
        ego_pos = ego_matrix[:3, 3]
        ego_forward_vector = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])

        for i, box in enumerate(pred_bb):
            x_rel, y_rel, w, h, yaw_rel, forward_speed, brake, cls = box
            if int(cls) not in self.config.Other_ACTORS_CLASSES:
                continue

            rel_pos = np.array([x_rel, y_rel, 0.0])
            world_pos = t_u.convert_relative_to_world(rel_pos, ego_matrix)
            world_yaw = t_u.normalize_angle(yaw_rel + ego_yaw)
            distance = np.linalg.norm(world_pos[:2] - ego_pos[:2])
            if distance > detection_radius:
                continue
            is_stationary = forward_speed < 0.1
            actor = {
                "position": world_pos,
                "yaw": world_yaw,
                "speed": forward_speed,
                "steer": 0.0,
                "throttle": 0.0 if is_stationary or brake > 0.6 else 0.3,
                "brake": 1.0 if is_stationary or brake > 0.6 else 0.0,
                "extent": np.array([w, h, 0.745139]),
            }
            nearby_actors[f"{i}"] = actor

            if is_stationary:
                static_bounding_boxes.append({
                    "actor_id": i,
                    "bbox": carla.BoundingBox(carla.Location(x=world_pos[0], y=world_pos[1], z=world_pos[2]),
                                              carla.Vector3D(w, h, 0.745139)),
                })

            actor_pos = actor["position"][:2]  # x, y
            direction_vector = actor_pos - ego_pos[:2]
            direction_norm = np.linalg.norm(direction_vector) + 1e-6
            dot = np.dot(direction_vector / direction_norm, ego_forward_vector)
            if dot < -0.5:
                rear_vehicle_ids.add(i)

        if len(nearby_actors) == 0:
            return predicted_bounding_boxes, static_bounding_boxes, rear_vehicle_ids, nearby_actors

        actions = np.array([[a["steer"], a["throttle"], a["brake"]]
                           for a in nearby_actors.values()])
        locations = np.array([a["position"] for a in nearby_actors.values()])
        headings = np.array([a["yaw"] for a in nearby_actors.values()])
        velocities = np.array([a["speed"] for a in nearby_actors.values()])
        extents = np.array([a["extent"] for a in nearby_actors.values()])

        future_locations = np.empty(
            (num_future_frames, len(nearby_actors), 3), dtype="float")
        future_headings = np.empty(
            (num_future_frames, len(nearby_actors)), dtype="float")
        future_velocities = np.empty(
            (num_future_frames, len(nearby_actors)), dtype="float")

        # Forecast the future locations, headings, and velocities for the nearby actors
        for i in range(num_future_frames):
            locations, headings, velocities = self.vehicle_model.forecast_other_vehicles(
                locations, headings, velocities, actions)
            future_locations[i] = locations.copy()
            future_headings[i] = headings.copy()
            future_velocities[i] = velocities.copy()

        future_headings_deg = np.rad2deg(future_headings)

        # Calculate the predicted bounding boxes for each nearby actor and future frame
        for actor_idx, actor_id in enumerate(nearby_actors.keys()):
            predicted_boxes = []
            base_extent = extents[actor_idx].copy()

            for t in range(num_future_frames):
                loc = future_locations[t, actor_idx]
                heading_deg = future_headings_deg[t, actor_idx]
                speed = future_velocities[t, actor_idx]

                extent = carla.Vector3D(
                    float(base_extent[0]),
                    float(base_extent[1]),
                    float(base_extent[2])
                )

                s = self.config.high_speed_min_extent_x_other_vehicle_lane_change if near_lane_change \
                    else self.config.high_speed_min_extent_x_other_vehicle
                extent.x *= self.config.slow_speed_extent_factor_ego if speed < self.config.extent_other_vehicles_bbs_speed_threshold else max(
                    s,
                    self.config.high_speed_min_extent_x_other_vehicle * float(t) / float(num_future_frames))

                extent.y *= self.config.slow_speed_extent_factor_ego if speed < self.config.extent_other_vehicles_bbs_speed_threshold else max(
                    self.config.high_speed_min_extent_y_other_vehicle,
                    self.config.high_speed_extent_y_factor_other_vehicle * float(t) / float(num_future_frames))
                location = carla.Location(x=loc[0], y=loc[1], z=loc[2])

                bbox = carla.BoundingBox(location, extent)

                bbox.rotation = carla.Rotation(
                    pitch=0, yaw=heading_deg, roll=0)

                predicted_boxes.append(bbox)

            predicted_bounding_boxes[actor_id] = predicted_boxes

        return predicted_bounding_boxes, static_bounding_boxes, rear_vehicle_ids, nearby_actors

    def forecast_walkers(self, pred_bb, ego_matrix, ego_yaw, num_future_frames):
        """
        Forecast the future locations of pedestrians in the vicinity of the ego vehicle assuming they
        move with constant velocity in their heading direction, and generate virtual pedestrians when
        obstructed by blocking vehicles.

        Args:
            pred_bb (np.ndarray): Detected bounding boxes from BEV model. Shape: [N, 8]
            ego_matrix (np.ndarray): 4x4 transformation matrix of the ego vehicle in world coordinates.
            ego_yaw (float): Ego vehicle's current yaw (heading) in radians.
            num_future_frames (int): Number of frames to forecast.
            near_lane_change (bool): Whether the ego vehicle is near a lane change (for safety considerations).

        Returns:
            tuple: A tuple containing:
                - List of future pedestrian bounding boxes for each pedestrian.
                - List of pedestrian IDs.
        """
        nearby_pedestrians_bbs = {}
        nearby_pedestrian_infos = {}
        virtual_pedestrians = {}

        if pred_bb is None or len(pred_bb) == 0:
            return nearby_pedestrians_bbs, nearby_pedestrian_infos
        # Filter vehicles only and within radius
        detection_radius = self.config.detection_radius
        ego_pos = ego_matrix[:3, 3]

        for i, box in enumerate(pred_bb):
            x_rel, y_rel, w, h, yaw_rel, forward_speed, brake, cls = box
            if int(cls) != 1:
                continue

            rel_pos = np.array([x_rel, y_rel, 0.0])
            world_pos = t_u.convert_relative_to_world(rel_pos, ego_matrix)
            world_yaw = t_u.normalize_angle(yaw_rel + ego_yaw)
            distance = np.linalg.norm(world_pos[:2] - ego_pos[:2])
            if distance > detection_radius:
                continue

            # Calculate pedestrian future locations assuming constant velocity
            pedestrian_speed = max(
                forward_speed, self.config.min_walker_speed)  # 0.5m/s
            pedestrian_direction = np.array(
                [np.cos(world_yaw), np.sin(world_yaw), 0])  # Heading direction

            # Forecast future pedestrian locations for the given number of frames
            future_pedestrian_locations = np.empty(
                (num_future_frames, 3), dtype="float")
            for t in range(num_future_frames):
                future_pedestrian_locations[
                    t] = world_pos + t * pedestrian_direction * pedestrian_speed / self.config.bicycle_frame_rate

            # Create bounding boxes for future pedestrian positions
            pedestrian_future_bboxes = []
            extent = carla.Vector3D(w, h, 0.745139)
            extent.x = max(self.config.pedestrian_minimum_extent,
                           extent.x)  # Ensure a minimum width
            extent.y = max(self.config.pedestrian_minimum_extent, extent.y)
            for t in range(num_future_frames):
                location = carla.Location(future_pedestrian_locations[t, 0], future_pedestrian_locations[t, 1],
                                          future_pedestrian_locations[t, 2])
                # Ensure a minimum length
                bbox = carla.BoundingBox(location, extent)
                bbox.rotation = carla.Rotation(pitch=0, yaw=world_yaw, roll=0)
                pedestrian_future_bboxes.append(bbox)

            walker_id = f"walker_{i}"
            nearby_pedestrians_bbs[walker_id] = pedestrian_future_bboxes
            nearby_pedestrian_infos[walker_id] = {
                "position": world_pos,
                "yaw": world_yaw,
                "speed": pedestrian_speed,
                "steer": 0.0,
                "throttle": 0.0,
                "brake": 0.0,
                "extent": np.array([extent.x, extent.y, extent.z])
            }

        return nearby_pedestrians_bbs, nearby_pedestrian_infos

    def construct_stop_sign_bboxes(self, bounding_boxes, ego_matrix, ego_yaw, ego_speed):
        stop_sign_bboxes = [bb for bb in bounding_boxes if int(
            bb[7]) == 3]  # type_id==3 stop_sign
        future_stop_sign_bboxes = {}
        collision_actor_infos = {}
        ego_pos = ego_matrix[:3, 3]

        for bb in stop_sign_bboxes:
            x_rel, y_rel, w, h, yaw_rel = bb[:5]
            rel_pos = np.array([x_rel, y_rel, 0.0])
            world_pos = t_u.convert_relative_to_world(rel_pos, ego_matrix)
            world_yaw = t_u.normalize_angle(yaw_rel + ego_yaw)
            distance = np.linalg.norm(world_pos[:2] - ego_pos[:2])
            stop_sign_id = (int(world_pos[0] // 1), int(world_pos[1] // 1))
            if stop_sign_id not in self.pending_stop_signs:
                self.pending_stop_signs[stop_sign_id] = {
                    "waiting_ticks": 0, "cleared": False}
            state = self.pending_stop_signs[stop_sign_id]

            if state["cleared"]:
                continue

            if ego_speed < 0.1 and distance < 6:
                state["waiting_ticks"] += 1
                # print("[Debug] ininin",flush=True)
                if state["waiting_ticks"] >= 5:  # 0.5s
                    state["cleared"] = True
            else:
                state["waiting_ticks"] = 0

            if not state["cleared"] and distance < self.config.unclearing_distance_to_stop_sign:
                extent = carla.Vector3D(1.5, 1.5, 0.5)
                location = carla.Location(
                    world_pos[0], world_pos[1], ego_pos[2])
                bbox = carla.BoundingBox(location, extent)
                bbox.rotation = carla.Rotation(yaw=np.rad2deg(world_yaw))
                stop_sign_actor_id = f"stop_sign_{stop_sign_id}"
                future_stop_sign_bboxes[stop_sign_actor_id] = bbox

                collision_actor_infos[stop_sign_actor_id] = {
                    "position": world_pos,
                    "yaw": world_yaw,
                    "speed": 0.0,
                    "brake": 1.0,
                    "throttle": 0.0,
                    "steer": 0.0,
                    "extent": np.array([w, h, 0.5])
                }

        return future_stop_sign_bboxes, collision_actor_infos

    def is_near_lane_change_from_pred_wp(self, pred_wp, lateral_threshold=1.0):
        if len(pred_wp) < 2:
            return False
        direction = pred_wp[-1] - pred_wp[0]
        direction_norm = direction / (np.linalg.norm(direction) + 1e-6)

        lateral_dir = np.array([-direction_norm[1], direction_norm[0]])
        rel_offsets = pred_wp - pred_wp[0]
        lateral_offsets = np.abs(np.dot(rel_offsets, lateral_dir))

        return np.any(lateral_offsets > lateral_threshold)

    def compute_collisions_from_bev_forecast(self, ego_bounding_boxes, predicted_bounding_boxes, nearby_pedestrians,
                                             stop_sign_bboxes, near_lane_change, rear_vehicle_ids):
        collisions = []
        collided_actor_ids = set()

        for i, ego_bb in enumerate(ego_bounding_boxes):
            for actor_id, bbs in predicted_bounding_boxes.items():
                if actor_id in collided_actor_ids:
                    continue

                if i >= len(bbs):
                    continue

                if not near_lane_change and actor_id in rear_vehicle_ids:
                    continue

                target_bb = bbs[i]
                if t_u.check_obb_intersection(ego_bb, target_bb):
                    info = {
                        "frame_idx": i,
                        "actor_id": actor_id,
                        "type_id": "vehicle",
                        "bbox": target_bb,
                    }
                    collisions.append(info)
                    collided_actor_ids.add(actor_id)

            for walker_id, ped_bbs in nearby_pedestrians.items():
                if walker_id in collided_actor_ids:
                    continue

                if i >= len(ped_bbs):
                    continue

                target_bb = ped_bbs[i]
                if t_u.check_obb_intersection(ego_bb, target_bb):
                    info = {
                        "frame_idx": i,
                        "actor_id": walker_id,
                        "type_id": "pedestrian",
                        "bbox": target_bb,
                    }
                    collisions.append(info)
                    collided_actor_ids.add(walker_id)

            for stop_sign_id, ss_bb in stop_sign_bboxes.items():
                if stop_sign_id in collided_actor_ids:
                    continue
                if t_u.check_obb_intersection(ego_bb, ss_bb):
                    collisions.append({
                        "frame_idx": i,
                        "actor_id": stop_sign_id,
                        "type_id": "stop_sign",
                        "bbox": ss_bb,
                    })
                    collided_actor_ids.add(stop_sign_id)

        return collisions

    def kbm_based_collision_risk(self, pred_wp, pred_bb, control):
        ego_transform = self._vehicle.get_transform()
        ego_control = control
        # ego_control = self._vehicle.get_control()
        ego_velocity = self._vehicle.get_velocity()
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_rotation = ego_transform.rotation
        ego_speed = self._get_forward_speed(
            transform=ego_transform, velocity=ego_velocity)

        # Compute if there will be a lane change close
        near_lane_change = self.is_near_lane_change_from_pred_wp(pred_wp)
        # normal 40 frame， 2s
        num_future_frames = int(
            self.config.bicycle_frame_rate *
            (self.config.forecast_length_lane_change if near_lane_change else self.config.default_forecast_length))

        # Forecast the ego vehicle's bounding boxes for the future frames
        ego_bounding_boxes = self.forecast_ego_agent(ego_transform, ego_speed, num_future_frames,
                                                     ego_control)
        # Predict bounding boxes of other actors (vehicles, bicycles, etc.)
        predicted_bounding_boxes, static_bounding_boxes, rear_vehicle_ids, other_actor_infos = self.predict_other_actors_bounding_boxes_from_bev(
            pred_bb, ego_matrix,
            np.deg2rad(ego_rotation.yaw), num_future_frames, near_lane_change)

        # Get future bounding boxes of pedestrians
        nearby_pedestrians, nearby_pedestrian_infos = self.forecast_walkers(pred_bb, ego_matrix,
                                                                            np.deg2rad(
                                                                                ego_rotation.yaw),
                                                                            num_future_frames)

        stop_sign_bboxes, stop_sign_actors = self.construct_stop_sign_bboxes(pred_bb, ego_matrix,
                                                                             np.deg2rad(ego_rotation.yaw), ego_speed)

        collisions = self.compute_collisions_from_bev_forecast(
            ego_bounding_boxes=ego_bounding_boxes,
            predicted_bounding_boxes=predicted_bounding_boxes,
            nearby_pedestrians=nearby_pedestrians,
            stop_sign_bboxes=stop_sign_bboxes,
            near_lane_change=near_lane_change,
            rear_vehicle_ids=rear_vehicle_ids
        )

        all_actor_infos = {**other_actor_infos, **
                           nearby_pedestrian_infos, **stop_sign_actors}

        if collisions:
            collision = min(collisions, key=lambda x: x["frame_idx"])
            collision_time = collision["frame_idx"] / self.config.time_step_inv
            collision_bbox = collision["bbox"]

            collision_actor_ids = set([c["actor_id"] for c in collisions])
            collision_actors = {
                aid: info for aid, info in all_actor_infos.items() if aid in collision_actor_ids}

            print(
                f"[Collision] Estimated collision at {collision_time:.2f}s with {collision_bbox}", flush=True)
            return static_bounding_boxes, collision_time, collision_bbox, collision_actors

        else:
            return static_bounding_boxes, None, None, None

    def is_near_lane_change_idm(self, ego_velocity, route_points):
        """
            Computes if the ego agent is/was close to a lane change maneuver.

            Args:
                ego_velocity (float): The current velocity of the ego agent in m/s.
                route_points (numpy.ndarray): An array of locations representing the planned route.

            Returns:
                bool: True if the ego agent is close to a lane change, False otherwise.
            """
        # Calculate the braking distance based on the ego velocity
        braking_distance = ((
            (
                ego_velocity * 3.6) / 10.0) ** 2 / 2.0) + self.config.braking_distance_calculation_safety_distance

        # Determine the number of waypoints to look ahead based on the braking distance
        look_ahead_points = max(self.config.minimum_lookahead_distance_to_compute_near_lane_change,
                                min(route_points.shape[0], self.config.points_per_meter * int(braking_distance)))
        current_route_index = self._waypoint_planner.route_index
        max_route_length = len(self._waypoint_planner.commands)

        from_index = max(0, current_route_index -
                         self.config.check_previous_distance_for_lane_change)
        to_index = min(max_route_length - 1,
                       current_route_index + look_ahead_points)
        # Iterate over the points around the current position, checking for lane change commands
        for i in range(from_index, to_index, 1):
            if self._waypoint_planner.commands[i] in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                return True

        return False

    def _compute_target_speed_idm(self,
                                  desired_speed,
                                  leading_actor_length,
                                  ego_speed,
                                  leading_actor_speed,
                                  distance_to_leading_actor,
                                  s0=4.,
                                  T=0.5):
        """
            Compute the target speed for the ego vehicle using the Intelligent Driver Model (IDM).

            Args:
                desired_speed (float): The desired speed of the ego vehicle.
                leading_actor_length (float): The length of the leading actor (vehicle or obstacle).
                ego_speed (float): The current speed of the ego vehicle.
                leading_actor_speed (float): The speed of the leading actor.
                distance_to_leading_actor (float): The distance to the leading actor.
                s0 (float, optional): The minimum desired net distance.
                T (float, optional): The desired time headway.

            Returns:
                float: The computed target speed for the ego vehicle.
            """

        a = self.config.idm_maximum_acceleration  # Maximum acceleration [m/s²]
        b = self.config.idm_comfortable_braking_deceleration_high_speed if ego_speed > \
            self.config.idm_comfortable_braking_deceleration_threshold else \
            self.config.idm_comfortable_braking_deceleration_low_speed  # Comfortable deceleration [m/s²]
        delta = self.config.idm_acceleration_exponent  # Acceleration exponent

        t_bound = self.config.idm_t_bound

        def idm_equations(t, x):
            """
                  Differential equations for the Intelligent Driver Model.

                  Args:
                      t (float): Time.
                      x (list): State variables [position, speed].

                  Returns:
                      list: Derivatives of the state variables.
                  """
            ego_position, ego_speed = x

            speed_diff = ego_speed - leading_actor_speed
            s_star = s0 + ego_speed * T + ego_speed * \
                speed_diff / 2. / np.sqrt(a * b)
            # The maximum is needed to avoid numerical unstabilities
            s = max(0.1, distance_to_leading_actor + t *
                    leading_actor_speed - ego_position - leading_actor_length)
            dvdt = a * (1. - (ego_speed / desired_speed)
                        ** delta - (s_star / s) ** 2)

            return [ego_speed, dvdt]

        # Set the initial conditions
        y0 = [0., ego_speed]

        # Integrate the differential equations using RK45
        rk45 = RK45(fun=idm_equations, t0=0., y0=y0, t_bound=t_bound)
        while rk45.status == "running":
            rk45.step()

        # The target speed is the final speed obtained from the integration
        target_speed = rk45.y[1]

        # Clip the target speed to non-negative values
        return np.clip(target_speed, 0, np.inf)

    def _get_angle_to(self, current_position, current_heading, target_position):
        """
            Calculate the angle (in degrees) from the current position and heading to a target position.

            Args:
                current_position (list): A list of (x, y) coordinates representing the current position.
                current_heading (float): The current heading angle in radians.
                target_position (tuple or list): A tuple or list of (x, y) coordinates representing the target position.

            Returns:
                float: The angle (in degrees) from the current position and heading to the target position.
            """
        cos_heading = math.cos(current_heading)
        sin_heading = math.sin(current_heading)

        # Calculate the vector from the current position to the target position
        position_delta = target_position - current_position

        # Calculate the dot product of the position delta vector and the current heading vector
        aim_x = cos_heading * \
            position_delta[0] + sin_heading * position_delta[1]
        aim_y = -sin_heading * \
            position_delta[0] + cos_heading * position_delta[1]

        # Calculate the angle (in radians) from the current heading to the target position
        angle_radians = -math.atan2(-aim_y, aim_x)

        # Convert the angle from radians to degrees
        angle_degrees = np.float_(math.degrees(angle_radians))

        return angle_degrees

    def _get_steer(self, route_points, current_position, current_heading, current_speed):
        """
            Calculate the steering angle based on the current position, heading, speed, and the route points.

            Args:
                route_points (numpy.ndarray): An array of (x, y) coordinates representing the route points.
                current_position (tuple): The current position (x, y) of the vehicle.
                current_heading (float): The current heading angle (in radians) of the vehicle.
                current_speed (float): The current speed of the vehicle (in m/s).

            Returns:
                float: The calculated steering angle.
            """
        speed_scale = self.config.lateral_pid_speed_scale
        speed_offset = self.config.lateral_pid_speed_offset

        # Calculate the lookahead index based on the current speed
        speed_in_kmph = current_speed * 3.6
        lookahead_distance = speed_scale * speed_in_kmph + speed_offset
        lookahead_distance = np.clip(lookahead_distance, self.config.lateral_pid_default_lookahead,
                                     self.config.lateral_pid_maximum_lookahead_distance)
        lookahead_index = int(
            min(lookahead_distance, route_points.shape[0] - 1))

        # Get the target point from the route points
        target_point = route_points[lookahead_index]

        # Calculate the angle between the current heading and the target point
        angle_unnorm = self._get_angle_to(
            current_position, current_heading, target_point)
        normalized_angle = angle_unnorm / 90

        # Calculate the steering angle using the turn controller
        steering_angle = self._turn_controller.step(
            route_points, current_speed, current_position, current_heading)
        steering_angle = round(steering_angle, 3)

        return steering_angle

    def get_brake_and_target_speed(self, route_points, distance_to_next_traffic_light, next_traffic_light,
                                   distance_to_next_stop_sign, next_stop_sign, vehicle_list, actor_list,
                                   initial_target_speed):
        """
            Compute the brake command and target speed for the ego vehicle based on various factors.

            Args:
                route_points (numpy.ndarray): An array of waypoints representing the planned route.
                distance_to_next_traffic_light (float): The distance to the next traffic light.
                next_traffic_light (carla.TrafficLight): The next traffic light actor.
                distance_to_next_stop_sign (float): The distance to the next stop sign.
                next_stop_sign (carla.StopSign): The next stop sign actor.
                vehicle_list (list): A list of vehicle actors in the simulation.
                actor_list (list): A list of all actors (vehicles, pedestrians, etc.) in the simulation.
                initial_target_speed (float): The initial target speed for the ego vehicle.

            Returns:
                tuple: A tuple containing the brake command (bool), target speed (float)
            """
        ego_speed = self._vehicle.get_velocity().length()
        target_speed = initial_target_speed

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_transform = self._vehicle.get_transform()

        # Calculate the global bounding box of the ego vehicle
        center_ego_bb_global = ego_vehicle_transform.transform(
            self._vehicle.bounding_box.location)
        ego_bb_global = carla.BoundingBox(
            center_ego_bb_global, self._vehicle.bounding_box.extent)
        ego_bb_global.rotation = ego_vehicle_transform.rotation

        # Reset hazard flags
        self.stop_sign_close = False
        self.walker_close = False
        self.walker_close_id = None
        self.vehicle_hazard = False
        self.vehicle_affecting_id = None
        self.walker_hazard = False
        self.walker_affecting_id = None
        self.traffic_light_hazard = False
        self.stop_sign_hazard = False
        self.walker_hazard = False
        self.stop_sign_close = False

        # Compute if there will be a lane change close
        near_lane_change = self.is_near_lane_change_idm(
            ego_speed, route_points)

        # Compute the number of future frames to consider for collision detection
        num_future_frames = int(
            self.config.bicycle_frame_rate *
            (self.config.forecast_length_lane_change if near_lane_change else self.config.default_forecast_length))

        # Get future bounding boxes of pedestrians
        nearby_pedestrians, nearby_pedestrian_ids = self.forecast_walkers_idm(actor_list, ego_vehicle_location,
                                                                              num_future_frames)

        # Forecast the ego vehicle's bounding boxes for the future frames
        ego_bounding_boxes = self.forecast_ego_agent_idm(ego_vehicle_transform, ego_speed, num_future_frames,
                                                         initial_target_speed, route_points)

        # Predict bounding boxes of other actors (vehicles, bicycles, etc.)
        predicted_bounding_boxes = self.predict_other_actors_bounding_boxes_idm(vehicle_list, ego_vehicle_location,
                                                                                num_future_frames, near_lane_change)

        blacklist = {"static.prop.dirtdebris01", "static.prop.dirtdebris02"}
        static_obstacles = [
            actor for actor in actor_list.filter('*static*')
            if 'dirtdebris' not in actor.type_id
        ]
        for ob in static_obstacles:
            if ob.get_location().distance(self._vehicle.get_location()) < self.config.static_obstacle_radius:
                bb_location = ob.get_transform().transform(ob.bounding_box.location)
                extent = ob.bounding_box.extent
                scaled_extent = carla.Vector3D(
                    x=extent.x * 0.7,
                    y=extent.y * 0.7,
                    z=extent.z * 0.7
                )

                bb = carla.BoundingBox(bb_location, scaled_extent)
                bb.rotation = ob.get_transform().rotation
                static_obstacle_bbs.append(bb)

        # Compute the leading and trailing vehicle IDs
        leading_vehicle_ids = self._waypoint_planner.compute_leading_vehicles(
            vehicle_list, self._vehicle.id)
        trailing_vehicle_ids = self._waypoint_planner.compute_trailing_vehicles(
            vehicle_list, self._vehicle.id)

        # Compute the target speed with respect to the leading vehicle
        target_speed_leading = self.compute_target_speed_wrt_leading_vehicle_idm(
            initial_target_speed, predicted_bounding_boxes, near_lane_change, ego_vehicle_location,
            trailing_vehicle_ids, leading_vehicle_ids)

        # Compute the target speeds with respect to all actors (vehicles, bicycles, pedestrians)
        target_speed_bicycle, target_speed_pedestrian, target_speed_vehicle, target_speed_static = \
            self.compute_target_speeds_wrt_all_actors_idm(initial_target_speed, ego_bounding_boxes,
                                                          predicted_bounding_boxes, near_lane_change,
                                                          leading_vehicle_ids,
                                                          trailing_vehicle_ids,
                                                          nearby_pedestrians, nearby_pedestrian_ids,
                                                          static_obstacle_bbs)

        # Compute the target speed with respect to the red light
        target_speed_red_light = self.ego_agent_affected_by_red_light_idm(ego_vehicle_location, ego_speed,
                                                                          distance_to_next_traffic_light,
                                                                          next_traffic_light,
                                                                          route_points, initial_target_speed)

        # Compute the target speed with respect to the stop sign
        target_speed_stop_sign = self.ego_agent_affected_by_stop_sign_idm(ego_vehicle_location, ego_speed,
                                                                          next_stop_sign,
                                                                          initial_target_speed, actor_list)

        # Compute the minimum target speed considering all factors
        target_speed = min(target_speed_leading, target_speed_bicycle, target_speed_vehicle, target_speed_pedestrian,
                           target_speed_red_light, target_speed_stop_sign, target_speed_static)

        # Determine if the ego vehicle needs to brake based on the target speed
        brake = target_speed == 0
        return brake, target_speed

    def ego_agent_affected_by_red_light_idm(self, ego_vehicle_location, ego_vehicle_speed, distance_to_traffic_light,
                                            next_traffic_light, route_points, target_speed):
        """
            Handles the behavior of the ego vehicle when approaching a traffic light.

            Args:
                ego_vehicle_location (carla.Location): The ego vehicle location.
                ego_vehicle_speed (float): The current speed of the ego vehicle in m/s.
                distance_to_traffic_light (float): The distance from the ego vehicle to the next traffic light.
                next_traffic_light (carla.TrafficLight or None): The next traffic light in the route.
                route_points (numpy.ndarray): An array of (x, y, z) coordinates representing the route.
                target_speed (float): The target speed for the ego vehicle.

            Returns:
                float: The adjusted target speed for the ego vehicle.
            """

        for light, center, waypoints in self.list_traffic_lights:

            center_loc = carla.Location(center)
            if center_loc.distance(ego_vehicle_location) > self.config.light_radius:
                continue

            for wp in waypoints:
                # * 0.9 to make the box slightly smaller than the street to prevent overlapping boxes.
                length_bounding_box = carla.Vector3D((wp.lane_width / 2.0) * 0.9, light.trigger_volume.extent.y,
                                                     light.trigger_volume.extent.z)
                length_bounding_box = carla.Vector3D(1.5, 1.5, 0.5)

                bounding_box = carla.BoundingBox(
                    wp.transform.location, length_bounding_box)

                gloabl_rot = light.get_transform().rotation
                bounding_box.rotation = carla.Rotation(
                    pitch=gloabl_rot.pitch, yaw=gloabl_rot.yaw, roll=gloabl_rot.roll)

                affects_ego = next_traffic_light is not None and light.id == next_traffic_light.id

        if next_traffic_light is None or next_traffic_light.state == carla.TrafficLightState.Green:
            # No traffic light or green light, continue with the current target speed
            return target_speed

        # Compute the target speed using the IDM
        target_speed = self._compute_target_speed_idm(desired_speed=target_speed,
                                                      leading_actor_length=0.,
                                                      ego_speed=ego_vehicle_speed,
                                                      leading_actor_speed=0.,
                                                      distance_to_leading_actor=distance_to_traffic_light,
                                                      s0=self.config.idm_red_light_minimum_distance,
                                                      T=self.config.idm_red_light_desired_time_headway)

        return target_speed

    def ego_agent_affected_by_stop_sign_idm(self, ego_vehicle_location, ego_vehicle_speed, next_stop_sign, target_speed,
                                            actor_list):
        """
            Handles the behavior of the ego vehicle when approaching a stop sign.

            Args:
                ego_vehicle_location (carla.Location): The location of the ego vehicle.
                ego_vehicle_speed (float): The current speed of the ego vehicle in m/s.
                next_stop_sign (carla.TrafficSign or None): The next stop sign in the route.
                target_speed (float): The target speed for the ego vehicle.
                actor_list (list): A list of all actors (vehicles, pedestrians, etc.) in the simulation.

            Returns:
                float: The adjusted target speed for the ego vehicle.
            """
        stop_signs = self.get_nearby_object(ego_vehicle_location, actor_list.filter('*traffic.stop*'),
                                            self.config.light_radius)

        for stop_sign in stop_signs:
            center_bb_stop_sign = stop_sign.get_transform(
            ).transform(stop_sign.trigger_volume.location)
            stop_sign_extent = carla.Vector3D(1.5, 1.5, 0.5)
            bounding_box_stop_sign = carla.BoundingBox(
                center_bb_stop_sign, stop_sign_extent)
            rotation_stop_sign = stop_sign.get_transform().rotation
            bounding_box_stop_sign.rotation = carla.Rotation(pitch=rotation_stop_sign.pitch,
                                                             yaw=rotation_stop_sign.yaw,
                                                             roll=rotation_stop_sign.roll)

            affects_ego = (
                next_stop_sign is not None and next_stop_sign.id == stop_sign.id and not self.cleared_stop_sign)

        if next_stop_sign is None:
            # No stop sign, continue with the current target speed
            return target_speed

        # Calculate the accurate distance to the stop sign
        distance_to_stop_sign = next_stop_sign.get_transform().transform(next_stop_sign.trigger_volume.location) \
            .distance(ego_vehicle_location)

        # Reset the stop sign flag if we are farther than 10m away
        if distance_to_stop_sign > self.config.unclearing_distance_to_stop_sign:
            self.cleared_stop_sign = False
            self.waiting_ticks_at_stop_sign = 0
        else:
            # Set the stop sign flag if we are closer than 3m and speed is low enough
            if ego_vehicle_speed < 0.1 and distance_to_stop_sign < self.config.clearing_distance_to_stop_sign:
                self.waiting_ticks_at_stop_sign += 1
                if self.waiting_ticks_at_stop_sign > 15:
                    self.cleared_stop_sign = True
            else:
                self.waiting_ticks_at_stop_sign = 0

        # Set the distance to stop sign as infinity if the stop sign has been cleared
        distance_to_stop_sign = np.inf if self.cleared_stop_sign else distance_to_stop_sign

        # Compute the target speed using the IDM
        target_speed = self._compute_target_speed_idm(desired_speed=target_speed,
                                                      leading_actor_length=0,
                                                      ego_speed=ego_vehicle_speed,
                                                      leading_actor_speed=0.,
                                                      distance_to_leading_actor=distance_to_stop_sign,
                                                      s0=self.config.idm_stop_sign_minimum_distance,
                                                      T=self.config.idm_stop_sign_desired_time_headway)

        # Return whether the ego vehicle is affected by the stop sign and the adjusted target speed
        return target_speed

    def compute_target_speed_wrt_leading_vehicle_idm(self, initial_target_speed, predicted_bounding_boxes,
                                                     near_lane_change,
                                                     ego_location, rear_vehicle_ids, leading_vehicle_ids):
        """
            Compute the target speed for the ego vehicle considering the leading vehicle.

            Args:
                initial_target_speed (float): The initial target speed for the ego vehicle.
                predicted_bounding_boxes (dict): A dictionary mapping actor IDs to lists of predicted bounding boxes.
                near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.
                ego_location (carla.Location): The current location of the ego vehicle.
                rear_vehicle_ids (list): A list of IDs for vehicles behind the ego vehicle.
                leading_vehicle_ids (list): A list of IDs for vehicles in front of the ego vehicle.

            Returns:
                float: The target speed considering the leading vehicle.
            """
        target_speed_wrt_leading_vehicle = initial_target_speed

        for vehicle_id, _ in predicted_bounding_boxes.items():
            if vehicle_id in leading_vehicle_ids and not near_lane_change:
                # Vehicle is in front of the ego vehicle
                ego_speed = self._vehicle.get_velocity().length()
                vehicle = self._world.get_actor(vehicle_id)
                other_speed = vehicle.get_velocity().length()
                distance_to_vehicle = ego_location.distance(
                    vehicle.get_location())

                # Compute the target speed using the IDM
                target_speed_wrt_leading_vehicle = min(
                    target_speed_wrt_leading_vehicle,
                    self._compute_target_speed_idm(desired_speed=initial_target_speed,
                                                   leading_actor_length=vehicle.bounding_box.extent.x * 2,
                                                   ego_speed=ego_speed,
                                                   leading_actor_speed=other_speed,
                                                   distance_to_leading_actor=distance_to_vehicle,
                                                   s0=self.config.idm_leading_vehicle_minimum_distance,
                                                   T=self.config.idm_leading_vehicle_time_headway))

        return target_speed_wrt_leading_vehicle

    def compute_target_speeds_wrt_all_actors_idm(self, initial_target_speed, ego_bounding_boxes,
                                                 predicted_bounding_boxes,
                                                 near_lane_change, leading_vehicle_ids, rear_vehicle_ids,
                                                 nearby_walkers, nearby_walkers_ids, nearby_static_obstacles):
        """
            Compute the target speeds for the ego vehicle considering all actors (vehicles, bicycles,
            and pedestrians) by checking for intersecting bounding boxes.

            Args:
                initial_target_speed (float): The initial target speed for the ego vehicle.
                ego_bounding_boxes (list): A list of bounding boxes for the ego vehicle at different future frames.
                predicted_bounding_boxes (dict): A dictionary mapping actor IDs to lists of predicted bounding boxes.
                near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.
                leading_vehicle_ids (list): A list of IDs for vehicles in front of the ego vehicle.
                rear_vehicle_ids (list): A list of IDs for vehicles behind the ego vehicle.
                nearby_walkers (dict): A list of predicted bounding boxes of nearby pedestrians.
                nearby_walkers_ids (list): A list of IDs for nearby pedestrians.

            Returns:
                tuple: A tuple containing the target speeds for bicycles, pedestrians, vehicles
            """
        target_speed_bicycle = initial_target_speed
        target_speed_pedestrian = initial_target_speed
        target_speed_vehicle = initial_target_speed
        target_speed_static = initial_target_speed
        ego_vehicle_location = self._vehicle.get_location()
        hazard_color = self.config.ego_vehicle_forecasted_bbs_hazard_color
        normal_color = self.config.ego_vehicle_forecasted_bbs_normal_color
        color = normal_color

        # Iterate over the ego vehicle's bounding boxes and predicted bounding boxes of other actors
        for i, ego_bounding_box in enumerate(ego_bounding_boxes):
            for vehicle_id, bounding_boxes in predicted_bounding_boxes.items():
                # Skip leading and rear vehicles if not near a lane change
                if vehicle_id in leading_vehicle_ids and not near_lane_change:
                    continue
                elif vehicle_id in rear_vehicle_ids and not near_lane_change:
                    continue
                else:
                    # Check if the ego bounding box intersects with the predicted bounding box of the actor
                    intersects_with_ego = t_u.check_obb_intersection(
                        ego_bounding_box, bounding_boxes[i])
                    ego_speed = self._vehicle.get_velocity().length()

                    if intersects_with_ego:
                        blocking_actor = self._world.get_actor(vehicle_id)

                        # Handle the case when the blocking actor is a bicycle
                        if "base_type" in blocking_actor.attributes and blocking_actor.attributes[
                                "base_type"] == "bicycle":
                            other_speed = blocking_actor.get_velocity().length()
                            distance_to_actor = ego_vehicle_location.distance(
                                blocking_actor.get_location())

                            # Compute the target speed for bicycles using the IDM
                            target_speed_bicycle = min(
                                target_speed_bicycle,
                                self._compute_target_speed_idm(desired_speed=initial_target_speed,
                                                               leading_actor_length=6 + blocking_actor.bounding_box.extent.x * 2,
                                                               ego_speed=ego_speed,
                                                               leading_actor_speed=other_speed,
                                                               distance_to_leading_actor=distance_to_actor,
                                                               s0=self.config.idm_bicycle_minimum_distance,
                                                               T=self.config.idm_bicycle_desired_time_headway))

                        # Handle the case when the blocking actor is not a bicycle
                        else:
                            target_speed_vehicle = 0  # Set the target speed for vehicles to zero

            # Iterate over nearby pedestrians and check for intersections with the ego bounding box
            for pedestrian_bb, pedestrian_id in zip(nearby_walkers, nearby_walkers_ids):
                if t_u.check_obb_intersection(ego_bounding_box, pedestrian_bb[i]):
                    color = hazard_color
                    ego_speed = self._vehicle.get_velocity().length()
                    blocking_actor = self._world.get_actor(pedestrian_id)
                    distance_to_actor = ego_vehicle_location.distance(
                        blocking_actor.get_location())

                    # Compute the target speed for pedestrians using the IDM
                    target_speed_pedestrian = min(
                        target_speed_pedestrian,
                        self._compute_target_speed_idm(desired_speed=initial_target_speed,
                                                       leading_actor_length=0.5 + self._vehicle.bounding_box.extent.x,
                                                       ego_speed=ego_speed,
                                                       leading_actor_speed=0.,
                                                       distance_to_leading_actor=distance_to_actor,
                                                       s0=self.config.idm_pedestrian_minimum_distance,
                                                       T=self.config.idm_pedestrian_desired_time_headway))
            for static_bb in nearby_static_obstacles:
                if t_u.check_obb_intersection(ego_bounding_box, static_bb):
                    # print("check_obb_intersection", flush=True)
                    ego_speed = self._vehicle.get_velocity().length()
                    distance_to_static = ego_vehicle_location.distance(
                        static_bb.location)

                    target_speed_static = min(
                        target_speed_static,
                        self._compute_target_speed_idm(
                            desired_speed=initial_target_speed,
                            leading_actor_length=6 + static_bb.extent.x * 2,
                            ego_speed=ego_speed,
                            leading_actor_speed=0.0,
                            distance_to_leading_actor=distance_to_static,
                            s0=self.config.idm_static_minimum_distance,
                            T=self.config.idm_static_desired_time_headway
                        )
                    )

        return target_speed_bicycle, target_speed_pedestrian, target_speed_vehicle, target_speed_static

    def forecast_ego_agent_idm(self, current_ego_transform, current_ego_speed, num_future_frames, initial_target_speed,
                               route_points):
        """
            Forecast the future states of the ego agent using the kinematic bicycle model and assume their is no hazard to
            check subsequently whether the ego vehicle would collide.

            Args:
                current_ego_transform (carla.Transform): The current transform of the ego vehicle.
                current_ego_speed (float): The current speed of the ego vehicle in m/s.
                num_future_frames (int): The number of future frames to forecast.
                initial_target_speed (float): The initial target speed for the ego vehicle.
                route_points (numpy.ndarray): An array of waypoints representing the planned route.

            Returns:
                list: A list of bounding boxes representing the future states of the ego vehicle.
            """
        self._turn_controller.save_state()
        self._waypoint_planner.save()

        # Initialize the initial state without braking
        location = np.array(
            [current_ego_transform.location.x, current_ego_transform.location.y, current_ego_transform.location.z])
        heading_angle = np.array(
            [np.deg2rad(current_ego_transform.rotation.yaw)])
        speed = np.array([current_ego_speed])

        # Calculate the throttle command based on the target speed and current speed
        throttle = self._longitudinal_controller.get_throttle_extrapolation(
            initial_target_speed, current_ego_speed)
        steering = self._turn_controller.step(
            route_points, speed, location, heading_angle.item())
        action = np.array([steering, throttle, 0.0]).flatten()

        future_bounding_boxes = []
        # Iterate over the future frames and forecast the ego agent's state
        for _ in range(num_future_frames):
            # Forecast the next state using the kinematic bicycle model
            location, heading_angle, speed = self.ego_model.forecast_ego_vehicle(
                location, heading_angle, speed, action)

            # Update the route and extrapolate steering and throttle commands
            extrapolated_route, _, _, _, _, _, _ = self._waypoint_planner.run_step(
                location)
            steering = self._turn_controller.step(
                extrapolated_route, speed, location, heading_angle.item())
            throttle = self._longitudinal_controller.get_throttle_extrapolation(
                initial_target_speed, speed)
            action = np.array([steering, throttle, 0.0]).flatten()

            heading_angle_degrees = np.rad2deg(heading_angle).item()

            # Decrease the ego vehicles bounding box if it is slow and resolve permanent bounding box
            # intersectinos at collisions.
            # In case of driving increase them for safety.
            extent = self._vehicle.bounding_box.extent
            # Otherwise we would increase the extent of the bounding box of the vehicle
            extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)
            extent.x *= self.config.slow_speed_extent_factor_ego if current_ego_speed < \
                self.config.extent_ego_bbs_speed_threshold else self.config.high_speed_extent_factor_ego_x
            extent.y *= self.config.slow_speed_extent_factor_ego if current_ego_speed < \
                self.config.extent_ego_bbs_speed_threshold else self.config.high_speed_extent_factor_ego_y

            transform = carla.Transform(
                carla.Location(x=location[0].item(), y=location[1].item(), z=location[2].item()))

            ego_bounding_box = carla.BoundingBox(transform.location, extent)
            ego_bounding_box.rotation = carla.Rotation(
                pitch=0, yaw=heading_angle_degrees, roll=0)

            future_bounding_boxes.append(ego_bounding_box)

        self._turn_controller.load_state()
        self._waypoint_planner.load()

        return future_bounding_boxes

    def forecast_walkers_idm(self, actors, ego_vehicle_location, number_of_future_frames):
        """
            Forecast the future locations of pedestrians in the vicinity of the ego vehicle assuming they
            keep their velocity and direction

            Args:
                actors (carla.ActorList): A list of actors in the simulation.
                ego_vehicle_location (carla.Location): The current location of the ego vehicle.
                number_of_future_frames (int): The number of future frames to forecast.

            Returns:
                tuple: A tuple containing two lists:
                    - list: A list of lists, where each inner list contains the future bounding boxes for a pedestrian.
                    - list: A list of IDs for the pedestrians whose locations were forecasted.
            """
        nearby_pedestrians_bbs, nearby_pedestrian_ids = [], []

        # Filter pedestrians within the detection radius
        pedestrians = [
            ped for ped in actors.filter("*walker*")
            if ped.get_location().distance(ego_vehicle_location) < self.config.detection_radius
        ]

        # If no pedestrians are found, return empty lists
        if not pedestrians:
            return nearby_pedestrians_bbs, nearby_pedestrian_ids

        # Extract pedestrian locations, speeds, and directions
        pedestrian_locations = np.array(
            [[ped.get_location().x, ped.get_location().y, ped.get_location().z] for ped in pedestrians])
        pedestrian_speeds = np.array(
            [ped.get_velocity().length() for ped in pedestrians])
        pedestrian_speeds = np.maximum(
            pedestrian_speeds, self.config.min_walker_speed)
        pedestrian_directions = np.array(
            [[ped.get_control().direction.x,
              ped.get_control().direction.y,
              ped.get_control().direction.z] for ped in pedestrians])

        # Calculate future pedestrian locations based on their current locations, speeds, and directions
        future_pedestrian_locations = pedestrian_locations[:, None, :] + np.arange(1, number_of_future_frames + 1)[
            None, :, None] * pedestrian_directions[:, None,
                                                   :] * pedestrian_speeds[:,
                                                                          None,
                                                                          None] / self.config.bicycle_frame_rate

        # Iterate over pedestrians and calculate their future bounding boxes
        for i, ped in enumerate(pedestrians):
            bb, transform = ped.bounding_box, ped.get_transform()
            rotation = carla.Rotation(pitch=bb.rotation.pitch + transform.rotation.pitch,
                                      yaw=bb.rotation.yaw + transform.rotation.yaw,
                                      roll=bb.rotation.roll + transform.rotation.roll)
            extent = bb.extent
            extent.x = max(self.config.pedestrian_minimum_extent,
                           extent.x)  # Ensure a minimum width
            extent.y = max(self.config.pedestrian_minimum_extent,
                           extent.y)  # Ensure a minimum length

            pedestrian_future_bboxes = []
            for j in range(number_of_future_frames):
                location = carla.Location(future_pedestrian_locations[i, j, 0], future_pedestrian_locations[i, j, 1],
                                          future_pedestrian_locations[i, j, 2])

                bounding_box = carla.BoundingBox(location, extent)
                bounding_box.rotation = rotation
                pedestrian_future_bboxes.append(bounding_box)

            nearby_pedestrian_ids.append(ped.id)
            nearby_pedestrians_bbs.append(pedestrian_future_bboxes)

        return nearby_pedestrians_bbs, nearby_pedestrian_ids

    def predict_other_actors_bounding_boxes_idm(self, actor_list, ego_vehicle_location, num_future_frames,
                                                near_lane_change):
        """
            Predict the future bounding boxes of actors for a given number of frames.

            Args:
                actor_list (list): A list of actors (e.g., vehicles) in the simulation.
                ego_vehicle_location (carla.Location): The current location of the ego vehicle.
                num_future_frames (int): The number of future frames to predict.
                near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.

            Returns:
                dict: A dictionary mapping actor IDs to lists of predicted bounding boxes for each future frame.
            """
        predicted_bounding_boxes = {}
        # Filter out nearby actors within the detection radius, excluding the ego vehicle
        nearby_actors = [
            actor for actor in actor_list if actor.id != self._vehicle.id and
            actor.get_location().distance(
                ego_vehicle_location) < self.config.detection_radius
        ]

        # If there are nearby actors, calculate their future bounding boxes
        if nearby_actors:
            # Get the previous control inputs (steering, throttle, brake) for the nearby actors
            previous_controls = [actor.get_control()
                                 for actor in nearby_actors]
            previous_actions = np.array(
                [[control.steer, control.throttle, control.brake] for control in previous_controls])

            # Get the current velocities, locations, and headings of the nearby actors
            velocities = np.array([actor.get_velocity().length()
                                  for actor in nearby_actors])
            locations = np.array([[actor.get_location().x,
                                   actor.get_location().y,
                                   actor.get_location().z] for actor in nearby_actors])
            headings = np.deg2rad(
                np.array([actor.get_transform().rotation.yaw for actor in nearby_actors]))

            # Initialize arrays to store future locations, headings, and velocities
            future_locations = np.empty(
                (num_future_frames, len(nearby_actors), 3), dtype="float")
            future_headings = np.empty(
                (num_future_frames, len(nearby_actors)), dtype="float")
            future_velocities = np.empty(
                (num_future_frames, len(nearby_actors)), dtype="float")

            # Forecast the future locations, headings, and velocities for the nearby actors
            for i in range(num_future_frames):
                locations, headings, velocities = self.vehicle_model.forecast_other_vehicles(
                    locations, headings, velocities, previous_actions)
                future_locations[i] = locations.copy()
                future_velocities[i] = velocities.copy()
                future_headings[i] = headings.copy()

            # Convert future headings to degrees
            future_headings = np.rad2deg(future_headings)

            # Calculate the predicted bounding boxes for each nearby actor and future frame
            for actor_idx, actor in enumerate(nearby_actors):
                predicted_actor_boxes = []

                for i in range(num_future_frames):
                    # Calculate the future location of the actor
                    location = carla.Location(x=future_locations[i, actor_idx, 0].item(),
                                              y=future_locations[i,
                                                                 actor_idx, 1].item(),
                                              z=future_locations[i, actor_idx, 2].item())

                    # Calculate the future rotation of the actor
                    rotation = carla.Rotation(
                        pitch=0, yaw=future_headings[i, actor_idx], roll=0)

                    # Get the extent (dimensions) of the actor's bounding box
                    extent = actor.bounding_box.extent
                    # Otherwise we would increase the extent of the bounding box of the vehicle
                    extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)

                    # Adjust the bounding box size based on velocity and lane change maneuver to adjust for
                    # uncertainty during forecasting
                    s = self.config.high_speed_min_extent_x_other_vehicle_lane_change if near_lane_change \
                        else self.config.high_speed_min_extent_x_other_vehicle
                    extent.x *= self.config.slow_speed_extent_factor_ego if future_velocities[
                        i, actor_idx] < self.config.extent_other_vehicles_bbs_speed_threshold else max(
                        s,
                        self.config.high_speed_min_extent_x_other_vehicle * float(i) / float(num_future_frames))
                    extent.y *= self.config.slow_speed_extent_factor_ego if future_velocities[
                        i, actor_idx] < self.config.extent_other_vehicles_bbs_speed_threshold else max(
                        self.config.high_speed_min_extent_y_other_vehicle,
                        self.config.high_speed_extent_y_factor_other_vehicle * float(i) / float(num_future_frames))

                    # Create the bounding box for the future frame
                    bounding_box = carla.BoundingBox(location, extent)
                    bounding_box.rotation = rotation

                    # Append the bounding box to the list of predicted bounding boxes for this actor
                    predicted_actor_boxes.append(bounding_box)

                # Store the predicted bounding boxes for this actor in the dictionary
                predicted_bounding_boxes[actor.id] = predicted_actor_boxes
        return predicted_bounding_boxes

    def get_nearby_object(self, ego_vehicle_position, all_actors, search_radius):
        """
            Find actors, who's trigger boxes are within a specified radius around the ego vehicle.

            Args:
                ego_vehicle_position (carla.Location): The position of the ego vehicle.
                all_actors (list): A list of all actors.
                search_radius (float): The radius (in meters) around the ego vehicle to search for nearby actors.

            Returns:
                list: A list of actors within the specified search radius.
            """
        nearby_objects = []
        for actor in all_actors:
            try:
                trigger_box_global_pos = actor.get_transform(
                ).transform(actor.trigger_volume.location)
            except:
                continue

            # Convert the vector to a carla.Location for distance calculation
            trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x,
                                                    y=trigger_box_global_pos.y,
                                                    z=trigger_box_global_pos.z)

            # Check if the actor's trigger volume is within the search radius
            if trigger_box_global_pos.distance(ego_vehicle_position) < search_radius:
                nearby_objects.append(actor)

        return nearby_objects

    def _get_control_idm(self, ego_position):
        """
            Get control from IDM
            Returns:
                tuple: A tuple containing the control commands (steer, throttle, brake) and the driving data.
            """

        route_np, route_wp, _, distance_to_next_traffic_light, next_traffic_light, distance_to_next_stop_sign, \
            next_stop_sign = self._waypoint_planner.run_step(ego_position)

        # Get the current speed and target speed
        ego_speed = self.tick_data["speed"]
        target_speed = 6

        # Get the list of vehicles in the scene
        actors = self._world.get_actors()
        vehicles = list(actors.filter("*vehicle*"))

        brake, target_speed = self.get_brake_and_target_speed(
            route_np, distance_to_next_traffic_light, next_traffic_light, distance_to_next_stop_sign,
            next_stop_sign, vehicles, actors, target_speed)

        # Compute throttle and brake control
        throttle, control_brake = self._longitudinal_controller.get_throttle_and_brake(
            brake, target_speed, ego_speed)

        # Compute steering control
        steer = self._get_steer(route_np, ego_position, t_u.preprocess_compass(
            self.tick_data["compass"]), ego_speed)

        # Create the control command
        control = carla.VehicleControl()
        control.steer = steer + self.config.steer_noise * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake or control_brake)

        # Apply brake if the vehicle is stopped to prevent rolling back
        if control.throttle == 0 and ego_speed < self.config.minimum_speed_to_prevent_rolling_back:
            control.brake = 1

        return control
