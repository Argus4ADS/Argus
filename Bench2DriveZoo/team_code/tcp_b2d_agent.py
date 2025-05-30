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
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T

from leaderboard.autoagents import autonomous_agent

from TCP.model import TCP
from TCP.config import GlobalConfig
from team_code.planner import RoutePlanner
from scipy.optimize import fsolve
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as  mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type
from pyquaternion import Quaternion


SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
PLANNER_TYPE = os.environ.get('PLANNER_TYPE', None)
print('*'*10)
print(PLANNER_TYPE)
print('*'*10)

EARTH_RADIUS_EQUA = 6378137.0

def get_entry_point():
    return 'TCPAgent'


class TCPAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steers = deque()
        if IS_BENCH2DRIVE:
            self.save_name = path_to_conf_file.split('+')[-1]
            self.config_path = path_to_conf_file.split('+')[0]
        else:
            self.config_path = path_to_conf_file
            self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.config = GlobalConfig()
        self.net = TCP(self.config)

        self.bevformer_pipeline = []
        for bevformer_pipeline in self.config.bevformer_pipeline:
            if bevformer_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph']:
                self.bevformer_pipeline.append(bevformer_pipeline)
        self.bevformer_pipeline = Compose(self.bevformer_pipeline)

        ckpt = torch.load(self.config_path, map_location="cuda")
        ckpt = ckpt["state_dict"]
        new_state_dict = OrderedDict()
        for key, value in ckpt.items():
            new_key = key.replace("model.","")
            new_state_dict[new_key] = value
        self.net.load_state_dict(new_state_dict, strict = False)
        self.net.cuda()
        self.net.eval()

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0

        self.save_path = None
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

        self.last_steers = deque()
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = self.save_name

            print (string)

            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'rgb').mkdir()
            (self.save_path / 'rgb_front').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()

        # Add for BEV
        self.lidar2img = {
        'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
                                  [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                                  [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
                                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
          'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
                                   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
          'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
                                       [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
                                    [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
         'CAM_BACK':np.array([[-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
                     [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                     [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
                                  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
        'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
                                    [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                    [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        }
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                                 [ 0.  ,  0.  , -1.  , -0.24],
                                 [ 0.  ,  1.  ,  0.  , -1.19],
                              [ 0.  ,  0.  ,  0.  ,  1.  ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.81915204,  0.  , -0.22517331],
                                      [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [-0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
          'CAM_FRONT_RIGHT':np.array([[ 0.57357644, -0.81915204, 0.  ,  0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [ 0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_BACK':np.array([[-1. ,  0.,  0.,  0.  ],
                             [ 0. ,  0., -1., -0.24],
                             [ 0. , -1.,  0., -1.61],
                             [ 0. ,  0.,  0.,  1.  ]]),
     
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.93969262,  0.  , -0.25388956],
                                  [ 0.        ,  0.        , -1.  , -0.24      ],
                                  [-0.93969262, -0.34202014,  0.  , -0.49288953],
                                  [ 0.        ,  0.        ,  0.  ,  1.        ]]),
  
        'CAM_BACK_RIGHT':np.array([[-0.34202014, -0.93969262,  0.  ,  0.25388956],
                                  [ 0.        ,  0.         , -1.  , -0.24      ],
                                  [ 0.93969262, -0.34202014 ,  0.  , -0.49288953],
                                  [ 0.        ,  0.         ,  0.  ,  1.        ]])
        }
        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        
        topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 50.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown

    def _init(self):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            E = EARTH_RADIUS_EQUA
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * E) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * E * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * E * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0
        print(self.lat_ref, self.lon_ref, self.save_name)
        #
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

    def sensors(self):
        sensors =  [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                    },
                # BEV
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                    },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                    },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                    },
                ]
        
        if IS_BENCH2DRIVE:
            sensors += [
                    {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    }]
        return sensors

    def tick(self, input_data):
        self.step += 1

        front_img = cv2.cvtColor(input_data['CAM_FRONT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        front_left_img = cv2.cvtColor(input_data['CAM_FRONT_LEFT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        front_right_img = cv2.cvtColor(input_data['CAM_FRONT_RIGHT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_front =  cv2.cvtColor(input_data['CAM_FRONT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]

        _, front_img = cv2.imencode('.jpg', front_img, encode_param)
        front_img = cv2.imdecode(front_img, cv2.IMREAD_COLOR)

        _, front_left_img = cv2.imencode('.jpg', front_left_img, encode_param)
        front_left_img = cv2.imdecode(front_left_img, cv2.IMREAD_COLOR)

        _, front_right_img = cv2.imencode('.jpg', front_right_img, encode_param)
        front_right_img = cv2.imdecode(front_right_img, cv2.IMREAD_COLOR)
        front_img = front_img[:, 200:1400, :]
        front_left_img = front_left_img[:, :1400, :]
        front_right_img = front_right_img[:, 200:, :]

        rgb = np.concatenate((front_left_img, front_img, front_right_img), axis=1)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        rgb = torch.nn.functional.interpolate(rgb, size=(256, 900), mode='bilinear', align_corners=False)
        rgb = rgb.squeeze(0).permute(1, 2, 0).byte().numpy()
        
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]

        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
   
        # for bev
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img

        result = {
                    'imgs': imgs,
                    'rgb': rgb,
                    'rgb_front': rgb_front,
                    'gps': gps,
                    'speed': speed,
                    'compass': compass,
                    'bev': bev
                }
        pos = self.gps_to_location(result['gps'])
        result['gps'] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result['next_command'] = next_cmd.value
        theta = compass - np.pi/2
        R = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta),  np.cos(theta)]
            ])

        local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
        local_command_point = R.dot(local_command_point)
        result['target_point'] = tuple(local_command_point)

        return result
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        self.tick_data = self.tick(input_data)
        if self.step < self.config.seq_len:
            rgb = self._im_transform(self.tick_data['rgb']).unsqueeze(0)

            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            return control
        
        # FOR BEV
        bevformer_results = {}
        bevformer_results['lidar2img'] = []
        bevformer_results['lidar2cam'] = []
        bevformer_results['img'] = []
        bevformer_results['folder'] = ' '
        bevformer_results['scene_token'] = ' '  
        bevformer_results['frame_idx'] = 0
        bevformer_results['timestamp'] = self.step / 20
        bevformer_results['box_type_3d'], _ = get_box_type('LiDAR')
  
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            bevformer_results['lidar2img'].append(self.lidar2img[cam])
            bevformer_results['lidar2cam'].append(self.lidar2cam[cam])
            bevformer_results['img'].append(self.tick_data['imgs'][cam])
        bevformer_results['lidar2img'] = np.stack(bevformer_results['lidar2img'],axis=0)
        bevformer_results['lidar2cam'] = np.stack(bevformer_results['lidar2cam'],axis=0)
        raw_theta = self.tick_data['compass']   if not np.isnan(self.tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        can_bus = np.zeros(18)
        can_bus[0] = self.tick_data['gps'][1]
        can_bus[1] = -self.tick_data['gps'][0]
        can_bus[3:7] = rotation
        can_bus[7] = self.tick_data['speed']
        can_bus[11] *= -1
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        bevformer_results['can_bus'] = can_bus
        bevformer_results['command'] = -1
        ego2world = np.eye(4)
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2,3] = can_bus[0:2]
        lidar2global = ego2world @ self.lidar2ego
        bevformer_results['l2g_r_mat'] = lidar2global[0:3,0:3]
        bevformer_results['l2g_t'] = lidar2global[0:3,3]
        stacked_imgs = np.stack(bevformer_results['img'],axis=-1)
        bevformer_results['img_shape'] = stacked_imgs.shape
        bevformer_results['ori_shape'] = stacked_imgs.shape
        bevformer_results['pad_shape'] = stacked_imgs.shape
        bevformer_results = self.bevformer_pipeline(bevformer_results)
        self.device="cuda"
        bev_input_data_batch = mm_collate_to_batch_form([bevformer_results], samples_per_gpu=1)
        for key, data in bev_input_data_batch.items():
            if key != 'img_metas':
                if torch.is_tensor(data[0]):
                    data[0] = data[0].to(self.device)
        self.bev_inputs = {
            "img": bev_input_data_batch["img"],
            "img_metas": bev_input_data_batch["img_metas"]
        }
        # END BEV

        gt_velocity = torch.FloatTensor([self.tick_data['speed']]).to('cuda', dtype=torch.float32)
        command = self.tick_data['next_command']
        if command < 0:
            command = 4
        command -= 1
        assert command in [0, 1, 2, 3, 4, 5]
        cmd_one_hot = [0] * 6
        cmd_one_hot[command] = 1
        cmd_one_hot = torch.tensor(cmd_one_hot).view(1, 6).to('cuda', dtype=torch.float32)
        speed = torch.FloatTensor([float(self.tick_data['speed'])]).view(1,1).to('cuda', dtype=torch.float32)
        speed = speed / 12
        rgb = self._im_transform(self.tick_data['rgb']).unsqueeze(0).to('cuda', dtype=torch.float32)

        self.tick_data['target_point'] = [torch.FloatTensor([self.tick_data['target_point'][0]]),
                                        torch.FloatTensor([self.tick_data['target_point'][1]])]
        target_point = torch.stack(self.tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)
        state = torch.cat([speed, target_point, cmd_one_hot], 1)

        pred= self.net(rgb, state, target_point)

        steer_ctrl, throttle_ctrl, brake_ctrl, metadata = self.net.process_action(pred, self.tick_data['next_command'], gt_velocity, target_point)

        self.pred_wp = pred['pred_wp'][0].cpu().numpy()
        self.pred_wp = self.pred_wp[:, [1, 0]]
        steer_traj, throttle_traj, brake_traj, metadata_traj = self.net.control_pid(pred['pred_wp'], gt_velocity, target_point)

        control = carla.VehicleControl()
        if not PLANNER_TYPE: 
            raise 'please set PLANNER_TYPE'
        if PLANNER_TYPE == 'only_traj':
            self.pid_metadata = metadata_traj
            self.pid_metadata['agent'] = 'only_traj'
            control.steer = np.clip(float(steer_traj), -1, 1)
            control.throttle = np.clip(float(throttle_traj), 0, 0.75)
            control.brake = np.clip(float(brake_traj), 0, 1)
        elif PLANNER_TYPE == 'only_ctrl':
            self.pid_metadata = metadata
            self.pid_metadata['agent'] = 'only_ctrl'
            control.steer = np.clip(float(steer_ctrl), -1, 1)
            control.throttle = np.clip(float(throttle_ctrl), 0, 0.75)
            control.brake = np.clip(float(brake_ctrl), 0, 1)
        elif PLANNER_TYPE == 'merge_ctrl_traj' or 'mrp' or 'mr':
            # traj only using when brake
            self.pid_metadata = metadata_traj
            self.pid_metadata['agent'] = PLANNER_TYPE
            self.alpha = 0.5
            control.steer = np.clip(self.alpha*steer_traj + (1-self.alpha)*steer_ctrl, -1, 1)
            control.throttle = np.clip(self.alpha*throttle_traj + (1-self.alpha)*throttle_ctrl, 0, 0.75)
            control.brake = max(np.clip(float(brake_ctrl), 0, 1), np.clip(float(brake_traj), 0, 1))

        self.pid_metadata['steer_ctrl'] = float(steer_ctrl)
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_ctrl'] = float(throttle_ctrl)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_ctrl'] = float(brake_ctrl)
        self.pid_metadata['brake_traj'] = float(brake_traj)

        if abs(control.steer) > 0.07: ## In turning
            speed_threshold = 1.0 ## Avoid stuck during turning
        else:
            speed_threshold = 1.5 ## Avoid pass stop/red light/collision
        if float(self.tick_data['speed']) > speed_threshold:
            max_throttle = 0.05
        else:
            max_throttle = 0.5
        control.throttle = np.clip(control.throttle, a_min=0.0, a_max=max_throttle)

        if control.brake > 0:
            control.brake = 1.0
        if control.brake > 0.5:
            control.throttle = float(0)

        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        if SAVE_PATH is not None and self.step % 1 == 0 and self.__class__ == TCPAgent:
            self.save(self.tick_data)
        return control

    def save(self, tick_data):
        frame = self.step // 10
        Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))
        Image.fromarray(tick_data['rgb_front']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))

        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

        # metric info
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()

    def destroy(self):
        del self.net
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])