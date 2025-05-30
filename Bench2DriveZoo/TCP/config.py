import os

class GlobalConfig:
    """ base architecture configurations """
    # Data
    seq_len = 1 # input timesteps
    pred_len = 4 # future waypoints predicted

    # data root
    root_dir_all = "bench2drive-base/"

    # train_towns = ['town01', 'town03', 'town04',  'town06', ]
    # val_towns = ['town02', 'town05', 'town07', 'town10']
    # train_data, val_data = [], []
    # for town in train_towns:		
    # 	train_data.append(os.path.join(root_dir_all, town))
    # 	train_data.append(os.path.join(root_dir_all, town+'_addition'))
    # for town in val_towns:
    # 	val_data.append(os.path.join(root_dir_all, town+'_val'))

    train_data = './tcp_bench2drive-train.npy'
    val_data = './tcp_bench2drive-val.npy'

    ignore_sides = True # don't consider side cameras
    ignore_rear = True # don't consider rear cameras

    input_resolution = 256

    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    lr = 1e-4 # learning rate

    # Controller
    turn_KP = 0.75
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40 # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40 # buffer size

    max_throttle = 0.75 # upper limit on throttle signal value in dataset
    brake_speed = 0.4 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25 # maximum change in speed input to logitudinal controller


    aim_dist = 4.0 # distance to search around for aim point
    angle_thresh = 0.3 # outlier control detection angle
    dist_thresh = 10 # target point y-distance for outlier filtering


    speed_weight = 0.05
    value_weight = 0.001
    features_weight = 0.05

    rl_ckpt = "roach/log/ckpt_11833344.pth"

    img_aug = True
 
    bev_img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
    class_names = [
        'car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others'
        ]
    file_client_args = dict(backend="disk")
    data_root = "data/bench2drive"
    bevformer_pipeline = [
        dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True,
                file_client_args=file_client_args, img_root=data_root),
        dict(type="NormalizeMultiviewImage", **bev_img_norm_cfg),
        dict(type="PadMultiViewImage", size_divisor=32),
        dict(
            type="MultiScaleFlipAug3D",
            img_scale=(1600, 900),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type="DefaultFormatBundle3D", class_names=class_names, with_label=False
                ),
                dict(
                    type="CustomCollect3D", keys=[
                                                "img",
                                                "timestamp",
                                                "l2g_r_mat",
                                                "l2g_t",
                                                "command",
                                            ]
                ),
            ],
        ),
    ]


    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
