import glob
import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import imageio
import torch
import matplotlib.pyplot as plt
from projectaria_tools.core import calibration
from torch.utils.data import Dataset
import cv2
import copy


class ego4dDataset(Dataset):
    """
    Load Ego4D dataset with only Ego(Aria) images for 3D hand pose estimation
    Reference: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
    """
    def __init__(self, cfg, anno_type, split, transform=None, use_preset=False):
        # TODO: Set parameters based on config file
        self.dataset_root = cfg.DATASET.ROOT
        self.anno_type = anno_type
        self.split = split
        self.use_preset = use_preset
        self.num_joints = cfg.MODEL.NUM_JOINTS                          # Number of joints for single hand
        self.undist_img_dim = np.array(cfg.DATASET.ORIGINAL_IMAGE_SIZE) # [H, W]
        self.valid_kpts_threshold = cfg.DATASET.VIS_THRESHOLD           # Threshold of minimum number of valid kpts in single hand
        self.bbox_padding = cfg.DATASET.BBOX_PADDING                    # Pixels to pad around hand kpts to find bbox 
        self.pixel_std = 200                                            # Pixel std to propose scale factor
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)                # Size of image after affine transformation
        self.heatmap_size = np.array(cfg.MODEL.EXTRA.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.EXTRA.SIGMA
        assert self.split in ['train', 'val', 'test'], f"{self.split} is invalid split. Option: train, val, test"

        self.takes = json.load(open(os.path.join(self.dataset_root, "takes.json")))
        self.hand_anno_dir = os.path.join(self.dataset_root, 'annotations/ego_pose/hand', self.anno_type)
        self.hand_bbox_anno_dir = os.path.join(self.dataset_root, 'annotations/ego_pose/hand/annotation_with_bbox')
        self.cam_pose_dir = os.path.join(self.dataset_root, 'annotations/ego_pose/hand/camera_pose')
        self.undist_img_dir = os.path.join(self.dataset_root, 'aria_undistorted_images', self.anno_type)
        self.all_take_uid = [k[:-5] for k in os.listdir(self.hand_anno_dir)]
        self.take_to_uid = {t['root_dir'] : t['take_uid'] for t in self.takes if t["take_uid"] in self.all_take_uid}
        self.uid_to_take = {uid:take for take, uid in self.take_to_uid.items()}
        self.takes_df = pd.read_csv(os.path.join(self.dataset_root, 'annotations/egoexo_split_latest_train_val_test.csv'))
        self.split_take_dict = self.init_split()
        self.curr_split_take = []
        self.db = self.load_raw_data()
        
        self.transform = transform
        self.joint_mean = np.array([[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
                                    [-3.9501650e+00, -8.6685377e-01,  2.4517984e+01],
                                    [-1.3187613e+01,  1.2967486e+00,  4.7673504e+01],
                                    [-2.2936522e+01,  1.5275195e+00,  7.2566208e+01],
                                    [-3.1109295e+01,  1.9404153e+00,  9.5952751e+01],
                                    [-4.8375599e+01,  4.6012049e+00,  6.7085617e+01],
                                    [-5.9843365e+01,  5.9568534e+00,  9.3948418e+01],
                                    [-5.7148232e+01,  5.7935758e+00,  1.1097713e+02],
                                    [-5.1052166e+01,  4.9937048e+00,  1.2502338e+02],
                                    [-5.1586624e+01,  2.5471370e+00,  7.2120811e+01],
                                    [-6.5926834e+01,  3.0671554e+00,  9.8404510e+01],
                                    [-6.1979191e+01,  2.8341565e+00,  1.1610429e+02],
                                    [-5.4618130e+01,  2.5274558e+00,  1.2917862e+02],
                                    [-4.6503471e+01,  3.3559692e-01,  7.3062035e+01],
                                    [-5.9186893e+01,  2.6649246e-02,  9.6192421e+01],
                                    [-5.6693432e+01, -8.4625520e-02,  1.1205978e+02],
                                    [-5.1260197e+01,  3.4378145e-02,  1.2381713e+02],
                                    [-3.5775276e+01, -1.0368422e+00,  7.0583588e+01],
                                    [-4.3695080e+01, -1.9620019e+00,  8.8694397e+01],
                                    [-4.4897186e+01, -2.6101866e+00,  1.0119468e+02],
                                    [-4.4571526e+01, -3.3564034e+00,  1.1180748e+02]])
        self.joint_std = np.array([[ 0.      ,  0.      ,  0.      ],
                                    [17.266953, 44.075836, 14.078445],
                                    [24.261362, 65.793236, 18.580193],
                                    [25.479671, 74.18796 , 19.767653],
                                    [30.458921, 80.729996, 23.553158],
                                    [21.826715, 45.61571 , 18.80888 ],
                                    [26.570208, 54.434124, 19.955523],
                                    [30.757236, 60.084938, 23.375763],
                                    [35.174015, 64.042404, 31.206692],
                                    [21.586899, 28.31489 , 16.090088],
                                    [29.26384 , 35.83172 , 18.48644 ],
                                    [35.396465, 40.93173 , 26.987226],
                                    [40.40074 , 45.358475, 37.419308],
                                    [20.73408 , 21.591717, 14.190551],
                                    [28.290194, 27.946808, 18.350618],
                                    [34.42277 , 31.388414, 28.024563],
                                    [39.819054, 35.205494, 38.80897 ],
                                    [19.79841 , 29.38799 , 14.820373],
                                    [26.476702, 34.7448  , 20.027615],
                                    [31.811651, 37.06962 , 27.742807],
                                    [36.893555, 38.98199 , 36.001797]])


    def __getitem__(self, idx):
        """
        Return transformed images, 2D heatmap, offset 3D GT, target_weight and metadata
        """
        all_curr_db = copy.deepcopy(self.db[idx])

        all_input = []
        all_curr_2d_kpts = []
        all_hm_2d = []
        all_target_weight = []
        all_curr_3d_kpts_cam_offset = []
        all_vis_flag = []
        all_meta = []
        all_wrist = []
        all_intrinsic = []
        all_joint_view_stat = []

        for curr_db in all_curr_db:
            # Load in undistorted Aria images in extracted view
            img_path = curr_db['image_path']
            img = imageio.imread(img_path, pilmode='RGB')

            # Affine transformation s.t. hand in center of image
            # TODO: Add data augmentation in training (with random flipping and scale factor)
            c, s = curr_db['center'], curr_db['scale']
            r = 0
            trans = get_affine_transform(c, s, r, self.image_size)
            input = cv2.warpAffine(
                img,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)
            # Apply transformation to input images
            if self.transform:
                input = self.transform(input)
            # Affine transformation to 2D hand kpts
            curr_2d_kpts = curr_db['joints_2d']
            curr_2d_kpts = affine_transform(curr_2d_kpts, trans)
            curr_2d_kpts = torch.from_numpy(curr_2d_kpts.astype(np.float32))

            # Offset 3D kpts in camera by hand wrist
            curr_3d_kpts_cam = curr_db['joints_3d'].copy()
            curr_3d_kpts_cam[np.any(curr_3d_kpts_cam==None, axis=1)] = 0
            # Make sure hand wrist stay unchanged
            curr_3d_kpts_cam = curr_3d_kpts_cam * 1000 # m to mm
            curr_3d_kpts_cam_offset = curr_3d_kpts_cam - curr_3d_kpts_cam[0]
            # Normalization
            curr_3d_kpts_cam_offset = (curr_3d_kpts_cam_offset - self.joint_mean) / (self.joint_std + 1e-8)
            curr_3d_kpts_cam_offset[~curr_db['valid_flag']] = None
            curr_3d_kpts_cam_offset = torch.from_numpy(curr_3d_kpts_cam_offset.astype(np.float32))

            # Generate 2D heatmap and corresponding weight
            vis_flag = torch.from_numpy(curr_db['valid_flag'])
            hm_2d, target_weight = self.generate_heatmap(curr_2d_kpts, vis_flag)
            hm_2d = torch.from_numpy(hm_2d)
            target_weight = torch.from_numpy(target_weight)

            # Record meta info for later reprojection
            meta = {
                "hand_wrist": torch.from_numpy(curr_db['joints_3d'][0].astype(np.float32)),
                "intrinsic": torch.from_numpy(curr_db['intrinsic'].astype(np.float32)),
                "joint_view_stat": torch.from_numpy(curr_db['joint_view_stat'].astype(np.float64)),
            }

            all_input.append(input.unsqueeze(1)) # 3 x 1 x H x W
            all_curr_2d_kpts.append(curr_2d_kpts.unsqueeze(0))  # 1, 21, 2
            all_hm_2d.append(hm_2d.unsqueeze(0)) # 1 x joints x H, W
            all_target_weight.append(target_weight.unsqueeze(1))
            all_curr_3d_kpts_cam_offset.append(curr_3d_kpts_cam_offset.unsqueeze(0))  # 1, 21, 3
            all_vis_flag.append(vis_flag.unsqueeze(0))  # 1 x H x W
            all_meta.append(meta)
            all_wrist.append(meta['hand_wrist'].unsqueeze(0)) # 1, 3
            all_intrinsic.append(meta['intrinsic'].unsqueeze(1)) # 1, 3, 3
            all_joint_view_stat.append(meta['joint_view_stat'].unsqueeze(0)) #-

        all_input = torch.cat(all_input, dim = 1)
        all_curr_2d_kpts = torch.cat(all_curr_2d_kpts, dim = 0)
        all_hm_2d = torch.cat(all_hm_2d, dim = 0)
        all_target_weight = torch.cat(all_target_weight, dim = 1)
        all_curr_3d_kpts_cam_offset = torch.cat(all_curr_3d_kpts_cam_offset, dim = 0)
        all_vis_flag = torch.cat(all_vis_flag, dim = 0)
        all_wrist = torch.cat(all_wrist, dim = 0)
        all_intrinsic = torch.cat(all_intrinsic, dim = 0)
        all_joint_view_stat = torch.cat(all_joint_view_stat, dim = 0)

        return all_input, all_curr_2d_kpts, all_hm_2d, all_target_weight, all_curr_3d_kpts_cam_offset, all_vis_flag, all_wrist, all_intrinsic, all_joint_view_stat


    def __len__(self,):
        return len(self.db)
    

    def load_raw_data(self):
        gt_db = []

        if not self.use_preset:
            # Based on split uids, found local take uids that has annotation
            curr_split_uid = self.split_take_dict[self.split]
            # available_curr_split_uid = [t for t in self.all_take_uid if t in curr_split_uid]

            # Instead of following provided train split, use all available takes (not in val/test) as train
            if self.split == 'train':
                available_curr_split_uid = [t for t in self.all_take_uid if t not in self.split_take_dict['val'] + self.split_take_dict['test']]
            else:
                available_curr_split_uid = [t for t in self.all_take_uid if t in curr_split_uid]
            print(f"Trying to use {len(available_curr_split_uid)} takes in {self.split} dataset")
        else:
            if self.split == 'train':
                available_curr_split_uid = [
                    "e3cb859e-73ca-4cef-8c08-296bafdb43cd",
                    "d2218738-2af2-4585-bd1c-af8ad10d7827",
                    "3940a14c-f0ae-4636-8f03-52daa071f084",
                    "989b038e-d46c-4433-b968-87b58a4c7037",
                    "354f076e-079f-440d-bd38-97ddfcd19002",
                    "7014a547-6f84-48cb-bc91-28012c4cce06",
                    "f0ebc587-3687-494d-a707-2a5d52b64719",
                    "c507b073-7bf9-40db-8537-de599b6f6565",
                    "794b3bd3-eac9-4d0d-9789-bd068bff3944",
                    "c53a1199-5ca1-4aa8-ac4e-38227ff44689",
                ]
            elif self.split == 'val':
                available_curr_split_uid = [
                    "6e5211e1-72d8-4032-ba56-b4095c0f2b36",
                    "a8d04142-fc0b-4ad4-acaa-8c17424411ff",
                    "e5beffc8-2cc5-4cc5-9e0e-b22b843aaa4c",
                ]

        # Iterate through all takes from annotation directory and check
        for curr_take_uid in available_curr_split_uid:
            curr_take_name = self.uid_to_take[curr_take_uid]
            # Load annotation, camera pose JSON and image directory
            curr_take_anno_path = os.path.join(self.hand_anno_dir, f"{curr_take_uid}.json")
            curr_take_cam_pose_path = os.path.join(self.cam_pose_dir, f"{curr_take_uid}.json")
            curr_take_img_dir = os.path.join(self.undist_img_dir, curr_take_name)
            curr_take_hand_bbox_dir = os.path.join(self.hand_bbox_anno_dir, f"{curr_take_uid}.json")
            # Check file existence
            if self.split == 'test' and not os.path.exists(curr_take_hand_bbox_dir):
                print(f"[Warning] {curr_take_name} misses necessary files. Skipped for now.")
                continue
            if not os.path.exists(curr_take_anno_path) or not os.path.exists(curr_take_cam_pose_path) or \
                not os.path.exists(curr_take_img_dir):
                print(f"[Warning] {curr_take_name} misses necessary files. Skipped for now.")
                continue
            self.curr_split_take.append(curr_take_name)
            # Load in JSON and image directory
            curr_take_anno = json.load(open(curr_take_anno_path))
            curr_take_cam_pose = json.load(open(curr_take_cam_pose_path))
            curr_take_hand_bbox = json.load(open(curr_take_hand_bbox_dir)) if self.split == 'test' else None
            # Get valid takes info for all frames
            if len(curr_take_anno) > 0 and len(curr_take_anno) <= len(curr_take_cam_pose):
                _, _, aria_mask = self.load_aria_calib(curr_take_name)
                gt_db.extend(self.load_take_raw_data(curr_take_name, 
                                                     curr_take_anno, 
                                                     curr_take_cam_pose,
                                                     curr_take_img_dir,
                                                     aria_mask,
                                                     curr_take_hand_bbox))
        return gt_db

    
    def load_take_raw_data(self, take_name, anno, cam_pose, img_root_dir, aria_mask, curr_take_hand_bbox):

        all_curr_take_db = []

        all_anno = list(anno.items())

        for i in range(0,len(all_anno)-4, 4):
            for hand_idx, hand_name in enumerate(HAND_ORDER):
                curr_take_db = []

                prev_frame_idx = int(all_anno[i][0])
                for frame_idx, curr_frame_anno in all_anno[i:i+4]:
                    if int(frame_idx) - prev_frame_idx > 10:
                        prev_frame_idx = int(frame_idx)
                        continue
                    else:
                        prev_frame_idx = int(frame_idx)
                    # Load in current frame's 2D & 3D annotation and camera parameter
                    curr_hand_3d_kpts, joint_view_stat = self.load_frame_hand_3d_kpts(curr_frame_anno)
                    curr_intri, curr_extri = self.load_frame_cam_pose(frame_idx, cam_pose)
                    # Skip this frame if missing valid data
                    if curr_hand_3d_kpts is None or curr_intri is None or curr_extri is None:
                        continue
                    # Look at each hand in current frame
                    
                    # Get current hand's 3D world kpts
                    start_idx, end_idx = self.num_joints*hand_idx, self.num_joints*(hand_idx+1)
                    one_hand_3d_kpts_world = curr_hand_3d_kpts[start_idx:end_idx]
                    # Skip this hand if the hand wrist (root) is None
                    if np.any(one_hand_3d_kpts_world[0] == None):
                        continue
                    
                    # 3D world to camera (original view)
                    one_hand_3d_kpts_cam = world_to_cam(one_hand_3d_kpts_world, curr_extri)
                    # Camera original to original aria image plane
                    one_hand_2d_kpts_original = cam_to_img(one_hand_3d_kpts_cam, curr_intri)
                    
                    # Get filtered 2D kpts in extracted view
                    one_hand_2d_kpts_extracted = aria_original_to_extracted(one_hand_2d_kpts_original, self.undist_img_dim)
                    one_hand_filtered_2d_kpts, valid_flag = self.one_hand_kpts_valid_check(one_hand_2d_kpts_extracted, aria_mask)
                    # Get filtered 3D kpts in camera original view
                    one_hand_filtered_3d_kpts_cam = one_hand_3d_kpts_cam.copy()
                    one_hand_filtered_3d_kpts_cam[~valid_flag] = None
                    # Regardless of whether 
                    if sum(valid_flag) >= self.valid_kpts_threshold:
                        # Assign original hand wrist 3d kpts back for later offset computation
                        one_hand_filtered_3d_kpts_cam[0] = one_hand_3d_kpts_cam[0]
                        # Get bbox based on 2D GT kpts if not test split, otherwise use provided bbox
                        if self.split == 'test':
                            # Check if provided bbox exists and non-empty
                            cur_frame_bbox_anno = curr_take_hand_bbox[frame_idx][0]
                            if f"hand_bbox_{hand_name}" in cur_frame_bbox_anno.keys() and \
                                len(cur_frame_bbox_anno[f"hand_bbox_{hand_name}"]) == 4:
                                one_hand_bbox_original = xywh2xyxy(cur_frame_bbox_anno[f"hand_bbox_{hand_name}"])
                                one_hand_bbox = aria_original_to_extracted(one_hand_bbox_original.reshape(2,2), (512,512)).flatten()
                            else:
                                continue
                        else:
                            one_hand_bbox = get_bbox_from_kpts(one_hand_filtered_2d_kpts[valid_flag], self.undist_img_dim, self.bbox_padding)
                        center, scale = xyxy2cs(*one_hand_bbox, self.undist_img_dim, self.pixel_std)
                        # Write into db
                        img_path = os.path.join(img_root_dir, f"{int(frame_idx):06d}.jpg")
                        curr_take_db.append({
                            'image_path': img_path,
                            'center': center,
                            'scale': scale,
                            'joints_2d': one_hand_filtered_2d_kpts,
                            'joints_3d': one_hand_filtered_3d_kpts_cam,
                            'valid_flag': valid_flag,
                            'take_name': take_name,
                            'intrinsic': curr_intri,
                            'frame_idx': frame_idx,
                            'hand_name': hand_name,
                            'bbox': one_hand_bbox,
                            'joint_view_stat': joint_view_stat[start_idx:end_idx],
                        })
                if len(curr_take_db) == 4:
                    all_curr_take_db.append(curr_take_db)
        return all_curr_take_db


    def init_split(self):
        # Get tain/val/test df
        train_df = self.takes_df[self.takes_df['split']=='TRAIN']
        val_df = self.takes_df[self.takes_df['split']=='VAL']
        test_df = self.takes_df[self.takes_df['split']=='TEST']
        # Get train/val/test uid
        all_train_uid = list(train_df['take_uid'])
        all_val_uid = list(val_df['take_uid'])
        all_test_uid = list(test_df['take_uid'])
        return {'train':all_train_uid, 'val':all_val_uid, 'test':all_test_uid}


    def load_aria_calib(self, curr_take_name):
        # Load aria calibration model
        capture_name = '_'.join(curr_take_name.split('_')[:-1])
        # Find aria names
        take = [t for t in self.takes if t["root_dir"] == curr_take_name]
        take = take[0]
        ego_cam_names = [
            x["cam_id"] for x in take["capture"]["cameras"] if str(x["is_ego"]).lower() == "true"
        ]
        assert len(ego_cam_names) > 0, "No ego cameras found!"
        if len(ego_cam_names) > 1:
            ego_cam_names = [
                cam for cam in ego_cam_names if cam in take["frame_aligned_videos"].keys()
            ]
            assert len(ego_cam_names) > 0, "No frame-aligned ego cameras found!"
            if len(ego_cam_names) > 1:
                ego_cam_names_filtered = [
                    cam for cam in ego_cam_names if "aria" in cam.lower()
                ]
                if len(ego_cam_names_filtered) == 1:
                    ego_cam_names = ego_cam_names_filtered
            assert (
                len(ego_cam_names) == 1
            ), f"Found too many ({len(ego_cam_names)}) ego cameras: {ego_cam_names}"
        ego_cam_names = ego_cam_names[0]
        # Load aria calibration model
        vrs_path = os.path.join(self.dataset_root, 'captures', capture_name, f'videos/{ego_cam_names}.vrs')
        aria_rgb_calib = get_aria_camera_models(vrs_path)['214-1']
        dst_cam_calib = calibration.get_linear_camera_calibration(512, 512, 150)
        # Generate mask in undistorted aria view
        mask = np.full((1408,1408), 255, dtype=np.uint8)
        undistorted_mask = calibration.distort_by_calibration(mask, dst_cam_calib, aria_rgb_calib)
        undistorted_mask = cv2.rotate(undistorted_mask, cv2.ROTATE_90_CLOCKWISE)
        undistorted_mask = undistorted_mask / 255
        return aria_rgb_calib, dst_cam_calib, undistorted_mask


    def load_frame_hand_3d_kpts(self, frame_anno):
        """
        Return GT 3D hand kpts in world frame & number of views for each joint
        """
        # Check if annotation data exists
        if frame_anno is None or 'annotation3D' not in frame_anno[0].keys():
            return None, None
        
        curr_frame_3d_anno = frame_anno[0]['annotation3D']
        curr_frame_3d_kpts = []
        joints_view_stat = []
        
        # Check if aria 3D annotation is non-empty
        if len(curr_frame_3d_anno) == 0:
            return None, None
        
        # Load 3D annotation for both hands
        for hand in HAND_ORDER:
            for finger, finger_joint_order in FINGER_DICT.items():
                if finger_joint_order:
                    for finger_joint_idx in finger_joint_order:
                        finger_k_json = f"{hand}_{finger}_{finger_joint_idx}"
                        # Load 3D
                        if finger_k_json in curr_frame_3d_anno.keys():
                            curr_frame_3d_kpts.append([curr_frame_3d_anno[finger_k_json]['x'],
                                                       curr_frame_3d_anno[finger_k_json]['y'],
                                                       curr_frame_3d_anno[finger_k_json]['z']])
                            joints_view_stat.append(curr_frame_3d_anno[finger_k_json]['num_views_for_3d'])
                        else:
                            curr_frame_3d_kpts.append([None, None, None])
                            joints_view_stat.append(None)
                else:
                    finger_k_json = f"{hand}_{finger}"
                    # Load 3D
                    if finger_k_json in curr_frame_3d_anno.keys():
                            curr_frame_3d_kpts.append([curr_frame_3d_anno[finger_k_json]['x'],
                                                       curr_frame_3d_anno[finger_k_json]['y'],
                                                       curr_frame_3d_anno[finger_k_json]['z']])
                            joints_view_stat.append(curr_frame_3d_anno[finger_k_json]['num_views_for_3d'])
                    else:
                        curr_frame_3d_kpts.append([None, None, None])
                        joints_view_stat.append(None)
        return np.array(curr_frame_3d_kpts), np.array(joints_view_stat)


    def load_frame_cam_pose(self, frame_idx, cam_pose):
        # Check if current frame has corresponding camera pose
        if frame_idx not in cam_pose.keys() or 'aria01' not in cam_pose[frame_idx].keys():
            return None, None
        # Build camera projection matrix
        curr_cam_intrinsic = np.array(cam_pose[frame_idx]['aria01']['camera_intrinsics'])
        curr_cam_extrinsics = np.array(cam_pose[frame_idx]['aria01']['camera_extrinsics'])
        return curr_cam_intrinsic, curr_cam_extrinsics


    def one_hand_kpts_valid_check(self, kpts, aria_mask):
        """
        Return valid kpts with three checks:
            - Has valid kpts
            - Inbound
            - Visible
        Input:
            kpts: (21,2) raw single 2D hand kpts
            aria_mask: (H,W) binary mask that has same shape as undistorted aria image
        Output:
            new_kpts: (21,2)
            flag: (21,)
        """
        new_kpts = kpts.copy()
        # 1. Check missing annotation kpts
        miss_anno_flag = np.any(kpts == None, axis=1)
        new_kpts[miss_anno_flag] = 0
        # 2. Check out-bound annotation kpts
        x_out_bound = np.logical_or(new_kpts[:,0] < 0, new_kpts[:,0] >= self.undist_img_dim[1])
        y_out_bound = np.logical_or(new_kpts[:,1] < 0, new_kpts[:,1] >= self.undist_img_dim[0])
        out_bound_flag = np.logical_or(x_out_bound, y_out_bound)
        new_kpts[out_bound_flag] = 0
        # 3. Check in-bound but invisible kpts
        invis_flag = aria_mask[new_kpts[:,1].astype(np.int64), new_kpts[:,0].astype(np.int64)] == 0
        # 4. Get valid flag
        invalid_flag = miss_anno_flag + out_bound_flag + invis_flag
        valid_flag = ~invalid_flag
        # 5. Assign invalid kpts as None
        new_kpts[invalid_flag] = None

        return new_kpts, valid_flag
        

    def generate_heatmap(self, joints, vis_flag):
        '''
        Generate 2D heatmap and corresponding weight (invisible: 0, visible: 1)
        Input:
            joint: (num_joints, 2)
            vis_flag: (num_joints,)
        Output:
            target: (num_joints, H_h, H_w)
            target_weight
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = vis_flag

        target = np.zeros((self.num_joints,
                           self.heatmap_size[1],
                           self.heatmap_size[0]),
                           dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            v = target_weight[joint_id]
            if v > 0:
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight


"""
Utils function
TODO: Put in separate file
"""
HAND_ORDER = ['right','left']
FINGER_DICT = {'wrist':None,
               'thumb':[1,2,3,4],
               'index':[1,2,3,4],
               'middle':[1,2,3,4],
               'ring':[1,2,3,4],
               'pinky':[1,2,3,4]}

def get_aria_camera_models(aria_path):
    try:
        from projectaria_tools.core import data_provider

        vrs_data_provider = data_provider.create_vrs_data_provider(aria_path)
        aria_camera_model = vrs_data_provider.get_device_calibration()
        slam_left = aria_camera_model.get_camera_calib("camera-slam-left")
        slam_right = aria_camera_model.get_camera_calib("camera-slam-right")
        rgb_cam = aria_camera_model.get_camera_calib("camera-rgb")
    except Exception as e:
        print(
            f"[Warning] Hitting exception {e}. Fall back to old projectaria_tools ..."
        )
        import projectaria_tools

        vrs_data_provider = projectaria_tools.dataprovider.AriaVrsDataProvider()
        vrs_data_provider.openFile(aria_path)

        aria_stream_id = projectaria_tools.dataprovider.StreamId(214, 1)
        vrs_data_provider.setStreamPlayer(aria_stream_id)
        vrs_data_provider.readFirstConfigurationRecord(aria_stream_id)

        aria_stream_id = projectaria_tools.dataprovider.StreamId(1201, 1)
        vrs_data_provider.setStreamPlayer(aria_stream_id)
        vrs_data_provider.readFirstConfigurationRecord(aria_stream_id)

        aria_stream_id = projectaria_tools.dataprovider.StreamId(1201, 2)
        vrs_data_provider.setStreamPlayer(aria_stream_id)
        vrs_data_provider.readFirstConfigurationRecord(aria_stream_id)

        assert vrs_data_provider.loadDeviceModel()

        aria_camera_model = vrs_data_provider.getDeviceModel()
        slam_left = aria_camera_model.getCameraCalib("camera-slam-left")
        slam_right = aria_camera_model.getCameraCalib("camera-slam-right")
        rgb_cam = aria_camera_model.getCameraCalib("camera-rgb")

    assert slam_left is not None
    assert slam_right is not None
    assert rgb_cam is not None

    return {
        "1201-1": slam_left,
        "1201-2": slam_right,
        "214-1": rgb_cam,
    }

def aria_original_to_extracted(kpts, img_shape=(1408, 1408)):
    """
    Rotate kpts coordinates from original view (hand horizontal) to extracted view (hand vertical)
    img_shape is the shape of original view image
    """
    # assert len(kpts.shape) == 2, "Only can rotate 2D arrays"
    H, _ = img_shape
    none_idx = np.any(kpts == None, axis=1)
    new_kpts = kpts.copy()
    new_kpts[~none_idx, 0] = H - kpts[~none_idx, 1] - 1
    new_kpts[~none_idx, 1] = kpts[~none_idx, 0]
    return new_kpts

def get_bbox_from_kpts(kpts, img_shape, padding=20):
    img_H, img_W = img_shape[:2]
    # Get proposed hand bounding box from hand keypoints
    x1, y1, x2, y2 = (
        kpts[:, 0].min(),
        kpts[:, 1].min(),
        kpts[:, 0].max(),
        kpts[:, 1].max(),
    )

    # Proposed hand bounding box with padding
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (
        np.clip(x1 - padding, 0, img_W - 1),
        np.clip(y1 - padding, 0, img_H - 1),
        np.clip(x2 + padding, 0, img_W - 1),
        np.clip(y2 + padding, 0, img_H - 1),
    )

    # Return bbox result
    return np.array([bbox_x1, bbox_y1, bbox_x2, bbox_y2])

def xyxy2cs(x1, y1, x2, y2, img_shape, pixel_std):
    aspect_ratio = img_shape[1] * 1.0 / img_shape[0]

    center = np.zeros((2), dtype=np.float32)
    center[0] = (x1 + x2) / 2
    center[1] = (y1 + y2) / 2

    w = x2 - x1
    h = y2 - y1

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std],dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def world_to_cam(kpts, extri):
    """
    Transform 3D world kpts to camera coordinate system
    Input:
        kpts: (N,3)
        extri: (3,4) [R|t] 
    Output:
        new_kpts: (N,3)
    """
    none_idx = np.any(kpts == None, axis=1)
    new_kpts = kpts.copy()
    new_kpts[none_idx] = 0
    new_kpts = np.append(new_kpts, np.ones((new_kpts.shape[0], 1)), axis=1).T # (4,N)
    new_kpts = (extri @ new_kpts).T # (N,3)
    new_kpts[none_idx] = None
    return new_kpts

def cam_to_img(kpts, intri):
    """
    Project points in camera coordinate system to image plane
    Input:
        kpts: (N,3)
    Output:
        new_kpts: (N,2)
    """
    none_idx = np.any(kpts == None, axis=1)
    new_kpts = kpts.copy()
    new_kpts[none_idx] = -1
    new_kpts = intri @ new_kpts.T # (3,N)
    new_kpts = new_kpts / new_kpts[2,:]
    new_kpts = new_kpts[:2,:].T
    new_kpts[none_idx] = None
    return new_kpts

def affine_transform(kpts, trans):
    """
    Affine transformation of 2d kpts
    Input:
        kpts: (N,2)
        trans: (3,3)
    Output:
        new_kpts: (N,2)
    """
    if trans.shape[0] == 2:
        trans = np.concatenate((trans, [[0,0,1]]), axis=0)
    new_kpts = kpts.copy()
    none_idx = np.any(new_kpts==None, axis=1)
    new_kpts[none_idx] = 0
    new_kpts = np.append(new_kpts, np.ones((new_kpts.shape[0], 1)), axis=1)
    new_kpts = (trans @ new_kpts.T).T
    new_kpts[none_idx] = None
    return new_kpts

def normalization_stat(loader):
        """
        Compute mean and std for all data
        """
        all_data = []
        for _, _, _, _, pose_3d_gt, _ in tqdm(loader):
            all_data.append(pose_3d_gt)
        all_data = np.concatenate(all_data, axis=0)
        # Calculate mean and std
        mean = np.nanmean(all_data, axis=0)
        std = np.nanstd(all_data, axis=0)
        return mean, std

def xywh2xyxy(bbox):
    """
    Given bbox in [x1,y1,w,h], return bbox corners [x1, y1, x2, y2]
    """
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return np.array([x1,y1,x2,y2])

