"""Image demo script."""
import argparse
import os

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPL
from hybrik.utils.render import SMPLRenderer
from hybrik.utils.vis import get_one_box, vis_smpl_3d
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

det_transform = T.Compose([T.ToTensor()])

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

parser = argparse.ArgumentParser(description='HybrIK Demo')
CKPT = './eval/potter_demo.pth'

parser.add_argument('--gpu',
                    help='gpu',
                    default=0,
                    type=int)
parser.add_argument('--img-dir',
                    help='image folder',
                    default='',
                    type=str)
parser.add_argument('--out-dir',
                    help='output folder',
                    default='',
                    type=str)
opt = parser.parse_args()


cfg_file = './configs/potter_demo.yaml'
cfg = update_config(cfg_file)

bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item*1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

transformation = SimpleTransform3DSMPL(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])

det_model = fasterrcnn_resnet50_fpn(pretrained=True)

hybrik_model = builder.build_sppe(cfg.MODEL)

print(f'Loading model from {CKPT}...')
hybrik_model.load_state_dict(torch.load(CKPT, map_location='cpu'), strict=False)

det_model.cuda(opt.gpu)
hybrik_model.cuda(opt.gpu)
det_model.eval()
hybrik_model.eval()

files = os.listdir(opt.img_dir)

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)

for file in tqdm(files):
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:
        # is an image
        if file[:4] == 'res_':
            continue
        
        # process file name
        img_path = os.path.join(opt.img_dir, file)
        dirname = os.path.dirname(img_path)
        basename = os.path.basename(img_path)

        # Run Detection
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        det_input = det_transform(input_image).to(opt.gpu)
        det_output = det_model([det_input])[0]

        tight_bbox = get_one_box(det_output)  # xyxy

        # Run HybrIK
        pose_input, bbox = transformation.test_transform(img_path, tight_bbox)
        pose_input = pose_input.to(opt.gpu)[None, :, :, :]
        pose_output = hybrik_model(pose_input)

        # Visualization
        image = input_image
        img_size = (image.shape[0], image.shape[1])
        focal = np.array([1000, 1000])
        bbox = xyxy2xywh(bbox)
        princpt = [bbox[0], bbox[1]]

        renderer = SMPLRenderer(faces=hybrik_model.smpl.faces,
                                img_size=img_size, focal=focal,
                                princpt=princpt)

        transl = pose_output.transl.detach().cpu().numpy().squeeze()
        transl[2] = transl[2] * 256 / bbox[2]

        image_vis = vis_smpl_3d(
            pose_output, image, cam_root=transl,
            f=focal, c=princpt, renderer=renderer)

        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)


        res_path = os.path.join(opt.out_dir, basename)
        cv2.imwrite(res_path, image_vis)


        # mesh_obj = pose_output.pred_vertices.detach().cpu().numpy().squeeze()
        # face = hybrik_model.smpl.faces
        # mesh_path = os.path.join(opt.out_dir, basename+f'mesh-{id}.obj')
        # save_obj(mesh_obj, face, mesh_path,)